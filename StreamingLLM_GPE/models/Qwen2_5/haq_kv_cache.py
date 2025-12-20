"""
HAQ-KV: Head-Aware Quantized Key-Value Cache
Hi-KV (Hierarchical Knowledge Virtualization)

核心创新: "Not all memories require the same precision."
- Retrieval Heads: 低精度量化 (INT4/INT2) - 长期语义记忆
- Induction/Local Heads: 高精度 (FP16/BF16) - 语法缓冲

相比传统eviction方法，量化方法可以:
1. 存储4-8倍的历史长度（相同显存）
2. 避免因丢弃tokens导致的准确率崩盘
3. 保持检索头的语义一致性（方向信息）
"""
import torch
from typing import List, Tuple, Optional, Dict
import sys
import os

# Add parent directory to path before importing
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, '../../..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from StreamingLLM_GPE.utils.head_analyzer import HeadAnalyzer
from StreamingLLM_GPE.utils.quantization import HeadAwareQuantizer

try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    from transformers.cache_utils import Cache as DynamicCache


class HAQKVCache(DynamicCache):
    """
    HAQ-KV: Head-Aware Quantized Key-Value Cache
    
    核心特性:
    1. 异构量化: 根据head类型分配不同精度
    2. 无限上下文感知: 通过量化存储更多历史
    3. 语义保留: 检索头保留方向信息而非精确值
    4. 语法保护: 归纳头和局部头保持高精度
    5. [FIX] 自动对齐不同长度的Head并生成Mask
    """

    def __init__(
        self,
        head_analyzer: HeadAnalyzer,
        group_tracker: Optional[object] = None,  # 已废弃，保留仅为兼容性
        total_budget: int = 2048,
        sink_tokens: int = 128,
        adaptive: bool = True,
        device: str = 'cuda',
        # HAQ-KV 特定参数
        retrieval_bits: int = 4,      # Retrieval Heads量化位数 (2, 4, 8)
        induction_bits: int = 16,     # Induction Heads量化位数 (16=FP16)
        local_bits: int = 16,         # Local Heads量化位数 (16=FP16)
        use_quantization: bool = True # 是否启用量化
    ):
        super().__init__()
        self.head_analyzer = head_analyzer
        self.total_budget = total_budget  # 基于FP16的预算 (Token数)
        self.sink_tokens = sink_tokens
        self.adaptive = adaptive
        self.device = device
        self.use_quantization = use_quantization

        # 保存配置供计算使用
        self.retrieval_bits = retrieval_bits
        self.induction_bits = induction_bits
        self.local_bits = local_bits

        # 初始化量化器
        if use_quantization:
            self.quantizer = HeadAwareQuantizer(
                retrieval_bits=retrieval_bits,
                induction_bits=induction_bits,
                local_bits=local_bits,
                device=device
            )
        else:
            self.quantizer = None

        # 初始化 _seen_tokens
        if not hasattr(self, "_seen_tokens"):
            self._seen_tokens = 0

        # 存储量化信息: {layer_idx: {head_idx: quant_info}}
        self.quantization_info: Dict[int, Dict[int, dict]] = {}

        # Padding masks: {layer_idx: Tensor}
        self.padding_masks: Dict[int, torch.Tensor] = {}

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新KV cache，使用异构量化策略
        """
        bsz, num_heads, seq_len, head_dim = key_states.shape

        # 维护 _seen_tokens
        if layer_idx == 0:
            self._seen_tokens += seq_len

        # 获取head类型
        head_types = self._get_head_types(layer_idx, num_heads)

        # 1. 恢复现有 Cache (反量化) 以便拼接
        if len(self.key_cache) > layer_idx:
            existing_keys = self.key_cache[layer_idx]
            existing_values = self.value_cache[layer_idx]

            if self.use_quantization and layer_idx in self.quantization_info:
                existing_keys, existing_values = self._dequantize_cache(
                    existing_keys, existing_values, layer_idx, head_types
                )

            combined_keys = torch.cat([existing_keys, key_states], dim=2)
            combined_values = torch.cat([existing_values, value_states], dim=2)
        else:
            combined_keys = key_states
            combined_values = value_states

        current_length = combined_keys.shape[2]

        # 2. 判断是否触发压缩逻辑
        # [CRITICAL LOGIC CHANGE]
        # 我们使用 "基础预算 (total_budget)" 作为触发阈值。
        # 为什么？因为 Local Heads 需要保持短窗口 (total_budget)。
        # 如果我们等到 "Effective Budget" (比如 8000) 才压缩，Local Heads 就会长达 8000，
        # 这浪费了显存，本该留给 Retrieval Heads 的空间被挤占了。
        # 所以：只要超过 total_budget，我们就进入 _compress，它会自动让 Retrieval 变长，Local 变短。
        trigger_threshold = self.total_budget

        if current_length <= trigger_threshold:
            # 长度还在基础预算内，不需要Eviction
            # 但我们需要生成 Mask (全0表示有效)，并可能进行量化存储

            mask = torch.zeros(
                (bsz, num_heads, 1, current_length),
                device=key_states.device, dtype=key_states.dtype
            )
            self.padding_masks[layer_idx] = mask

            if self.use_quantization:
                # 存储量化版
                quantized_keys, quantized_values, quant_info = self.quantizer.quantize_by_head_type(
                    combined_keys, combined_values, head_types, layer_idx
                )
                self.quantization_info[layer_idx] = quant_info

                if len(self.key_cache) <= layer_idx:
                    self.key_cache.append(quantized_keys)
                    self.value_cache.append(quantized_values)
                else:
                    self.key_cache[layer_idx] = quantized_keys
                    self.value_cache[layer_idx] = quantized_values

                # 返回反量化后的 FP16 用于当前计算
                return self._dequantize_cache(quantized_keys, quantized_values, layer_idx, head_types)
            else:
                # 纯 FP16 存储
                if len(self.key_cache) <= layer_idx:
                    self.key_cache.append(combined_keys)
                    self.value_cache.append(combined_values)
                else:
                    self.key_cache[layer_idx] = combined_keys
                    self.value_cache[layer_idx] = combined_values
                return combined_keys, combined_values

        # 3. 超出预算，执行 "压缩 + 量化 + 对齐"
        # 这里的 compressed 过程会根据 Head 类型应用不同的 budget (Retrieval 长，Local 短)
        compressed_keys, compressed_values = self._compress_with_quantization(
            combined_keys, combined_values, layer_idx, head_types, cache_kwargs
        )

        return compressed_keys, compressed_values

    def _calculate_effective_budget(self, layer_idx: int, num_heads: int) -> int:
        """
        [REAL FIX] 精确计算有效预算

        计算逻辑:
        1. 计算 FP16 下的总比特数: total_budget * num_heads * 16
        2. 计算当前配置下的平均每 Token 比特数
        3. 真实容量 = 总比特数 / 平均比特数
        """
        if not self.use_quantization:
            return self.total_budget

        head_types = self._get_head_types(layer_idx, num_heads)

        total_bits_capacity = self.total_budget * num_heads * 16  # 基准容量 (bits)
        current_bits_sum = 0

        for head_idx in range(num_heads):
            h_type = head_types.get(head_idx, 'local')
            if h_type == 'retrieval':
                current_bits_sum += self.retrieval_bits
            elif h_type == 'induction':
                current_bits_sum += self.induction_bits
            else:
                current_bits_sum += self.local_bits

        # 防止除零
        if current_bits_sum == 0:
            return self.total_budget

        # 平均每个 Head 每 Token 占用的 bits
        avg_bits_per_token = current_bits_sum / num_heads

        # 扩容倍数
        expansion_ratio = 16.0 / avg_bits_per_token

        effective_budget = int(self.total_budget * expansion_ratio)
        return effective_budget

    def _get_head_types(self, layer_idx: int, num_heads: int) -> dict:
        """获取head类型"""
        head_types = {}
        for head_idx in range(num_heads):
            # 默认为 'local' 更安全，防止 bug
            head_type = self.head_analyzer.head_profiles.get(
                layer_idx, {}
            ).get(head_idx, 'local')
            head_types[head_idx] = head_type
        return head_types

    def _compress_with_quantization(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        head_types: dict,
        cache_kwargs: Optional[dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        压缩并量化KV cache (包含 Padding 逻辑)
        """
        bsz, num_heads, seq_len, head_dim = key_states.shape

        attention_scores = None
        if cache_kwargs is not None:
            attention_scores = cache_kwargs.get("attention_scores", None)

        compressed_keys_list = []
        compressed_values_list = []

        # 获取精确计算的 Effective Budget，用于分配给 Retrieval Head
        effective_budget_limit = self._calculate_effective_budget(layer_idx, num_heads)

        # 1. 独立压缩每个 Head
        for head_idx in range(num_heads):
            head_type = head_types.get(head_idx, 'local')
            head_key = key_states[:, head_idx, :, :]
            head_value = value_states[:, head_idx, :, :]

            head_attn_score = None
            if attention_scores is not None and attention_scores.dim() == 4:
                if attention_scores.shape[1] == num_heads:
                    if attention_scores.shape[2] == 1:
                        head_attn_score = attention_scores[:, head_idx, 0, :]
                    else:
                        head_attn_score = attention_scores[:, head_idx, :, :].sum(dim=1)

            # 策略核心：异构预算
            if head_type == 'retrieval':
                # Retrieval Head 可以使用更大的预算 (接近 Effective Budget)
                # 留一些余量防止单一 Head 占满
                retrieval_budget = int(effective_budget_limit * 0.8)
                comp_key, comp_value = self._compress_retrieval_head(
                    head_key, head_value, retrieval_budget, head_attn_score
                )
            elif head_type == 'local':
                # Local Head 严格遵守基础预算，保持短窗口
                comp_key, comp_value = self._compress_recent(
                    head_key, head_value, self.total_budget
                )
            else:  # induction
                # Induction Head 也遵守基础预算
                comp_key, comp_value = self._compress_induction_head(
                    head_key, head_value, self.total_budget, head_attn_score
                )

            compressed_keys_list.append(comp_key)
            compressed_values_list.append(comp_value)

        # 2. [CRITICAL FIX] 对齐 Padding
        # 找出所有 Head 中最长的长度 (通常是 Retrieval Head 的长度)
        max_len = max(k.size(1) for k in compressed_keys_list)

        padded_keys_list = []
        padded_values_list = []

        # 记录每个 Head 的真实长度，用于生成 Mask
        valid_lengths = []

        for k, v in zip(compressed_keys_list, compressed_values_list):
            curr_len = k.size(1)
            valid_lengths.append(curr_len)

            if curr_len < max_len:
                # 补 0 对齐
                pad_len = max_len - curr_len
                k_padded = torch.nn.functional.pad(k, (0, 0, 0, pad_len), value=0.0)
                v_padded = torch.nn.functional.pad(v, (0, 0, 0, pad_len), value=0.0)
                padded_keys_list.append(k_padded)
                padded_values_list.append(v_padded)
            else:
                padded_keys_list.append(k)
                padded_values_list.append(v)

        # 3. Stack (现在形状一致了，不会报错)
        compressed_keys = torch.stack(padded_keys_list, dim=1) # [bsz, num_heads, max_len, head_dim]
        compressed_values = torch.stack(padded_values_list, dim=1)

        # 4. 生成 Mask
        dtype = compressed_keys.dtype
        device = compressed_keys.device
        min_value = torch.finfo(dtype).min
        mask = torch.zeros((bsz, num_heads, 1, max_len), device=device, dtype=dtype)

        for i, valid_len in enumerate(valid_lengths):
            if valid_len < max_len:
                mask[:, i, :, valid_len:] = min_value

        self.padding_masks[layer_idx] = mask

        # 5. 量化存储
        if self.use_quantization:
            quantized_keys, quantized_values, quant_info = self.quantizer.quantize_by_head_type(
                compressed_keys, compressed_values, head_types, layer_idx
            )
            self.quantization_info[layer_idx] = quant_info

            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(quantized_keys)
                self.value_cache.append(quantized_values)
            else:
                self.key_cache[layer_idx] = quantized_keys
                self.value_cache[layer_idx] = quantized_values

            # 返回未量化的 Padding 版用于当前 Attention
            return compressed_keys, compressed_values
        else:
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(compressed_keys)
                self.value_cache.append(compressed_values)
            else:
                self.key_cache[layer_idx] = compressed_keys
                self.value_cache[layer_idx] = compressed_values

            return compressed_keys, compressed_values

    def _compress_retrieval_head(self, key, value, budget, importance_scores=None):
        bsz, seq_len, head_dim = key.shape
        if seq_len <= budget:
            return key, value

        sink_size = min(self.sink_tokens, budget // 4)
        sink_keys = key[:, :sink_size, :]
        sink_values = value[:, :sink_size, :]

        remaining_keys = key[:, sink_size:, :]
        remaining_values = value[:, sink_size:, :]

        if importance_scores is not None:
            scores = importance_scores[:, sink_size:]
        else:
            scores = torch.norm(remaining_keys, p=2, dim=-1)

        remaining_budget = budget - sink_size
        k = min(remaining_budget, scores.size(1))

        if k <= 0:
             return sink_keys, sink_values

        _, top_indices = torch.topk(scores, k, dim=1)
        top_indices, _ = torch.sort(top_indices, dim=1)

        selected_keys = torch.gather(remaining_keys, 1, top_indices.unsqueeze(-1).expand(-1, -1, head_dim))
        selected_values = torch.gather(remaining_values, 1, top_indices.unsqueeze(-1).expand(-1, -1, head_dim))

        compressed_key = torch.cat([sink_keys, selected_keys], dim=1)
        compressed_value = torch.cat([sink_values, selected_values], dim=1)
        return compressed_key, compressed_value

    def _compress_recent(self, key, value, budget):
        bsz, seq_len, head_dim = key.shape
        if seq_len <= budget:
            return key, value

        sink_size = min(self.sink_tokens, budget // 2)
        window_size = budget - sink_size

        sink_keys = key[:, :sink_size, :]
        sink_values = value[:, :sink_size, :]
        recent_keys = key[:, -window_size:, :]
        recent_values = value[:, -window_size:, :]

        compressed_key = torch.cat([sink_keys, recent_keys], dim=1)
        compressed_value = torch.cat([sink_values, recent_values], dim=1)
        return compressed_key, compressed_value

    def _compress_induction_head(self, key, value, budget, importance_scores=None):
        bsz, seq_len, head_dim = key.shape
        if seq_len <= budget:
            return key, value

        sink_size = min(self.sink_tokens, budget // 3)
        sink_keys = key[:, :sink_size, :]
        sink_values = value[:, :sink_size, :]

        window_size = budget // 3
        recent_keys = key[:, -window_size:, :]
        recent_values = value[:, -window_size:, :]

        remaining_budget = budget - sink_size - window_size
        middle_start = sink_size
        middle_end = seq_len - window_size

        if middle_end > middle_start and remaining_budget > 0:
            middle_keys = key[:, middle_start:middle_end, :]
            middle_values = value[:, middle_start:middle_end, :]

            if importance_scores is not None:
                scores = importance_scores[:, middle_start:middle_end]
            else:
                scores = torch.norm(middle_keys, p=2, dim=-1)

            k = min(remaining_budget, scores.size(1))

            if k > 0:
                _, top_indices = torch.topk(scores, k, dim=1)
                top_indices, _ = torch.sort(top_indices, dim=1)

                selected_keys = torch.gather(middle_keys, 1, top_indices.unsqueeze(-1).expand(-1, -1, head_dim))
                selected_values = torch.gather(middle_values, 1, top_indices.unsqueeze(-1).expand(-1, -1, head_dim))
            else:
                selected_keys = torch.empty(bsz, 0, head_dim, device=key.device, dtype=key.dtype)
                selected_values = torch.empty(bsz, 0, head_dim, device=value.device, dtype=value.dtype)
        else:
            selected_keys = torch.empty(bsz, 0, head_dim, device=key.device, dtype=key.dtype)
            selected_values = torch.empty(bsz, 0, head_dim, device=value.device, dtype=value.dtype)

        compressed_key = torch.cat([sink_keys, selected_keys, recent_keys], dim=1)
        compressed_value = torch.cat([sink_values, selected_values, recent_values], dim=1)
        return compressed_key, compressed_value

    def _dequantize_cache(self, quantized_keys, quantized_values, layer_idx, head_types):
        if not self.use_quantization or layer_idx not in self.quantization_info:
            return quantized_keys, quantized_values
        quant_info = self.quantization_info[layer_idx]
        return self.quantizer.dequantize_by_head_type(quantized_keys, quantized_values, quant_info, head_types)

    def get_seq_length(self, layer_idx: Optional[int] = None) -> int:
        return self._seen_tokens

    def get_max_length(self) -> Optional[int]:
        if not self.use_quantization:
            return self.total_budget
        # 使用精确计算的有效预算作为最大长度参考
        # 注意：这里假设所有head平均分布，仅作参考
        return self._calculate_effective_budget(0, 1) # 简化调用

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = None) -> int:
        return self._seen_tokens

    def get_memory_usage(self) -> float:
        total_elements = 0
        for layer_idx, (key_cache, value_cache) in enumerate(zip(self.key_cache, self.value_cache)):
            if key_cache is None: continue
            if self.use_quantization and layer_idx in self.quantization_info:
                quant_info = self.quantization_info[layer_idx]
                bsz, num_heads, seq_len, head_dim = key_cache.shape
                for head_idx in range(num_heads):
                    info = quant_info.get(head_idx, {'bits': 16})
                    bits = info.get('bits', 16)
                    bytes_per_element = bits / 8 if bits < 16 else 2
                    head_elements = seq_len * head_dim * 2
                    total_elements += head_elements * bytes_per_element / 2
            else:
                total_elements += key_cache.numel() + value_cache.numel()
        return total_elements * 2 / (1024 ** 3)