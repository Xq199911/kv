"""
Head-Aware Dynamic KV Cache

基于head功能特性的动态KV cache压缩和驱逐
根据attention head的功能特性动态分配KV cache预算
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

# Import utils modules first
from StreamingLLM_GPE.utils.head_analyzer import HeadAnalyzer

# 优先使用 transformers 的标准 DynamicCache
try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    # 兼容旧版本 transformers
    from transformers.cache_utils import Cache as DynamicCache

class HeadAwareDynamicCache(DynamicCache):
    """
    Head-Aware Dynamic KV Cache
    (修复版：增加了 Padding Mask 支持，修复 Attention 对齐问题)

    核心特性：
    1. 基于head功能特性分配预算
    2. 自适应预算调整
    3. [FIX] 自动生成 Padding Mask 防止 Attention 关注填充的 0
    4. [FIX] 默认策略改为 Local 防止上下文崩溃
    5. [FIX] Induction Heads 采用 Hybrid 策略 (Sink + Heavy Hitter + Recent)
    """

    def __init__(
        self,
        head_analyzer: HeadAnalyzer,
        group_tracker: Optional[object] = None,  # 保留参数以兼容，但不再使用
        total_budget: int = 2048,
        sink_tokens: int = 4,
        adaptive: bool = True,
        device: str = 'cuda'
    ):
        """
        Args:
            head_analyzer: Head分析器
            group_tracker: 已废弃，保留仅为兼容性
            total_budget: 总预算（tokens per layer）
            sink_tokens: Attention sink tokens数量
            adaptive: 是否使用自适应分配
            device: 设备
        """
        super().__init__()
        self.head_analyzer = head_analyzer
        self.total_budget = total_budget
        self.sink_tokens = sink_tokens
        self.adaptive = adaptive
        self.device = device

        # 初始化 _seen_tokens (如果父类没有初始化)
        if not hasattr(self, "_seen_tokens"):
            self._seen_tokens = 0

        # 存储每个head的独立cache（可选，用于细粒度控制）
        self.head_caches: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}

        # [新增] 用于存储每一层的 Padding Mask
        # Key: layer_idx, Value: Tensor [bsz, num_heads, 1, seq_len]
        self.padding_masks: Dict[int, torch.Tensor] = {}

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新KV cache，根据head特性进行压缩
        Args:
            key_states: [batch_size, num_heads, seq_len, head_dim]
            value_states: [batch_size, num_heads, seq_len, head_dim]
            layer_idx: 层索引
            cache_kwargs: 缓存参数 (可能包含 attention_scores)
        Returns:
            (compressed_key_states, compressed_value_states)
        """
        bsz, num_heads, seq_len, head_dim = key_states.shape
        # 维护 _seen_tokens (对于 Position Embedding 至关重要)
        if layer_idx == 0:
            self._seen_tokens += seq_len

        # 如果有现有cache，先合并
        if len(self.key_cache) > layer_idx:
            # 合并新tokens到现有cache
            existing_keys = self.key_cache[layer_idx]
            existing_values = self.value_cache[layer_idx]

            # 拼接
            combined_keys = torch.cat([existing_keys, key_states], dim=2)
            combined_values = torch.cat([existing_values, value_states], dim=2)
        else:
            combined_keys = key_states
            combined_values = value_states

        # 检查是否需要压缩
        current_length = combined_keys.shape[2]

        if current_length <= self.total_budget:
            # [新增] 生成全 0 的 Mask (表示所有 Token 都是有效的)
            # Mask 形状: [bsz, num_heads, 1, current_length]
            mask = torch.zeros(
                (bsz, num_heads, 1, current_length),
                device=key_states.device,
                dtype=key_states.dtype
            )
            self.padding_masks[layer_idx] = mask

            # 不需要压缩，直接更新
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(combined_keys)
                self.value_cache.append(combined_values)
            else:
                self.key_cache[layer_idx] = combined_keys
                self.value_cache[layer_idx] = combined_values

            return combined_keys, combined_values

        # 需要压缩：根据head特性进行压缩
        # 尝试从 cache_kwargs 获取 attention_scores
        attention_scores = None
        if cache_kwargs is not None:
            attention_scores = cache_kwargs.get("attention_scores", None)

        compressed_keys, compressed_values = self._compress_by_head(
            combined_keys, combined_values, layer_idx, attention_scores
        )

        # 更新cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(compressed_keys)
            self.value_cache.append(compressed_values)
        else:
            self.key_cache[layer_idx] = compressed_keys
            self.value_cache[layer_idx] = compressed_values

        return compressed_keys, compressed_values

    def _compress_by_head(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        attention_scores: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据head特性进行压缩
        """
        bsz, num_heads, seq_len, head_dim = key_states.shape

        # 获取每个head的预算
        head_budgets = self.head_analyzer.get_all_head_budgets(
            layer_idx, self.total_budget, self.adaptive
        )

        compressed_keys_list = []
        compressed_values_list = []

        for head_idx in range(num_heads):
            head_key = key_states[:, head_idx, :, :]  # [bsz, seq_len, head_dim]
            head_value = value_states[:, head_idx, :, :]

            # 提取该 head 的 attention scores (如果存在)
            head_attn_score = None
            if attention_scores is not None:
                # 参考H2O的实现，更宽松的维度检查
                if attention_scores.dim() == 4 and attention_scores.shape[1] == num_heads:
                    if attention_scores.shape[2] == 1:
                        head_attn_score = attention_scores[:, head_idx, 0, :]  # [bsz, kv_len]
                    else:
                        head_attn_score = attention_scores[:, head_idx, :, :].sum(dim=1)  # [bsz, kv_len]

                    if head_attn_score.shape[1] != seq_len:
                        head_attn_score = None

            budget = head_budgets.get(head_idx, self.total_budget // num_heads)

            # [CRITICAL FIX 1]: 默认 Head 类型改为 'local'
            # 之前的默认 'induction' 会导致分析失效时采用均匀采样，这在语言模型中是毁灭性的。
            # 'local' 是最安全的 fallback，保证至少像 sliding window 一样工作。
            head_type = self.head_analyzer.head_profiles.get(layer_idx, {}).get(head_idx, 'local')

            # 根据head类型选择压缩策略
            if head_type == 'retrieval':
                # Retrieval heads: 保留重要性高的tokens
                comp_key, comp_value = self._compress_by_importance(
                    head_key, head_value, budget, head_attn_score
                )
            elif head_type == 'local':
                # Local heads: 只保留最近的tokens
                comp_key, comp_value = self._compress_recent(
                    head_key, head_value, budget
                )
            else:  # induction
                # [CRITICAL FIX 2]: Induction heads 逻辑升级
                # 改为 Hybrid 策略：保留 Sink + Recent Window + High Importance
                comp_key, comp_value = self._compress_pattern(
                    head_key, head_value, budget, head_attn_score
                )
            compressed_keys_list.append(comp_key)
            compressed_values_list.append(comp_value)

        # ==================== [统一所有head的长度并生成 Mask] ====================
        target_len = self.total_budget
        padded_keys_list = []
        padded_values_list = []
        valid_lengths = []

        for k, v in zip(compressed_keys_list, compressed_values_list):
            curr_len = k.size(1)
            valid_lengths.append(curr_len)

            if curr_len < target_len:
                pad_len = target_len - curr_len
                k_padded = torch.nn.functional.pad(k, (0, 0, 0, pad_len), value=0.0)
                v_padded = torch.nn.functional.pad(v, (0, 0, 0, pad_len), value=0.0)
                padded_keys_list.append(k_padded)
                padded_values_list.append(v_padded)
            elif curr_len > target_len:
                k_padded = k[:, :target_len, :]
                v_padded = v[:, :target_len, :]
                padded_keys_list.append(k_padded)
                padded_values_list.append(v_padded)
            else:
                padded_keys_list.append(k)
                padded_values_list.append(v)

        compressed_keys = torch.stack(padded_keys_list, dim=1)
        compressed_values = torch.stack(padded_values_list, dim=1)

        dtype = compressed_keys.dtype
        device = compressed_keys.device
        min_value = torch.finfo(dtype).min

        mask = torch.zeros((bsz, num_heads, 1, target_len), device=device, dtype=dtype)

        for i, valid_len in enumerate(valid_lengths):
            if valid_len < target_len:
                mask[:, i, :, valid_len:] = min_value

        self.padding_masks[layer_idx] = mask

        return compressed_keys, compressed_values

    def _compress_by_importance(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        budget: int,
        importance_scores: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        基于重要性压缩（用于retrieval heads）
        """
        bsz, seq_len, head_dim = key.shape
        if seq_len <= budget:
            return key, value

        # 1. 保留sink tokens
        sink_size = min(self.sink_tokens, budget // 2)
        sink_keys = key[:, :sink_size, :]
        sink_values = value[:, :sink_size, :]

        # 2. 准备剩余部分的 keys 和 values
        remaining_keys = key[:, sink_size:, :]
        remaining_values = value[:, sink_size:, :]

        # 3. 计算重要性分数
        if importance_scores is not None:
            scores = importance_scores[:, sink_size:]
        else:
            scores = torch.norm(remaining_keys, p=2, dim=-1)

        # 选择top-k重要的tokens
        remaining_budget = budget - sink_size
        k = min(remaining_budget, scores.size(1))
        _, top_indices = torch.topk(scores, k, dim=1)
        top_indices, _ = torch.sort(top_indices, dim=1)

        selected_keys = torch.gather(remaining_keys, 1, top_indices.unsqueeze(-1).expand(-1, -1, head_dim))
        selected_values = torch.gather(remaining_values, 1, top_indices.unsqueeze(-1).expand(-1, -1, head_dim))

        compressed_key = torch.cat([sink_keys, selected_keys], dim=1)
        compressed_value = torch.cat([sink_values, selected_values], dim=1)

        return compressed_key, compressed_value

    def _compress_recent(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        budget: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        保留最近的tokens（用于local heads），同时必须保留 Sinks！
        """
        bsz, seq_len, head_dim = key.shape
        if seq_len <= budget:
            return key, value

        sink_size = self.sink_tokens
        if budget <= sink_size:
            return key[:, :budget, :], value[:, :budget, :]

        sink_keys = key[:, :sink_size, :]
        sink_values = value[:, :sink_size, :]

        recent_budget = budget - sink_size
        recent_keys = key[:, -recent_budget:, :]
        recent_values = value[:, -recent_budget:, :]

        compressed_key = torch.cat([sink_keys, recent_keys], dim=1)
        compressed_value = torch.cat([sink_values, recent_values], dim=1)
        return compressed_key, compressed_value

    def _compress_pattern(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        budget: int,
        importance_scores: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        [FIXED Strategy] 混合保留策略：Sinks + Heavy Hitters + Recent Window

        之前的均匀采样(Uniform Sampling)会破坏N-gram结构，导致Induction Head失效。
        现在的策略：
        1. Sinks: 始终保留最前面的几个token (Attention Sink)
        2. Recent Window: 强制保留最近的一部分token (保证上下文连贯性)
        3. Heavy Hitters: 在中间区域保留Attention分数最高的token (捕捉长距离依赖)
        """
        bsz, seq_len, head_dim = key.shape
        if seq_len <= budget:
            return key, value

        # 1. 保留 Sink Tokens (不变)
        sink_size = min(self.sink_tokens, budget // 4)
        sink_keys = key[:, :sink_size, :]
        sink_values = value[:, :sink_size, :]

        # 2. 保留 Recent Window (新增 - 关键修复)
        # 为最近的上下文预留预算，例如总预算的 20% 或者至少 32 个 token
        recent_window_size = max(32, int(budget * 0.2))
        # 确保不超出剩余预算
        recent_window_size = min(recent_window_size, budget - sink_size)

        recent_keys = key[:, -recent_window_size:, :]
        recent_values = value[:, -recent_window_size:, :]

        # 3. 中间区域：基于重要性采样 (Heavy Hitters)
        # 剩余的预算分配给中间区域的高分 Token
        remaining_budget = budget - sink_size - recent_window_size

        if remaining_budget > 0:
            # 中间区域范围：从 sink_size 到 seq_len - recent_window_size
            middle_keys = key[:, sink_size:-recent_window_size, :]
            middle_values = value[:, sink_size:-recent_window_size, :]

            # 计算中间区域的 importance scores
            if importance_scores is not None:
                # 对应的 score 切片
                mid_scores = importance_scores[:, sink_size:-recent_window_size]
            else:
                # Fallback: L2 Norm
                mid_scores = torch.norm(middle_keys, p=2, dim=-1)

            # 选取 Top-K
            # 确保 k 不超过中间区域实际长度
            k = min(remaining_budget, mid_scores.size(1))

            if k > 0:
                _, top_indices = torch.topk(mid_scores, k, dim=1)
                top_indices, _ = torch.sort(top_indices, dim=1) # 保持原有顺序

                selected_mid_keys = torch.gather(
                    middle_keys, 1,
                    top_indices.unsqueeze(-1).expand(-1, -1, head_dim)
                )
                selected_mid_values = torch.gather(
                    middle_values, 1,
                    top_indices.unsqueeze(-1).expand(-1, -1, head_dim)
                )
            else:
                selected_mid_keys = torch.empty(bsz, 0, head_dim, device=key.device, dtype=key.dtype)
                selected_mid_values = torch.empty(bsz, 0, head_dim, device=value.device, dtype=value.dtype)
        else:
            selected_mid_keys = torch.empty(bsz, 0, head_dim, device=key.device, dtype=key.dtype)
            selected_mid_values = torch.empty(bsz, 0, head_dim, device=value.device, dtype=value.dtype)

        # 4. 拼接三部分：Sink + Heavy Hitters + Recent
        compressed_key = torch.cat([sink_keys, selected_mid_keys, recent_keys], dim=1)
        compressed_value = torch.cat([sink_values, selected_mid_values, recent_values], dim=1)

        return compressed_key, compressed_value

    def evict_by_groups(
        self,
        layer_idx: int,
        evict_start: int,
        evict_end: int
    ):
        """
        根据Group边界驱逐tokens
        """
        if len(self.key_cache) <= layer_idx:
            return

        key_cache = self.key_cache[layer_idx]  # [bsz, num_heads, seq_len, head_dim]
        value_cache = self.value_cache[layer_idx]

        # 移除指定范围的tokens
        before_keys = key_cache[:, :, :evict_start, :]
        after_keys = key_cache[:, :, evict_end:, :]
        before_values = value_cache[:, :, :evict_start, :]
        after_values = value_cache[:, :, evict_end:, :]

        # 重新拼接
        self.key_cache[layer_idx] = torch.cat([before_keys, after_keys], dim=2)
        self.value_cache[layer_idx] = torch.cat([before_values, after_values], dim=2)

    def get_memory_usage(self) -> float:
        """
        获取当前KV cache的内存占用（GB）
        """
        total_elements = 0

        for key_cache, value_cache in zip(self.key_cache, self.value_cache):
            if key_cache is not None:
                total_elements += key_cache.numel()
            if value_cache is not None:
                total_elements += value_cache.numel()

        # 假设float16 (2 bytes per element)
        memory_bytes = total_elements * 2
        memory_gb = memory_bytes / (1024 ** 3)

        return memory_gb

    def get_seq_length(self, layer_idx: Optional[int] = None) -> int:
        """
        返回逻辑序列长度（用于position_ids计算）
        注意：返回的是_seen_tokens（逻辑长度），不是物理cache长度
        """
        return self._seen_tokens

    def get_max_length(self) -> Optional[int]:
        """返回最大cache长度（预算）"""
        return self.total_budget

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = None) -> int:
        """
        返回可用的cache长度（用于position_ids计算）
        对于Head-Aware Cache，返回逻辑长度_seen_tokens
        """
        return self._seen_tokens

    def adjust_budget(self, new_budget: int):
        """动态调整预算"""
        self.total_budget = new_budget

    def pop(self):
        """
        移除最后一个token的cache（用于回退）
        确保 pop 操作也能正确执行
        """
        if len(self.key_cache) > 0:
            # 简单实现：截断最后一个 token
            for i in range(len(self.key_cache)):
                self.key_cache[i] = self.key_cache[i][:, :, :-1, :]
                self.value_cache[i] = self.value_cache[i][:, :, :-1, :]