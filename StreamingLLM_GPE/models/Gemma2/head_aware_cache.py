"""
Head-Aware Dynamic KV Cache for Gemma2
基于head功能特性的动态KV cache压缩和驱逐
结合Group-aware策略，实现细粒度的内存管理
"""
import torch
from typing import List, Tuple, Optional, Dict
from transformers.cache_utils import Cache, DynamicCache as TransformersDynamicCache
import sys
import os

# Add parent directory to path before importing
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, '../../..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import utils modules first
from StreamingLLM_GPE.utils.head_analyzer import HeadAnalyzer
from StreamingLLM_GPE.utils.group_tracker import GroupTracker

# Import the custom DynamicCache from gemma2_streaming (which has pop method)
from StreamingLLM_GPE.models.Gemma2.gemma2_streaming import DynamicCache as GemmaDynamicCache


class HeadAwareDynamicCache(GemmaDynamicCache):
    """
    Head-Aware Dynamic KV Cache for Gemma2
    
    核心特性：
    1. 基于head功能特性分配预算
    2. 支持Group-level驱逐
    3. 自适应预算调整
    """
    
    def __init__(
        self,
        head_analyzer: HeadAnalyzer,
        group_tracker: Optional[GroupTracker] = None,
        total_budget: int = 2048,
        sink_tokens: int = 4,
        adaptive: bool = True,
        device: str = 'cuda'
    ):
        """
        Args:
            head_analyzer: Head分析器
            group_tracker: Group跟踪器（可选）
            total_budget: 总预算（tokens per layer）
            sink_tokens: Attention sink tokens数量
            adaptive: 是否使用自适应分配
            device: 设备
        """
        super().__init__()
        self.head_analyzer = head_analyzer
        self.group_tracker = group_tracker
        self.total_budget = total_budget
        self.sink_tokens = sink_tokens
        self.adaptive = adaptive
        self.device = device
        
        # 存储每个head的独立cache（可选，用于细粒度控制）
        self.head_caches: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        
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
            cache_kwargs: 缓存参数
        
        Returns:
            (compressed_key_states, compressed_value_states)
        """
        bsz, num_heads, seq_len, head_dim = key_states.shape
        
        # 如果有现有cache，先合并
        if len(self.key_cache) > layer_idx:
            # 合并新tokens到现有cache
            existing_keys = self.key_cache[layer_idx]  # [bsz, num_heads, cached_len, head_dim]
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
            # 不需要压缩，直接更新
            if len(self.key_cache) <= layer_idx:
                self.key_cache.append(combined_keys)
                self.value_cache.append(combined_values)
            else:
                self.key_cache[layer_idx] = combined_keys
                self.value_cache[layer_idx] = combined_values
            
            return combined_keys, combined_values
        
        # 需要压缩：根据head特性进行压缩
        compressed_keys, compressed_values = self._compress_by_head(
            combined_keys, combined_values, layer_idx
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
        layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据head特性进行压缩
        
        Args:
            key_states: [bsz, num_heads, seq_len, head_dim]
            value_states: [bsz, num_heads, seq_len, head_dim]
            layer_idx: 层索引
        
        Returns:
            (compressed_keys, compressed_values)
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
            
            budget = head_budgets.get(head_idx, self.total_budget // num_heads)
            head_type = self.head_analyzer.head_profiles.get(layer_idx, {}).get(head_idx, 'induction')
            
            # 根据head类型选择压缩策略
            if head_type == 'retrieval':
                # Retrieval heads: 保留重要性高的tokens
                comp_key, comp_value = self._compress_by_importance(
                    head_key, head_value, budget
                )
            elif head_type == 'local':
                # Local heads: 只保留最近的tokens
                comp_key, comp_value = self._compress_recent(
                    head_key, head_value, budget
                )
            else:  # induction
                # Induction heads: 保留有模式的关键tokens
                comp_key, comp_value = self._compress_pattern(
                    head_key, head_value, budget
                )
            
            compressed_keys_list.append(comp_key)
            compressed_values_list.append(comp_value)
        # 重新组合
        compressed_keys = torch.stack(compressed_keys_list, dim=1)  # [bsz, num_heads, budget, head_dim]
        compressed_values = torch.stack(compressed_values_list, dim=1)
        
        return compressed_keys, compressed_values
    
    def _compress_by_importance(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        budget: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        基于重要性压缩（用于retrieval heads）
        
        策略：
        1. 保留sink tokens（前N个）
        2. 保留重要性高的tokens（基于key的norm）
        """
        bsz, seq_len, head_dim = key.shape
        
        if seq_len <= budget:
            return key, value
        
        # 1. 保留sink tokens
        sink_size = min(self.sink_tokens, budget // 2)
        sink_keys = key[:, :sink_size, :]
        sink_values = value[:, :sink_size, :]
        
        # 2. 计算剩余tokens的重要性分数（基于key的L2 norm）
        remaining_keys = key[:, sink_size:, :]
        remaining_values = value[:, sink_size:, :]
        
        # 计算每个token的重要性（key的norm）
        importance_scores = torch.norm(remaining_keys, dim=-1)  # [bsz, seq_len - sink_size]
        
        # 选择top-k重要的tokens
        remaining_budget = budget - sink_size
        _, top_indices = torch.topk(importance_scores, remaining_budget, dim=1)
        
        # 收集重要的tokens
        selected_keys = torch.gather(
            remaining_keys, 1, 
            top_indices.unsqueeze(-1).expand(-1, -1, head_dim)
        )
        selected_values = torch.gather(
            remaining_values, 1,
            top_indices.unsqueeze(-1).expand(-1, -1, head_dim)
        )
        
        # 合并sink和重要的tokens
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
        保留最近的tokens（用于local heads）
        """
        bsz, seq_len, head_dim = key.shape
        
        if seq_len <= budget:
            return key, value
        
        # 只保留最近的budget个tokens
        compressed_key = key[:, -budget:, :]
        compressed_value = value[:, -budget:, :]
        
        return compressed_key, compressed_value
    
    def _compress_pattern(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        budget: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        保留有模式的关键tokens（用于induction heads）
        
        策略：
        1. 保留sink tokens
        2. 均匀采样剩余tokens（保持模式）
        """
        bsz, seq_len, head_dim = key.shape
        
        if seq_len <= budget:
            return key, value
        
        # 1. 保留sink tokens
        sink_size = min(self.sink_tokens, budget // 3)
        sink_keys = key[:, :sink_size, :]
        sink_values = value[:, :sink_size, :]
        
        # 2. 均匀采样剩余tokens
        remaining_keys = key[:, sink_size:, :]
        remaining_values = value[:, sink_size:, :]
        remaining_len = remaining_keys.shape[1]
        
        remaining_budget = budget - sink_size
        
        # 均匀采样
        step = remaining_len / remaining_budget
        indices = torch.arange(remaining_budget, device=key.device) * step
        indices = indices.long()
        
        selected_keys = remaining_keys[:, indices, :]
        selected_values = remaining_values[:, indices, :]
        
        # 合并
        compressed_key = torch.cat([sink_keys, selected_keys], dim=1)
        compressed_value = torch.cat([sink_values, selected_values], dim=1)
        
        return compressed_key, compressed_value
    
    def evict_by_groups(
        self,
        layer_idx: int,
        evict_start: int,
        evict_end: int
    ):
        """
        根据Group边界驱逐tokens
        
        Args:
            layer_idx: 层索引
            evict_start: 驱逐起始位置
            evict_end: 驱逐结束位置
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
        
        # 更新seen_tokens
        evicted_count = evict_end - evict_start
        self._seen_tokens -= evicted_count
    
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
    
    def adjust_budget(self, new_budget: int):
        """动态调整预算"""
        self.total_budget = new_budget

