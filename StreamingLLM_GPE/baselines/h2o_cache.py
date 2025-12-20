"""
H2O (Heavy-Hitter Oracle) Cache Implementation - FIXED v4 (Robust)
Based on: "H2O: Heavy-Hitter Oracle for Efficient Generative Inference" (NeurIPS 2023)
"""
import torch
from typing import Optional, Tuple, List
from transformers.cache_utils import Cache

class H2OCache(Cache):
    """
    H2O Cache: 基于重要性分数的统一压缩
    """

    def __init__(
        self,
        budget_per_layer: int = 2048,
        sink_tokens: int = 128,  # [FIXED] 统一sink tokens，从4改为128
        device: str = "cuda"
    ):
        super().__init__()
        self.budget_per_layer = budget_per_layer
        self.sink_tokens = sink_tokens
        self.device = device

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # [逻辑长度] 仅在第0层更新
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # [缓存拼接]
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([
                self.key_cache[layer_idx], key_states
            ], dim=2)
            self.value_cache[layer_idx] = torch.cat([
                self.value_cache[layer_idx], value_states
            ], dim=2)

        current_seq_len = self.key_cache[layer_idx].size(2)

        # [压缩触发条件] 严格检查是否超过预算
        if current_seq_len > self.budget_per_layer:
            attention_scores = None
            if cache_kwargs is not None:
                attention_scores = cache_kwargs.get("attention_scores", None)

            compressed_key, compressed_value = self._compress(
                self.key_cache[layer_idx],
                self.value_cache[layer_idx],
                attention_scores,
                layer_idx
            )

            self.key_cache[layer_idx] = compressed_key
            self.value_cache[layer_idx] = compressed_value

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def _sliding_window_fallback(self, key, value, remaining_budget):
        seq_len = key.size(2)

        # [Bug修复] 如果当前长度还未超过总预算（含Sink），不需要压缩！
        # 虽然外部 update 有检查，但这里作为 fallback 更安全
        if seq_len <= self.budget_per_layer:
            return key, value

        sink_key = key[:, :, :self.sink_tokens]
        sink_value = value[:, :, :self.sink_tokens]

        # 取最近的 remaining_budget 个 tokens
        # 注意：这里前提是 seq_len > budget，所以不会与 sink 重叠
        window_key = key[:, :, -remaining_budget:]
        window_value = value[:, :, -remaining_budget:]

        return torch.cat([sink_key, window_key], dim=2), torch.cat([sink_value, window_value], dim=2)

    def _compress(
            self,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_scores: Optional[torch.Tensor],
            layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, num_heads, seq_len, head_dim = key.shape
        remaining_budget = self.budget_per_layer - self.sink_tokens

        # [Safety Check] 如果长度未超标，直接返回
        if seq_len <= self.budget_per_layer:
            return key, value

        # Fallback to Sliding Window if no attention scores
        if attention_scores is None:
            if not hasattr(self, "_has_warned_no_attn"):
                # print(f"[H2O Warning] Layer {layer_idx}: No attention scores found! Falling back.")
                self._has_warned_no_attn = True
            return self._sliding_window_fallback(key, value, remaining_budget)

        # Handle Attention Scores
        if attention_scores.dim() == 4:
            if attention_scores.size(2) == 1:
                importance = attention_scores.squeeze(2)
            else:
                importance = attention_scores.sum(dim=-2)
        else:
            return self._sliding_window_fallback(key, value, remaining_budget)

        importance = importance.mean(dim=1)  # [batch, seq_len]

        if importance.size(1) != seq_len:
            return self._sliding_window_fallback(key, value, remaining_budget)

        # Select Top-K
        importance_no_sink = importance[:, self.sink_tokens:]
        current_candidates = importance_no_sink.size(1)
        actual_k = min(remaining_budget, current_candidates)

        _, top_indices = torch.topk(importance_no_sink, actual_k, dim=1)
        top_indices = top_indices + self.sink_tokens
        top_indices, _ = top_indices.sort(dim=1)

        try:
            top_indices = top_indices.long().to(key.device)
            gather_index = top_indices.unsqueeze(1).unsqueeze(-1).expand(
                batch_size, num_heads, actual_k, head_dim
            )

            selected_key = torch.gather(key, dim=2, index=gather_index)
            selected_value = torch.gather(value, dim=2, index=gather_index)

            sink_key = key[:, :, :self.sink_tokens]
            sink_value = value[:, :, :self.sink_tokens]

            compressed_key = torch.cat([sink_key, selected_key], dim=2)
            compressed_value = torch.cat([sink_value, selected_value], dim=2)

            return compressed_key, compressed_value

        except RuntimeError as e:
            print(f"[H2O Error] Layer {layer_idx} failed: {e}")
            return self._sliding_window_fallback(key, value, remaining_budget)

    def get_seq_length(self, layer_idx: Optional[int] = None) -> int:
        return self._seen_tokens

    def get_max_length(self) -> Optional[int]:
        return self.budget_per_layer

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = None) -> int:
        return self._seen_tokens