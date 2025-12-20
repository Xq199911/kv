"""
StreamingLLM Cache Implementation - FIXED for RoPE
Based on: "StreamingLLM: Efficient Inference with Attention Sinks" (ICLR 2024)
"""
import torch
from typing import Optional, Tuple, List
from transformers.cache_utils import Cache

class StreamingLLMCache(Cache):
    """
    StreamingLLM Cache: Fixed window size + Attention Sinks
    Fixed: Adds logical token tracking (_seen_tokens) to prevent RoPE collision.
    """

    def __init__(
        self,
        window_size: int = 512,
        sink_tokens: int = 128,  # [FIXED] 统一sink tokens，从4改为128
        device: str = "cuda"
    ):
        super().__init__()
        self.window_size = window_size
        self.sink_tokens = sink_tokens
        self.device = device

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0  #  追踪逻辑长度

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        #  仅在第0层更新计数，避免重复
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # 初始化
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
        max_len = self.sink_tokens + self.window_size

        # 压缩 (Eviction)
        if current_seq_len > max_len:
            sink_key = self.key_cache[layer_idx][:, :, :self.sink_tokens]
            sink_value = self.value_cache[layer_idx][:, :, :self.sink_tokens]

            window_key = self.key_cache[layer_idx][:, :, -self.window_size:]
            window_value = self.value_cache[layer_idx][:, :, -self.window_size:]

            self.key_cache[layer_idx] = torch.cat([sink_key, window_key], dim=2)
            self.value_cache[layer_idx] = torch.cat([sink_value, window_value], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = None) -> int:
        #  返回逻辑长度，保证 position_ids 持续增长
        return self._seen_tokens

    def get_max_length(self) -> Optional[int]:
        return self.sink_tokens + self.window_size

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = None) -> int:
        return self._seen_tokens