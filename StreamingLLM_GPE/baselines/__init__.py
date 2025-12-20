"""
Baseline implementations for KV cache compression
- H2O: Heavy-Hitter Oracle
- StreamingLLM: Fixed window + Attention Sinks
"""

from .h2o_cache import H2OCache
from .streamingllm_cache import StreamingLLMCache

__all__ = ['H2OCache', 'StreamingLLMCache']

