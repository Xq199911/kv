# Modified 2025 by Junlong Tong (Shanghai Jiao Tong University & Eastern Institute of Technology).
# Copy and modified from 'Simul-LLM' repository.

# Modified 2025 by Junlong Tong.
import torch
from transformers.generation.stopping_criteria import StoppingCriteria
import sys
import os

# 添加 utils 路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, '../..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from StreamingLLM_GPE.utils.tokenizer_utils import get_all_eos_token_ids
except ImportError:
    # 如果导入失败，使用简单的 fallback
    def get_all_eos_token_ids(tokenizer):
        eos_ids = set()
        if tokenizer.eos_token_id is not None:
            if isinstance(tokenizer.eos_token_id, list):
                eos_ids.update(tokenizer.eos_token_id)
            else:
                eos_ids.add(tokenizer.eos_token_id)
        return eos_ids


class StopTokenCriteria(StoppingCriteria):
    def __init__(
            self,
            tokenizer,
            max_new_tokens: int,
            end_Instruct=None
    ):
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        # [通用修复] 自动获取所有可能的 EOS token IDs
        self.eos_token_ids = get_all_eos_token_ids(tokenizer)
        # 字符串兜底
        if end_Instruct is None:
            self.end_Instruct = tokenizer.eos_token if tokenizer.eos_token else ""
        else:
            self.end_Instruct = end_Instruct
    def __call__(self, target_ids: torch.LongTensor, scores: torch.FloatTensor, token_count, **kwargs) -> bool:
        # 1. 最快、最准的判断：检查最后一个 Token 的 ID
        last_token_id = target_ids[0, -1].item()

        # [通用修复] 检查所有可能的 EOS token IDs
        if last_token_id in self.eos_token_ids:
            return torch.tensor(True), torch.tensor(False)
        
        # 2. 检查长度限制
        if token_count >= self.max_new_tokens:
            is_done = True
            return torch.tensor(is_done), torch.tensor(False)

        # 3. 字符串检查 (作为双重保险)
        token_pred = self.tokenizer.decode(target_ids[0][-1:])
        if self.end_Instruct and self.end_Instruct in token_pred:
            return torch.tensor(True), torch.tensor(False)

        return torch.tensor(False), torch.tensor(False)