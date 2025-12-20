# Modified 2025 by Junlong Tong (Shanghai Jiao Tong University & Eastern Institute of Technology).
# Copy and modified from 'Simul-LLM' repository.

# Modified 2025 by Junlong Tong.
import torch
from transformers.generation.stopping_criteria import StoppingCriteria


class StopTokenCriteria(StoppingCriteria):
    def __init__(
            self,
            tokenizer,
            max_new_tokens: int,
            end_Instruct=None
    ):
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        # [核心修复] 自动获取 EOS ID，不再依赖默认字符串
        self.eos_token_id = tokenizer.eos_token_id
        # 针对 Qwen 的特殊处理：如果 tokenizer 没设好，手动添加 Qwen 的结束符 ID
        self.qwen_eos_id = 151645
        # 字符串兜底
        if end_Instruct is None:
            self.end_Instruct = tokenizer.eos_token if tokenizer.eos_token else ""
        else:
            self.end_Instruct = end_Instruct
    def __call__(self, target_ids: torch.LongTensor, scores: torch.FloatTensor, token_count, **kwargs) -> bool:
        # 1. 最快、最准的判断：检查最后一个 Token 的 ID
        last_token_id = target_ids[0, -1].item()

        # 检查标准 EOS
        if self.eos_token_id is not None:
            if isinstance(self.eos_token_id, list):
                if last_token_id in self.eos_token_id:
                    return torch.tensor(True), torch.tensor(False)
            elif last_token_id == self.eos_token_id:
                return torch.tensor(True), torch.tensor(False)
        # 检查 Qwen 特有 EOS (防止 config 加载错误)
        if last_token_id == self.qwen_eos_id:
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