"""
Tokenizer 工具函数
用于多模型支持，统一处理 EOS token 等配置
"""
from typing import List, Set, Optional
from transformers import PreTrainedTokenizerBase


def get_all_eos_token_ids(tokenizer: PreTrainedTokenizerBase) -> Set[int]:
    """
    从 tokenizer 获取所有可能的 EOS token IDs
    
    这个方法会：
    1. 检查 tokenizer.eos_token_id（可能是一个 ID 或列表）
    2. 检查 tokenizer 的 special_tokens_map 和 added_tokens_decoder
    3. 对于特定模型（如 Qwen），添加常见的结束标记
    
    Args:
        tokenizer: Transformers tokenizer
        
    Returns:
        所有可能的 EOS token IDs 的集合
    """
    eos_token_ids = set()
    
    # 1. 标准 EOS token ID
    if tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, list):
            eos_token_ids.update(tokenizer.eos_token_id)
        else:
            eos_token_ids.add(tokenizer.eos_token_id)
    
    # 2. 检查 tokenizer 的 special_tokens_map 和 added_tokens_decoder
    # 查找所有包含 "end" 或 "eos" 的特殊 token
    if hasattr(tokenizer, 'added_tokens_decoder'):
        for token_id, token_info in tokenizer.added_tokens_decoder.items():
            token_id_int = int(token_id)
            if isinstance(token_info, dict):
                content = token_info.get('content', '')
            else:
                content = str(token_info)
            
            # 检查是否是结束标记（常见的结束标记关键词）
            content_lower = content.lower()
            if any(keyword in content_lower for keyword in ['end', 'eos', 'eot', 'stop']):
                eos_token_ids.add(token_id_int)
    
    # 3. 模型特定的处理（作为兜底，但优先使用 tokenizer 配置）
    # 对于 Qwen Instruct 模型，可能需要检查 <|im_end|> 和 <|endoftext|>
    if hasattr(tokenizer, 'encode'):
        try:
            # 尝试编码常见的结束标记字符串
            test_strings = ['<|im_end|>', '<|endoftext|>', '</s>', '<eos>']
            for test_str in test_strings:
                try:
                    encoded = tokenizer.encode(test_str, add_special_tokens=False)
                    if len(encoded) == 1:  # 单个 token
                        eos_token_ids.add(encoded[0])
                except:
                    continue
        except:
            pass
    
    return eos_token_ids


def normalize_tokenizer_eos_token(tokenizer: PreTrainedTokenizerBase, model_name: Optional[str] = None) -> Set[int]:
    """
    标准化 tokenizer 的 EOS token 配置
    
    这个方法会：
    1. 确保 tokenizer.eos_token_id 被正确设置
    2. 对于特定模型（如 Qwen Instruct），可能需要设置特殊的 EOS token
    
    Args:
        tokenizer: Transformers tokenizer
        model_name: 模型名称（如 'Qwen', 'Llama'），用于特定处理
    """
    all_eos_ids = get_all_eos_token_ids(tokenizer)
    
    if not all_eos_ids:
        # 如果没有找到任何 EOS token，尝试设置默认值
        if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token:
            try:
                eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
                if eos_id != tokenizer.unk_token_id:
                    tokenizer.eos_token_id = eos_id
                    all_eos_ids.add(eos_id)
            except:
                pass
    
    # 对于 Qwen Instruct 模型，优先使用 <|im_end|>
    if model_name and "Qwen" in model_name:
        try:
            im_end_id = tokenizer.encode('<|im_end|>', add_special_tokens=False)
            if len(im_end_id) == 1:
                im_end_id_int = im_end_id[0]
                # 如果找到了 <|im_end|>，将其设置为主要 EOS token
                if im_end_id_int != tokenizer.unk_token_id:
                    tokenizer.eos_token_id = im_end_id_int
                    all_eos_ids.add(im_end_id_int)
        except:
            pass
    
    # 确保 pad_token 也被设置
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.eos_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return all_eos_ids

