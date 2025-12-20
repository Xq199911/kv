"""
Head-Aware KV Cache推理示例

展示如何使用Head-Aware Dynamic Cache进行推理
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from transformers import AutoTokenizer, AutoConfig
from StreamingLLM_GPE.models.Qwen2_5.qwen_streaming import Qwen2ForCausalLM_stream
from StreamingLLM_GPE.models.Qwen2_5.head_aware_cache import HeadAwareDynamicCache
from StreamingLLM_GPE.utils.head_analyzer import HeadAnalyzer
from StreamingLLM_GPE.utils.group_tracker import GroupTracker
from StreamingLLM_GPE.utils.budget_monitor import BudgetMonitor


def initialize_head_aware_model(
        model_path: str,
        total_budget: int = 2048,
        max_memory_gb: float = 4.0,
        device: str = 'cuda'
):
    """
    初始化带Head-Aware Cache的模型
    
    Args:
        model_path: 模型路径
        total_budget: KV cache总预算（tokens per layer）
        max_memory_gb: 最大内存占用（GB）
        device: 设备
    """
    # 加载模型和tokenizer
    config = AutoConfig.from_pretrained(model_path)
    config._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='right', config=config)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型（使用量化以节省显存）
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = Qwen2ForCausalLM_stream.from_pretrained(
        model_path,
        config=config,
        quantization_config=quantization_config,
        device_map="auto"
    )

    # 初始化Head分析器
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_analyzer = HeadAnalyzer(num_layers, num_heads, device=device)

    # 初始化Group跟踪器
    group_tracker = GroupTracker(sink_groups=2)

    # 初始化预算监控器
    budget_monitor = BudgetMonitor(max_memory_gb=max_memory_gb)

    return model, tokenizer, head_analyzer, group_tracker, budget_monitor


def analyze_model_heads(
        model,
        tokenizer,
        head_analyzer: HeadAnalyzer,
        sample_texts: list,
        device: str = 'cuda'
):
    """
    在样本数据上分析head特性
    
    Args:
        model: 模型
        tokenizer: tokenizer
        head_analyzer: Head分析器
        sample_texts: 样本文本列表
        device: 设备
    """
    print("Analyzing model heads...")

    sample_inputs = []
    for text in sample_texts[:10]:  # 使用前10个样本
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        sample_inputs.append(inputs)

    # 分析head特性
    head_analyzer.analyze_model(model, sample_inputs, num_samples=len(sample_inputs))

    print("Head analysis completed!")


def create_head_aware_cache(
        head_analyzer: HeadAnalyzer,
        group_tracker: GroupTracker,
        total_budget: int = 2048,
        device: str = 'cuda'
) -> HeadAwareDynamicCache:
    """
    创建Head-Aware Cache实例
    """
    cache = HeadAwareDynamicCache(
        head_analyzer=head_analyzer,
        group_tracker=group_tracker,
        total_budget=total_budget,
        sink_tokens=4,
        adaptive=True,
        device=device
    )
    return cache


def inference_with_head_aware_cache(
        model,
        tokenizer,
        prompt: str,
        head_analyzer: HeadAnalyzer,
        group_tracker: GroupTracker,
        budget_monitor: BudgetMonitor,
        max_new_tokens: int = 512,
        total_budget: int = 2048,
        device: str = 'cuda'
):
    """
    使用Head-Aware Cache进行推理
    
    Args:
        model: 模型
        tokenizer: tokenizer
        prompt: 输入提示
        head_analyzer: Head分析器
        group_tracker: Group跟踪器
        budget_monitor: 预算监控器
        max_new_tokens: 最大生成token数
        total_budget: KV cache预算
        device: 设备
    """
    # 创建Head-Aware Cache
    cache = create_head_aware_cache(head_analyzer, group_tracker, total_budget, device)

    # 编码输入
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)

    # 初始化cache
    model.past_key_values = cache

    # 生成
    generated_ids = input_ids.input_ids
    memory_stats = []

    for step in range(max_new_tokens):
        # 前向传播
        outputs = model(
            input_ids=generated_ids[:, -1:],  # 只使用最后一个token
            past_key_values=cache,
            use_cache=True,
            output_attentions=True  # 需要attention weights用于head分析
        )

        # 更新head分析（如果还没分析）
        if not head_analyzer.analyzed and outputs.attentions:
            for layer_idx, attn in enumerate(outputs.attentions):
                head_analyzer.analyze_head_functionality(attn, layer_idx)

        # 采样下一个token
        logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(logits, dim=-1).unsqueeze(0)
        generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

        # 检查预算
        budget_monitor.check_and_evict(cache, group_tracker)

        # 记录内存使用
        if step % 50 == 0:
            memory = cache.get_memory_usage()
            memory_stats.append((step, memory))
            print(f"Step {step}: Memory = {memory:.2f}GB, Tokens = {generated_ids.shape[1]}")

        # 检查结束条件
        if next_token_id.item() == tokenizer.eos_token_id:
            break

    # 解码输出
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return generated_text, memory_stats


if __name__ == "__main__":
    # 配置
    model_path = "./models/Qwen2.5-3B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_budget = 2048  # 每层2048 tokens
    max_memory_gb = 4.0  # 最大4GB KV cache

    print("=" * 50)
    print("Head-Aware KV Cache Inference Example")
    print("=" * 50)

    # 初始化
    model, tokenizer, head_analyzer, group_tracker, budget_monitor = \
        initialize_head_aware_model(model_path, total_budget, max_memory_gb, device)

    # 分析head特性（可选，如果已有分析结果可以跳过）
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In a world where artificial intelligence is rapidly advancing...",
        # 添加更多样本...
    ]
    # analyze_model_heads(model, tokenizer, head_analyzer, sample_texts, device)

    # 推理
    prompt = "Translate the following English text to Chinese: Hello, how are you?"

    print(f"\nPrompt: {prompt}")
    print("Generating...")

    generated_text, memory_stats = inference_with_head_aware_cache(
        model, tokenizer, prompt,
        head_analyzer, group_tracker, budget_monitor,
        max_new_tokens=512,
        total_budget=total_budget,
        device=device
    )

    print(f"\nGenerated: {generated_text}")
    print(f"\nMemory Stats:")
    for step, memory in memory_stats:
        print(f"  Step {step}: {memory:.2f}GB")

    # 打印最终统计
    final_stats = budget_monitor.get_memory_stats()
    print(f"\nFinal Memory Stats: {final_stats}")
