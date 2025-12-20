"""
Head-Aware KV Cache评估脚本

对比不同KV cache策略的性能：
1. Baseline (GPE, 无压缩)
2. Head-Aware Dynamic Cache
3. Group-Aware Eviction
4. Head-Aware + Group-Aware (完整方案)
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from transformers import AutoTokenizer, AutoConfig
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate import Accelerator
import peft
from argparse import ArgumentParser
import logging
from transformers import BitsAndBytesConfig
import json
import time

from StreamingLLM_GPE.dataloader_hf import StreamingDataCollator
from StreamingLLM_GPE.models.Qwen2_5.qwen_streaming import Qwen2ForCausalLM_stream
from StreamingLLM_GPE.models.Qwen2_5.head_aware_cache import HeadAwareDynamicCache
from StreamingLLM_GPE.utils.head_analyzer import HeadAnalyzer
from StreamingLLM_GPE.utils.group_tracker import GroupTracker
from StreamingLLM_GPE.utils.budget_monitor import BudgetMonitor
# Import utils.py directly (avoid conflict with utils package directory)
import importlib.util
_utils_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils.py')
_spec = importlib.util.spec_from_file_location("StreamingLLM_GPE.utils_module", _utils_file_path)
utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(utils)
from StreamingLLM_GPE.evaluate.lagging import calculate_al_and_laal
import sacrebleu


class MemoryMonitor:
    """监控GPU内存使用"""
    def __init__(self, device=0):
        self.device = device
        self.peak_memory = 0
        self.memory_history = []
    
    def record(self):
        if torch.cuda.is_available():
            current = torch.cuda.memory_allocated(self.device) / (1024**3)  # GB
            self.memory_history.append(current)
            self.peak_memory = max(self.peak_memory, current)
            return current
        return 0
    
    def get_stats(self):
        return {
            'peak_memory_gb': self.peak_memory,
            'avg_memory_gb': sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0,
            'final_memory_gb': self.memory_history[-1] if self.memory_history else 0
        }


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--pe_cache_length", type=float, default=0)
    parser.add_argument("--inference_mode", type=str, default="streaming",
                        choices=["batch", "streaming"])
    parser.add_argument("--LLM_backbone", type=str, default="Qwen",
                        choices=["Qwen", "Llama", "Gemma"])
    parser.add_argument("--LLM_path", type=str, required=True, help="Path to the LLM model.")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--wait_k", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default='./output_logs')
    parser.add_argument("--split_mode", type=str, default="word",
                        choices=["word", "token"])
    parser.add_argument("--params", type=str, default="./StreamingLLM_GPE/configs/params_qwen_inference.json")
    
    # Head-Aware specific arguments
    parser.add_argument("--use_head_aware", action="store_true", help="Use Head-Aware Cache")
    parser.add_argument("--use_group_aware", action="store_true", help="Use Group-Aware Eviction")
    parser.add_argument("--total_budget", type=int, default=2048, help="KV cache budget per layer")
    parser.add_argument("--max_memory_gb", type=float, default=4.0, help="Max KV cache memory (GB)")
    parser.add_argument("--analyze_heads", action="store_true", help="Analyze head functionality")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process (for testing)")
    parser.add_argument("--min_source_length", type=int, default=0, help="Minimum source length in words (filter short sequences)")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens to generate")
    
    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def initialize_head_aware_components(model, config, args):
    """初始化Head-Aware相关组件"""
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    
    head_analyzer = HeadAnalyzer(num_layers, num_heads, device=device)
    group_tracker = GroupTracker(sink_groups=2) if args.use_group_aware else None
    budget_monitor = BudgetMonitor(max_memory_gb=args.max_memory_gb) if args.use_head_aware else None
    
    return head_analyzer, group_tracker, budget_monitor


def create_cache(head_analyzer, group_tracker, args):
    """创建KV cache"""
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    
    if args.use_head_aware:
        from StreamingLLM_GPE.models.Qwen2_5.head_aware_cache import HeadAwareDynamicCache
        cache = HeadAwareDynamicCache(
            head_analyzer=head_analyzer,
            group_tracker=group_tracker,
            total_budget=args.total_budget,
            sink_tokens=4,
            adaptive=True,
            device=device
        )
        return cache
    else:
        from StreamingLLM_GPE.models.Qwen2_5.qwen_streaming import DynamicCache
        return DynamicCache()


def main():
    args = get_args()
    params = utils.Params(args.params)
    setup_seed(0)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # 设置日志
    log_file = f"{args.output_dir}/head_aware_eval.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    # 记录配置
    config_str = f"""
    Configuration:
    - Model: {args.LLM_backbone} ({args.LLM_path})
    - Inference Mode: {args.inference_mode}
    - Head-Aware: {args.use_head_aware}
    - Group-Aware: {args.use_group_aware}
    - Total Budget: {args.total_budget} tokens/layer
    - Max Memory: {args.max_memory_gb} GB
    - Wait-k: {args.wait_k}
    """
    print(config_str)
    logging.info(config_str)
    
    # 加载模型
    config = AutoConfig.from_pretrained(args.LLM_path)
    config._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(args.LLM_path, padding_side='right', config=config)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 量化配置（节省显存）
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = Qwen2ForCausalLM_stream.from_pretrained(
        args.LLM_path,
        ignore_mismatched_sizes=True,
        config=config,
        quantization_config=quantization_config,
        device_map="auto"
    )
    
    if args.lora_path is not None:
        model = peft.PeftModel.from_pretrained(model, args.lora_path)
    
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id
    
    # 初始化Head-Aware组件
    head_analyzer, group_tracker, budget_monitor = initialize_head_aware_components(
        model, config, args
    )
    
    # 如果启用Head-Aware，预分析heads（可选）
    if args.use_head_aware and args.analyze_heads:
        print("Analyzing head functionality...")
        # 这里可以添加head分析逻辑
        # 暂时跳过，在实际推理中动态分析
    
    # 数据加载
    data_collator = StreamingDataCollator(
        file_path=params.file_path,
        tokenizer=tokenizer,
        Instruct=params.Instruct,
        user_Instruct=params.user_Instruct,
        assistant_Instruct=params.assistant_Instruct,
        end_Instruct=params.end_Instruct,
        source_key=params.source_key,
        target_key=params.target_key,
        inference_mode=args.inference_mode,
        split_mode=args.split_mode,
        if_add_space=params.if_add_space,
        pe_cache_length=args.pe_cache_length,
        wait_k=args.wait_k,
    )
    
    data_collator_dataset = data_collator.dataset_loader()
    
    # 过滤短序列（如果指定了最小长度）
    if args.min_source_length > 0:
        def filter_long_sequences(example):
            source_words = example.get("source_txt", "").split()
            return len(source_words) >= args.min_source_length
        
        data_collator_dataset = data_collator_dataset.filter(filter_long_sequences)
        logging.info(f"Filtered dataset: keeping sequences with >= {args.min_source_length} source words")
    
    # 限制样本数量（用于测试）
    if args.max_samples is not None and args.max_samples > 0:
        data_collator_dataset = data_collator_dataset.select(range(min(args.max_samples, len(data_collator_dataset))))
        logging.info(f"Limited dataset to {len(data_collator_dataset)} samples")
    
    dataloader = DataLoader(
        data_collator_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=data_collator.collate_fn_inference
    )
    
    accelerator = Accelerator(mixed_precision="bf16")
    stream_model, dataloader = accelerator.prepare(model, dataloader)
    
    # 评估指标
    target_txt_lt = []
    output_text_lt = []
    AL = []
    LAAL = []
    
    # 内存监控
    memory_monitor = MemoryMonitor(device=args.device)
    
    # 统计信息
    stats = {
        'total_tokens': 0,
        'max_length': 0,
        'cache_memory_gb': [],
        'inference_times': []
    }
    
    stream_model.eval()
    
    for step, batch in enumerate(tqdm(dataloader)):
        # 创建cache（在generate之前设置，generate内部会检查并使用）
        if args.use_head_aware:
            cache = create_cache(head_analyzer, group_tracker, args)
            # 设置source和target cache
            stream_model.source_key_values = cache
            stream_model.target_key_values = create_cache(head_analyzer, group_tracker, args)
            stream_model.past_key_values = create_cache(head_analyzer, group_tracker, args)
        else:
            from StreamingLLM_GPE.models.Qwen2_5.qwen_streaming import DynamicCache
            stream_model.source_key_values = DynamicCache()
            stream_model.target_key_values = DynamicCache()
            stream_model.past_key_values = DynamicCache()
        
        input_ids = batch.get("source_tokens", None)
        attention_mask = batch.get("attention_mask", None)
        _lengths = batch.get("_lengths", None)
        inference_mode = batch.get("inference_mode", "streaming")
        split_mode = batch.get("split_mode", None)
        _lengths_index = batch.get("_lengths_index", None)
        wait_k = batch.get("wait_k", None)
        assistant_token = batch.get("assistant_token", None)
        source_txt = batch.get("source_txt", None)
        target_txt = batch.get("target_txt", None)
        
        # 记录内存
        memory_monitor.record()
        
        # 推理
        start_time = time.time()
        
        if inference_mode == "streaming":
            output_sequences, wait_lagging = stream_model.generate(
                input_ids=input_ids.to(accelerator.device),
                attention_mask=attention_mask.to(accelerator.device),
                max_new_tokens=args.max_new_tokens,
                generate_mode=inference_mode,
                split_mode=split_mode,
                pe_cache_length=args.pe_cache_length,
                tokenizer=tokenizer,
                end_Instruct=params.end_Instruct,
                _lengths=_lengths,
                _lengths_index=_lengths_index,
                wait_k=wait_k,
                source_words=source_txt,
                assistant_token=assistant_token.to(accelerator.device),
            )
        elif inference_mode == "batch":
            output_sequences, wait_lagging = stream_model.generate(
                input_ids=input_ids.to(accelerator.device),
                attention_mask=attention_mask.to(accelerator.device),
                max_new_tokens=1024,
            )
        
        inference_time = time.time() - start_time
        stats['inference_times'].append(inference_time)
        
        # 解码输出
        output_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        target_txt_lt.extend(target_txt)
        output_text_lt.extend([output_text])
        
        # 记录统计信息
        seq_len = len(output_sequences[0])
        stats['total_tokens'] += seq_len
        stats['max_length'] = max(stats['max_length'], seq_len)
        
        # 记录cache内存
        if args.use_head_aware and isinstance(stream_model.source_key_values, HeadAwareDynamicCache):
            cache_memory = stream_model.source_key_values.get_memory_usage()
            stats['cache_memory_gb'].append(cache_memory)
        
        # 检查预算（如果启用）
        if args.use_head_aware and budget_monitor is not None:
            budget_monitor.check_and_evict(
                stream_model.source_key_values if isinstance(stream_model.source_key_values, HeadAwareDynamicCache) else None,
                group_tracker
            )
        
        if inference_mode == "streaming":
            # 修复延迟计算
            # 计算源端和目标端的单词数（不包括instruct tokens）
            source_txt_words = source_txt[0].split() if source_txt and len(source_txt) > 0 else []
            target_txt_words = target_txt[0].split() if target_txt and len(target_txt) > 0 else []
            output_text_words = output_text.split()
            
            # source_length 是源端实际单词数（不包括instruct）
            source_length = len(source_txt_words)
            # target_length 是目标端实际单词数（用于计算）
            target_length = len(output_text_words)
            
            # wait_lagging 记录的是 source_words（源端单词索引，从wait_k开始）
            # 延迟 = 生成第i个目标词时，已经读取的源端单词数
            
            # 总是打印调试信息（用于诊断）
            if accelerator.is_main_process:
                print(f"\n[DEBUG] Step {step} - Delay Calculation:")
                print(f"  Source length (words): {source_length}")
                print(f"  Target length (words): {target_length}")
                print(f"  Wait_lagging is None: {wait_lagging is None}")
                if wait_lagging is not None:
                    print(f"  Wait_lagging length: {len(wait_lagging)}")
                    if len(wait_lagging) > 0:
                        print(f"  Wait_lagging (first 10): {wait_lagging[:10]}")
                        print(f"  Wait_lagging (last 10): {wait_lagging[-10:]}")
                        print(f"  Wait_lagging min/max: {min(wait_lagging)}/{max(wait_lagging)}")
                    else:
                        print(f"  Wait_lagging is empty!")
                else:
                    print(f"  Wait_lagging is None!")
                logging.info(f"[DEBUG] Step {step} - wait_lagging: {wait_lagging}")
            
            if wait_lagging is not None and len(wait_lagging) > 0:
                # wait_lagging记录的是每个token的延迟，不是每个word的延迟
                # 对于streaming generation，我们需要使用所有token的延迟值
                # 但target_length是word数，所以我们需要调整
                # 实际上，delays应该对应target tokens，而不是target words
                # 所以我们应该使用wait_lagging的所有值，而不是截断到target_length
                # 但为了计算AL/LAAL，我们需要知道实际的target token数
                # 暂时使用wait_lagging的所有值，target_length使用len(wait_lagging)
                wait_lagging_valid = wait_lagging  # 使用所有延迟值
                # 但计算时使用实际的token数作为target_length
                target_length_for_calc = len(wait_lagging)  # 使用token数而不是word数
                
                # wait_lagging应该是源端单词索引，不应该超过source_length
                if wait_lagging_valid and len(wait_lagging_valid) > 0:
                    # 检查并修复异常值
                    max_delay = max(wait_lagging_valid) if wait_lagging_valid else 0
                    min_delay = min(wait_lagging_valid) if wait_lagging_valid else 0
                    
                    # 如果延迟值异常（负数或过大），进行修复
                    if min_delay < 0 or max_delay > source_length * 2:
                        if accelerator.is_main_process:
                            print(f"[WARNING] Abnormal delay values detected: min={min_delay}, max={max_delay}, source_length={source_length}")
                        # 修复：限制在合理范围内 [0, source_length]
                        wait_lagging_valid = [max(0, min(int(d), source_length)) for d in wait_lagging_valid]
                    
                    # 确保所有值都是整数且非负
                    wait_lagging_valid = [int(max(0, d)) for d in wait_lagging_valid]
                    
                    try:
                        # 使用token数而不是word数来计算AL/LAAL
                        # 因为wait_lagging记录的是每个token的延迟
                        al, laal = calculate_al_and_laal(
                            source_length,
                            target_length_for_calc,  # 使用token数
                            wait_lagging_valid
                        )
                        
                        if accelerator.is_main_process and step == 0:
                            print(f"  [DEBUG] Using target_length={target_length_for_calc} (tokens) for AL/LAAL calculation")
                            print(f"  [DEBUG] source_length={source_length}, wait_lagging_valid length={len(wait_lagging_valid)}")
                        # 确保结果非负
                        al = max(0.0, al)
                        laal = max(0.0, laal)
                        AL.append(al)
                        LAAL.append(laal)
                        
                        if accelerator.is_main_process and step == 0:
                            print(f"  Calculated AL: {al:.2f}, LAAL: {laal:.2f}")
                    except Exception as e:
                        if accelerator.is_main_process:
                            print(f"[ERROR] Failed to calculate AL/LAAL: {e}")
                            print(f"  source_length={source_length}, target_length={target_length}")
                            print(f"  wait_lagging_valid length={len(wait_lagging_valid)}")
                            if len(wait_lagging_valid) > 0:
                                print(f"  wait_lagging_valid sample: {wait_lagging_valid[:10]}")
                        AL.append(0)
                        LAAL.append(0)
                else:
                    if accelerator.is_main_process:
                        print(f"[WARNING] Empty wait_lagging_valid for step {step}")
                    AL.append(0)
                    LAAL.append(0)
            else:
                if accelerator.is_main_process:
                    print(f"[WARNING] No wait_lagging for step {step}")
                AL.append(0)
                LAAL.append(0)
        
        if accelerator.is_main_process:
            print(f"\nStep {step}:")
            print(f"  Input: {source_txt[0][:200] if source_txt and len(source_txt) > 0 else 'N/A'}...")
            print(f"  Target: {target_txt[0][:200] if target_txt and len(target_txt) > 0 else 'N/A'}...")
            print(f"  Output: {output_text[:200]}...")
            output_text_len = len(output_text.split())
            print(f"  Output Length: {seq_len} tokens ({output_text_len} words)" if inference_mode == "streaming" else f"  Output Length: {seq_len} tokens")
            if args.use_head_aware:
                print(f"  Cache Memory: {stats['cache_memory_gb'][-1]:.2f}GB" if stats['cache_memory_gb'] else "  Cache Memory: 0.00GB")
            
            # 详细输出生成质量信息（用于诊断BLEU=0问题）
            logging.info(f"\n=== Step {step} Generation Details ===")
            logging.info(f"Source (first 500 chars): {source_txt[0][:500] if source_txt and len(source_txt) > 0 else 'N/A'}")
            logging.info(f"Target (first 500 chars): {target_txt[0][:500] if target_txt and len(target_txt) > 0 else 'N/A'}")
            logging.info(f"Output (first 500 chars): {output_text[:500]}")
            logging.info(f"Source length: {len(source_txt[0]) if source_txt and len(source_txt) > 0 else 0} chars, {len(source_txt[0].split()) if source_txt and len(source_txt) > 0 else 0} words")
            logging.info(f"Target length: {len(target_txt[0]) if target_txt and len(target_txt) > 0 else 0} chars, {len(target_txt[0].split()) if target_txt and len(target_txt) > 0 else 0} words")
            logging.info(f"Output length: {len(output_text)} chars, {output_text_len} words")
            logging.info(f"Output tokens: {seq_len}")
    
    # 计算最终指标
    # 确保输出和目标都是列表格式
    if len(output_text_lt) == 0:
        logging.warning("No output texts generated!")
        bleu_score = 0.0
    else:
        # 检查目标文本格式
        if isinstance(target_txt_lt[0], list):
            # 如果target_txt_lt是嵌套列表，需要展平
            target_txt_lt = [item for sublist in target_txt_lt for item in sublist]
        
        # 确保长度匹配
        min_len = min(len(output_text_lt), len(target_txt_lt))
        if min_len == 0:
            logging.warning("Empty target or output lists!")
            bleu_score = 0.0
        else:
            output_text_lt = output_text_lt[:min_len]
            target_txt_lt = target_txt_lt[:min_len]
            
            # 记录前几个样本用于调试
            logging.info(f"\n=== BLEU Calculation Debug ===")
            logging.info(f"Number of samples: {len(output_text_lt)}")
            for i in range(min(3, len(output_text_lt))):
                logging.info(f"\nSample {i}:")
                logging.info(f"  Target: {target_txt_lt[i][:200] if len(target_txt_lt[i]) > 200 else target_txt_lt[i]}")
                logging.info(f"  Output: {output_text_lt[i][:200] if len(output_text_lt[i]) > 200 else output_text_lt[i]}")
            
            try:
                bleu = sacrebleu.corpus_bleu(output_text_lt, [target_txt_lt])
                bleu_score = bleu.score
                logging.info(f"BLEU score: {bleu_score:.2f}")
            except Exception as e:
                logging.error(f"Failed to calculate BLEU: {e}")
                logging.error(f"Output texts count: {len(output_text_lt)}")
                logging.error(f"Target texts count: {len(target_txt_lt)}")
                bleu_score = 0.0
    
    # 内存统计
    memory_stats = memory_monitor.get_stats()
    logging.info(f"Peak GPU Memory: {memory_stats['peak_memory_gb']:.2f}GB")
    logging.info(f"Average GPU Memory: {memory_stats['avg_memory_gb']:.2f}GB")
    
    if args.use_head_aware:
        if stats['cache_memory_gb']:
            logging.info(f"Average Cache Memory: {sum(stats['cache_memory_gb'])/len(stats['cache_memory_gb']):.2f}GB")
            logging.info(f"Peak Cache Memory: {max(stats['cache_memory_gb']):.2f}GB")
    
    if inference_mode == "streaming":
        avg_LAAL = sum(LAAL) / len(LAAL) if LAAL else 0
        avg_AL = sum(AL) / len(AL) if AL else 0
        logging.info(f"Average AL: {avg_AL:.2f}")
        logging.info(f"Average LAAL: {avg_LAAL:.2f}")
    
    # 保存结果
    results = {
        'bleu_score': bleu_score,
        'memory_stats': memory_stats,
        'cache_stats': {
            'avg_cache_memory_gb': sum(stats['cache_memory_gb'])/len(stats['cache_memory_gb']) if stats['cache_memory_gb'] else 0,
            'peak_cache_memory_gb': max(stats['cache_memory_gb']) if stats['cache_memory_gb'] else 0,
        },
        'length_stats': {
            'total_tokens': stats['total_tokens'],
            'max_length': stats['max_length'],
            'avg_length': stats['total_tokens'] / len(output_text_lt) if output_text_lt else 0,
        },
        'latency_stats': {
            'avg_inference_time': sum(stats['inference_times']) / len(stats['inference_times']) if stats['inference_times'] else 0,
        },
        'streaming_stats': {
            'avg_AL': avg_AL if inference_mode == "streaming" else 0,
            'avg_LAAL': avg_LAAL if inference_mode == "streaming" else 0,
        } if inference_mode == "streaming" else {}
    }
    
    results_file = f"{args.output_dir}/results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    print(f"BLEU Score: {bleu_score:.2f}")
    print(f"Peak GPU Memory: {memory_stats['peak_memory_gb']:.2f}GB")


if __name__ == "__main__":
    main()

