"""
多模型评估脚本
支持: Qwen, Llama, Gemma, Phi3
用于会议/期刊的多模型验证
"""
import os
import sys
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# ============================================
# 禁用HuggingFace数据集缓存，确保使用最新的数据文件
os.environ["HF_DATASETS_DISABLE_CACHE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NO_TF"] = "1"
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
from StreamingLLM_GPE.models.Llama3.llama_streaming import LlamaForCausalLM_stream
from StreamingLLM_GPE.models.Gemma2.gemma2_streaming import Gemma2ForCausalLM_stream
from StreamingLLM_GPE.models.Qwen2_5.head_aware_cache import HeadAwareDynamicCache as QwenHeadAwareCache
from StreamingLLM_GPE.models.Llama3.head_aware_cache import HeadAwareDynamicCache as LlamaHeadAwareCache
from StreamingLLM_GPE.models.Gemma2.head_aware_cache import HeadAwareDynamicCache as GemmaHeadAwareCache
# HAQ-KV support
try:
    from StreamingLLM_GPE.models.Qwen2_5.haq_kv_cache import HAQKVCache as QwenHAQKVCache
except ImportError:
    QwenHAQKVCache = None
from StreamingLLM_GPE.utils.head_analyzer import HeadAnalyzer
from StreamingLLM_GPE.utils.budget_monitor import BudgetMonitor
try:
    from StreamingLLM_GPE.utils.tokenizer_utils import normalize_tokenizer_eos_token, get_all_eos_token_ids
except ImportError:
    # Fallback: 如果导入失败，定义简单的 fallback 函数
    def normalize_tokenizer_eos_token(tokenizer, model_name=None):
        eos_ids = set()
        if tokenizer.eos_token_id is not None:
            if isinstance(tokenizer.eos_token_id, list):
                eos_ids.update(tokenizer.eos_token_id)
            else:
                eos_ids.add(tokenizer.eos_token_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return eos_ids
    def get_all_eos_token_ids(tokenizer):
        eos_ids = set()
        if tokenizer.eos_token_id is not None:
            if isinstance(tokenizer.eos_token_id, list):
                eos_ids.update(tokenizer.eos_token_id)
            else:
                eos_ids.add(tokenizer.eos_token_id)
        return eos_ids
import importlib.util

_utils_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils.py')
_spec = importlib.util.spec_from_file_location("StreamingLLM_GPE.utils_module", _utils_file_path)
utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(utils)
from StreamingLLM_GPE.evaluate.lagging import calculate_al_and_laal
import sacrebleu

# 模型类映射
MODEL_CLASSES = {
    'Qwen': Qwen2ForCausalLM_stream,
    'Llama': LlamaForCausalLM_stream,
    'Gemma': Gemma2ForCausalLM_stream,
}


class MemoryMonitor:
    """监控GPU内存使用"""

    def __init__(self, device=0):
        self.device = device
        self.peak_memory = 0
        self.memory_history = []

    def record(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            # 使用 max_memory_allocated 获取峰值
            current_peak = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)  # GB
            current_alloc = torch.cuda.memory_allocated(self.device) / (1024 ** 3)

            self.memory_history.append(current_alloc)
            self.peak_memory = max(self.peak_memory, current_peak)

            return current_alloc
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
    parser.add_argument("--inference_mode", type=str, default="streaming", choices=["batch", "streaming"])
    parser.add_argument("--LLM_backbone", type=str, default="Qwen", choices=list(MODEL_CLASSES.keys()),
                        help="Model architecture")
    parser.add_argument("--LLM_path", type=str, required=True, help="Path to the LLM model.")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--wait_k", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default='./output_logs')
    parser.add_argument("--split_mode", type=str, default="word", choices=["word", "token"])
    parser.add_argument("--params", type=str, default="./StreamingLLM_GPE/configs/params_qwen_inference.json")

    # Head-Aware specific arguments
    parser.add_argument("--use_head_aware", action="store_true", help="Use Head-Aware Cache")
    parser.add_argument("--total_budget", type=int, default=2048, help="KV cache budget per layer")
    parser.add_argument("--max_memory_gb", type=float, default=4.0, help="Max KV cache memory (GB)")
    parser.add_argument("--analyze_heads", action="store_true", help="Analyze head functionality")
    
    # HAQ-KV specific arguments
    parser.add_argument("--use_haq_kv", action="store_true", help="Use HAQ-KV (Head-Aware Quantized KV Cache)")
    parser.add_argument("--retrieval_bits", type=int, default=4, help="Quantization bits for Retrieval Heads (2, 4, 8)")
    parser.add_argument("--induction_bits", type=int, default=16, help="Quantization bits for Induction Heads (16=FP16)")
    parser.add_argument("--local_bits", type=int, default=16, help="Quantization bits for Local Heads (16=FP16)")

    # Baseline methods
    parser.add_argument("--use_h2o", action="store_true", help="Use H2O baseline")
    parser.add_argument("--use_streamingllm", action="store_true", help="Use StreamingLLM baseline")
    parser.add_argument("--h2o_budget", type=int, default=2048, help="H2O budget per layer")
    parser.add_argument("--streamingllm_window", type=int, default=512, help="StreamingLLM window size")

    # Multi-model evaluation arguments
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")

    # 强制测试长序列 (默认3000)
    parser.add_argument("--min_source_length", type=int, default=3000, help="Minimum source length in words")

    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens to generate")

    # Quantization options
    parser.add_argument("--quantization", type=str, default="4bit",
                        choices=["4bit", "8bit", "none"],
                        help="Quantization strategy: 4bit (default), 8bit (better performance), none (best performance)")

    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_model_class(model_name):
    """根据模型名称返回模型类"""
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(MODEL_CLASSES.keys())}")
    return MODEL_CLASSES[model_name]


def initialize_head_aware_components(model, config, args):
    """初始化Head-Aware相关组件"""
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    head_analyzer = HeadAnalyzer(num_layers, num_heads, device=device)
    budget_monitor = BudgetMonitor(max_memory_gb=args.max_memory_gb) if args.use_head_aware else None

    return head_analyzer, budget_monitor


def create_cache(head_analyzer, args, model_name='Qwen'):
    """创建KV cache"""
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    # =================  增加 Sink Tokens =================
    if args.use_h2o:
        try:
            from StreamingLLM_GPE.baselines.h2o_cache import H2OCache
            return H2OCache(
                budget_per_layer=args.h2o_budget,
                sink_tokens=128,  # [FIXED] 4 -> 128
                device=device
            )
        except ImportError:
            raise ImportError("H2O baseline not implemented. See BASELINE_IMPLEMENTATION_GUIDE.md")

    if args.use_streamingllm:
        try:
            from StreamingLLM_GPE.baselines.streamingllm_cache import StreamingLLMCache
            return StreamingLLMCache(
                window_size=args.streamingllm_window,
                sink_tokens=128,
                device=device
            )
        except ImportError:
            raise ImportError("StreamingLLM baseline not implemented. See BASELINE_IMPLEMENTATION_GUIDE.md")
    # ================================================================

    if args.use_haq_kv:
        # 使用HAQ-KV (Head-Aware Quantized KV Cache)
        if model_name == 'Qwen' and QwenHAQKVCache is not None:
            cache = QwenHAQKVCache(
                head_analyzer=head_analyzer,
                group_tracker=None,
                total_budget=args.total_budget,
                sink_tokens=128,
                adaptive=True,
                device=device,
                retrieval_bits=args.retrieval_bits,
                induction_bits=args.induction_bits,
                local_bits=args.local_bits,
                use_quantization=True
            )
        else:
            raise ValueError(f"HAQ-KV not supported for {model_name} or not imported")
        return cache
    
    if args.use_head_aware:
        # 根据模型类型选择对应的HeadAwareCache
        if model_name == 'Qwen':
            cache = QwenHeadAwareCache(
                head_analyzer=head_analyzer,
                group_tracker=None,
                total_budget=args.total_budget,
                sink_tokens=128,
                adaptive=True,
                device=device
            )
        elif model_name == 'Llama':
            cache = LlamaHeadAwareCache(
                head_analyzer=head_analyzer,
                group_tracker=None,
                total_budget=args.total_budget,
                sink_tokens=128,
                adaptive=True,
                device=device
            )
        elif model_name == 'Gemma':
            cache = GemmaHeadAwareCache(
                head_analyzer=head_analyzer,
                group_tracker=None,
                total_budget=args.total_budget,
                sink_tokens=128,
                adaptive=True,
                device=device
            )
        else:
            # 默认使用Qwen的实现
            cache = QwenHeadAwareCache(
                head_analyzer=head_analyzer,
                group_tracker=None,
                total_budget=args.total_budget,
                sink_tokens=128,
                adaptive=True,
                device=device
            )
        return cache
    else:
        # 根据模型类型选择DynamicCache
        if model_name == 'Qwen':
            from StreamingLLM_GPE.models.Qwen2_5.qwen_streaming import DynamicCache
            return DynamicCache()
        elif model_name == 'Llama':
            from StreamingLLM_GPE.models.Llama3.llama_streaming import DynamicCache
            return DynamicCache()
        elif model_name == 'Gemma':
            from StreamingLLM_GPE.models.Gemma2.gemma2_streaming import DynamicCache
            return DynamicCache()
        else:
            from StreamingLLM_GPE.models.Qwen2_5.qwen_streaming import DynamicCache
            return DynamicCache()
def main():
    args = get_args()
    params = utils.Params(args.params)
    setup_seed(0)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # 保存每个样本的输出目录（按模式分组：batch / streaming）
    samples_output_root = os.path.join(args.output_dir, "samples")
    os.makedirs(samples_output_root, exist_ok=True)
    # 设置日志
    log_file = f"{args.output_dir}/multi_model_eval.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    config_str = f"""
    Multi-Model Evaluation Configuration (FIXED PROMPT):
    - Model Architecture: {args.LLM_backbone}
    - Model Path: {args.LLM_path}
    - Inference Mode: {args.inference_mode}
    - Head-Aware: {args.use_head_aware}
    - HAQ-KV: {args.use_haq_kv}
    - H2O: {args.use_h2o}
    - StreamingLLM: {args.use_streamingllm}
    - Max Memory: {args.max_memory_gb} GB
    - Wait-k: {args.wait_k}
    - Min Source Length: {args.min_source_length}
    - Max New Tokens: {args.max_new_tokens}
    """
    print(config_str)
    logging.info(config_str)
    # 获取模型类
    ModelClass = get_model_class(args.LLM_backbone)
    # 加载模型
    config = AutoConfig.from_pretrained(args.LLM_path)
    config._attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(args.LLM_path, padding_side='right', config=config)

    # ================= [通用修复] 标准化 EOS Token 配置 =================
    # 使用通用工具函数自动处理所有模型的 EOS token 配置
    all_eos_ids = normalize_tokenizer_eos_token(tokenizer, model_name=args.LLM_backbone)
    print(f"[{args.LLM_backbone}] EOS token ID: {tokenizer.eos_token_id}")
    print(f"[{args.LLM_backbone}] All EOS token IDs: {get_all_eos_token_ids(tokenizer)}")
    # =====================================================================

    # 量化配置
    quantization_config = None
    torch_dtype = torch.bfloat16

    if args.quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif args.quantization == "8bit":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )
    # 加载模型
    model_kwargs = {
        "ignore_mismatched_sizes": True,
        "config": config,
        "device_map": "auto"
    }

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = torch_dtype
    model = ModelClass.from_pretrained(
        args.LLM_path,
        **model_kwargs
    )
    if args.lora_path is not None:
        model = peft.PeftModel.from_pretrained(model, args.lora_path)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id
    # 初始化Head-Aware组件
    head_analyzer, budget_monitor = initialize_head_aware_components(
        model, config, args
    )
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
    initial_size = len(data_collator_dataset)
    logging.info(f"Loaded dataset with {initial_size} samples from {params.file_path}")

    if initial_size == 0:
        raise ValueError(f"No samples available in data file: {params.file_path}.")
    # 过滤短序列
    if args.min_source_length > 0:
        def filter_long_sequences(example):
            source_txt = example.get("source_txt", "")
            if not source_txt:
                return False
            source_words = source_txt.split()
            return len(source_words) >= args.min_source_length

        before_filter_size = len(data_collator_dataset)
        data_collator_dataset = data_collator_dataset.filter(filter_long_sequences)
        logging.info(f"Filtered dataset: {before_filter_size} -> {len(data_collator_dataset)} samples")
    if args.max_samples is not None and args.max_samples > 0:
        data_collator_dataset = data_collator_dataset.select(range(min(args.max_samples, len(data_collator_dataset))))

    dataloader = DataLoader(
        data_collator_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=data_collator.collate_fn_inference
    )
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    use_accelerate = False
    if quantization_config is not None:
        stream_model = model
        try:
            device = next(model.parameters()).device
        except:
            device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu")
    else:
        accelerator = Accelerator(mixed_precision="bf16")
        stream_model, dataloader = accelerator.prepare(model, dataloader)
        device = accelerator.device
        use_accelerate = True

    target_txt_lt = []
    output_text_lt = []
    AL = []
    LAAL = []
    memory_monitor = MemoryMonitor(device=args.device)

    stats = {
        'total_tokens': 0,
        'max_length': 0,
        'cache_memory_gb': [],
        'inference_times': []
    }
    # ================= [CRITICAL FIX START] PHASE DE CALIBRATION =================
    if (args.use_head_aware or args.use_haq_kv) and args.analyze_heads:
        print("\n[Calibration] Starting Head Analysis on 10 samples...")
        logging.info("[Calibration] Starting Head Analysis...")
        # 使用前10个样本进行校准
        calibration_subset = data_collator_dataset.select(range(min(10, len(data_collator_dataset))))
        # 创建临时的 DataLoader
        calibration_loader = DataLoader(
            calibration_subset,
            batch_size=1,
            shuffle=False,
            collate_fn=data_collator.collate_fn_inference
        )
        calibration_inputs = []
        for batch in calibration_loader:
            # 为校准构造与正式推理一致的输入（含 chat_template），确保 head 统计可靠
            source_txt = batch.get("source_txt", None)
            input_ids = None
            attn_mask = None

            if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None and source_txt:
                raw_source = source_txt[0] if isinstance(source_txt, list) else source_txt
                system_prompt = "Translate the following English paragraph to French"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": raw_source}
                ]
                try:
                    chat_inputs = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=False,
                        return_tensors="pt"
                    ).to(device)
                    input_ids = chat_inputs
                    attn_mask = torch.ones_like(chat_inputs).to(device)
                except Exception as e:
                    logging.warning(f"Calibration chat_template failed: {e}, fallback to raw tokens.")

            if input_ids is None:
                # 回退：使用数据集原始 tokens + mask
                input_ids = batch.get("source_tokens", None).to(device)
                attn_mask = batch.get("attention_mask", None)
                if attn_mask is not None:
                    attn_mask = attn_mask.to(device)

            calibration_inputs.append({
                "input_ids": input_ids,
                "attention_mask": attn_mask
            })
        head_analyzer.analyze_model(model, calibration_inputs)
        print("[Calibration] Head Analysis Complete.")
        logging.info("[Calibration] Head Analysis Complete.")
    # ================= [CRITICAL FIX END] =================

    stream_model.eval()
    # 获取总样本数，用于判断最后两个样本
    total_samples = len(dataloader)
    print(f"\n[INFO] Total samples to process: {total_samples}")
    
    # 循环评估
    for step, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {args.LLM_backbone}")):
        source_txt = batch.get("source_txt", None)
        target_txt = batch.get("target_txt", None)
        _lengths = batch.get("_lengths", None)
        inference_mode = batch.get("inference_mode", "streaming")
        split_mode = batch.get("split_mode", None)
        _lengths_index = batch.get("_lengths_index", None)
        wait_k = batch.get("wait_k", None)
        input_ids = batch.get("source_tokens", None).to(device)
        attention_mask = batch.get("attention_mask", None).to(device)
        assistant_token = batch.get("assistant_token", None).to(device)
        # ==============================================================================
        # [CRITICAL FIX]: Dynamic Prompt Reconstruction using Chat Template
        # ==============================================================================
        # Only apply for Instruct models if chat_template exists
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
            raw_source = source_txt[0] if isinstance(source_txt, list) else source_txt
            system_prompt = "Translate the following English paragraph to French"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_source}
            ]

            try:
                full_input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(device)
                source_input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=False,
                    return_tensors="pt"
                ).to(device)
                assistant_token_len = full_input_ids.shape[1] - source_input_ids.shape[1]
                if assistant_token_len > 0:
                    assistant_token = full_input_ids[0, source_input_ids.shape[1]:]
                    input_ids = source_input_ids
                    attention_mask = torch.ones_like(input_ids).to(device)
                    if _lengths:
                        _lengths[0]['source_token_len'] = input_ids.shape[1]
                        _lengths[0]['input_batch_len'] = input_ids.shape[1]
                        _lengths[0]['source_seg_len'] = [input_ids.shape[1]]
            except Exception as e:
                logging.warning(f"Failed to apply chat template: {e}. Falling back to dataloader inputs.")
        # ==========================================================================
        # 创建 Cache (在获取到正确的 input_ids 之后)
        if args.use_haq_kv or args.use_head_aware:
            cache = create_cache(head_analyzer, args, args.LLM_backbone)
            stream_model.source_key_values = cache
            stream_model.target_key_values = create_cache(head_analyzer, args, args.LLM_backbone)
            stream_model.past_key_values = create_cache(head_analyzer, args, args.LLM_backbone)
        else:
            stream_model.source_key_values = create_cache(head_analyzer, args, args.LLM_backbone)
            stream_model.target_key_values = create_cache(head_analyzer, args, args.LLM_backbone)
            stream_model.past_key_values = create_cache(head_analyzer, args, args.LLM_backbone)

        # Debug print first sample
        if step == 0:
            print(f"\n[DEBUG] Input IDs Length: {input_ids.shape[1]}")
            print(f"[DEBUG] Assistant Token Shape: {assistant_token.shape}")
            # Verify if it looks like Qwen (151644 is <|im_start|>)
            if 151644 in assistant_token.tolist():
                print("[DEBUG] SUCCESS: Detected <|im_start|> in assistant token.")
            else:
                print("[DEBUG] WARNING: <|im_start|> NOT detected in assistant token!")

        # 记录内存
        memory_monitor.record()
        # 推理
        start_time = time.time()
        if inference_mode == "streaming":
            output_sequences, wait_lagging = stream_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
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
                assistant_token=assistant_token,
            )
        elif inference_mode == "batch":
            output_sequences, wait_lagging = stream_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
            )

        inference_time = time.time() - start_time
        stats['inference_times'].append(inference_time)

        if inference_mode == "batch":
            input_token_len = input_ids.shape[1]
            generated_tokens = output_sequences[0][input_token_len:]
        else:
            generated_tokens = output_sequences[0]

        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        target_txt_lt.extend(target_txt)
        output_text_lt.extend([output_text])
        # 保存每个样本的目标与生成文本到文件，按模式存放，便于逐样本对比
        try:
            mode_dir = os.path.join(args.output_dir, "samples", inference_mode)
            os.makedirs(mode_dir, exist_ok=True)
            target_filename = os.path.join(mode_dir, f"sample_{step:04d}_target.txt")
            gen_filename = os.path.join(mode_dir, f"sample_{step:04d}_generated.txt")
            with open(target_filename, "w", encoding="utf-8") as f_t:
                # 使用安全的 target_text（可能为 list 或空）
                f_t.write(target_text if target_text is not None else "")
            with open(gen_filename, "w", encoding="utf-8") as f_g:
                f_g.write(output_text if output_text is not None else "")
            logging.info(f"Saved sample {step} files to {mode_dir}")
        except Exception as e:
            logging.warning(f"Failed to save sample files for step {step}: {e}")

        # ================= [详细打印] 打印第一个样本和最后两个样本的完整目标文本和生成文本 =================
        # 安全获取目标文本
        target_text = ""
        if target_txt and len(target_txt) > 0:
            if isinstance(target_txt[0], list):
                target_text = target_txt[0][0] if len(target_txt[0]) > 0 else ""
            else:
                target_text = target_txt[0]
        
        # 判断是否需要打印完整内容：第一个样本（step == 0）或最后两个样本
        should_print_full = False
        if total_samples >= 3:
            # 如果有3个或更多样本：打印第一个和最后两个
            should_print_full = (step == 0) or (step == total_samples - 2) or (step == total_samples - 1)
        elif total_samples == 2:
            # 如果只有2个样本：打印第一个和最后一个
            should_print_full = (step == 0) or (step == total_samples - 1)
        else:
            # 如果只有1个样本：只打印第一个
            should_print_full = (step == 0)
        
        if should_print_full:
            print(f"\n{'='*80}")
            print(f"Sample {step} - Full Text Comparison")
            if step == 0:
                print("(First Sample)")
            elif step == total_samples - 1:
                print("(Last Sample)")
            elif step == total_samples - 2:
                print("(Second Last Sample)")
            print(f"{'='*80}")
            
            print(f"\n[Target Text] (Length: {len(target_text)} chars):")
            print("-" * 80)
            if target_text:
                print(target_text)
            else:
                print("N/A")
            print("-" * 80)
            
            print(f"\n[Generated Text] (Length: {len(output_text)} chars, Tokens: {len(generated_tokens)}):")
            print("-" * 80)
            print(output_text)
            print("-" * 80)
            
            # 检查是否有明显的重复或垃圾内容
            if target_text and len(output_text) > len(target_text) * 2:
                ratio = len(output_text) / len(target_text)
                print(f"\n[WARNING] Generated text is {ratio:.2f}x longer than target!")
                print("This might indicate a stopping condition issue.")
            
            # 检查是否包含明显的重复模式
            if len(output_text) > 500:
                last_200 = output_text[-200:]
                second_last_200 = output_text[-400:-200] if len(output_text) > 400 else ""
                if second_last_200 and last_200 == second_last_200:
                    print(f"\n[WARNING] Detected repetitive pattern in generated text!")
            
            # 检查是否以EOS token结尾
            if generated_tokens is not None and len(generated_tokens) > 0:
                try:
                    # 安全获取最后一个token ID
                    if isinstance(generated_tokens, torch.Tensor):
                        last_token_id = int(generated_tokens[-1].item())
                    elif isinstance(generated_tokens, list):
                        last_token_id = int(generated_tokens[-1])
                    else:
                        last_token_id = None
                    
                    if last_token_id is not None:
                        all_eos_ids = get_all_eos_token_ids(tokenizer) if tokenizer else set()
                        if last_token_id in all_eos_ids:
                            print(f"\n[INFO] Generation ended with EOS token (ID: {last_token_id})")
                        else:
                            print(f"\n[WARNING] Generation did NOT end with EOS token! Last token ID: {last_token_id}")
                            print(f"Expected EOS token IDs: {all_eos_ids}")
                except Exception as e:
                    print(f"\n[INFO] Could not check EOS token: {e}")
            
            print(f"{'='*80}\n")
        else:
            # 其他样本只打印简要信息
            print(f"Sample {step}: Generated {len(generated_tokens)} tokens, Output length: {len(output_text)} chars")
        
        # 同时记录到日志（只记录前500字符以节省空间）
        logging.info(f"[DEBUG] Sample {step} output:")
        logging.info(f"  Output length: {len(generated_tokens)} tokens")
        logging.info(f"  Output text (first 500 chars): {output_text[:500]}...")
        if target_txt and len(target_txt) > 0:
            logging.info(f"  Target text (first 500 chars): {target_txt[0][:500]}...")
        # ========================================================================

        seq_len = len(generated_tokens)
        stats['total_tokens'] += seq_len
        stats['max_length'] = max(stats['max_length'], seq_len)

        if args.use_head_aware or args.use_haq_kv:
            if hasattr(stream_model.source_key_values, 'get_memory_usage'):
                cache_memory = stream_model.source_key_values.get_memory_usage()
                stats['cache_memory_gb'].append(cache_memory)

        if (args.use_head_aware or args.use_haq_kv) and budget_monitor is not None:
            if hasattr(stream_model.source_key_values, 'get_memory_usage'):
                budget_monitor.check_and_evict(
                    stream_model.source_key_values,
                    group_tracker=None  # Group-Aware已移除
                )

        if inference_mode == "streaming":
            source_txt_words = source_txt[0].split() if source_txt and len(source_txt) > 0 else []
            output_text_words = output_text.split()

            source_length = len(source_txt_words)
            target_length = len(output_text_words)

            if wait_lagging is not None and len(wait_lagging) > 0:
                wait_lagging_valid = wait_lagging
                target_length_for_calc = len(wait_lagging)

                if wait_lagging_valid and len(wait_lagging_valid) > 0:
                    max_delay = max(wait_lagging_valid) if wait_lagging_valid else 0
                    min_delay = min(wait_lagging_valid) if wait_lagging_valid else 0

                    if min_delay < 0 or max_delay > source_length * 2:
                        wait_lagging_valid = [max(0, min(int(d), source_length)) for d in wait_lagging_valid]

                    wait_lagging_valid = [int(max(0, d)) for d in wait_lagging_valid]

                    try:
                        al, laal = calculate_al_and_laal(
                            source_length,
                            target_length_for_calc,
                            wait_lagging_valid
                        )
                        al = max(0.0, al)
                        laal = max(0.0, laal)
                        AL.append(al)
                        LAAL.append(laal)
                    except Exception as e:
                        logging.error(f"Failed to calculate AL/LAAL: {e}")
                        AL.append(0)
                        LAAL.append(0)
                else:
                    AL.append(0)
                    LAAL.append(0)
            else:
                AL.append(0)
                LAAL.append(0)

        if step % 10 == 0 or step == len(dataloader) - 1:
            print(f"\n[{args.LLM_backbone}] Step {step}:")
            print(f"  Output Length: {seq_len} tokens")
            if args.use_head_aware or args.use_haq_kv:
                print(f"  Cache Memory: {stats['cache_memory_gb'][-1]:.2f}GB" if stats[
                    'cache_memory_gb'] else "  Cache Memory: 0.00GB")

    # 计算最终指标
    if len(output_text_lt) == 0:
        logging.warning("No output texts generated!")
        bleu_score = 0.0
    else:
        if isinstance(target_txt_lt[0], list):
            target_txt_lt = [item for sublist in target_txt_lt for item in sublist]

        min_len = min(len(output_text_lt), len(target_txt_lt))
        if min_len == 0:
            logging.warning("Empty target or output lists!")
            bleu_score = 0.0
        else:
            output_text_lt = output_text_lt[:min_len]
            target_txt_lt = target_txt_lt[:min_len]

            logging.info(f"\n=== BLEU Calculation Debug ===")
            logging.info(f"Number of samples: {len(output_text_lt)}")
            for i in range(min(3, len(output_text_lt))):
                logging.info(f"\nSample {i}:")
                logging.info(f"  Target: {target_txt_lt[i][:200] if len(target_txt_lt[i]) > 200 else target_txt_lt[i]}")
                logging.info(
                    f"  Output: {output_text_lt[i][:200] if len(output_text_lt[i]) > 200 else output_text_lt[i]}")

            try:
                # 明确指定 sacrebleu 的 tokenizer，以保证 batch 与 streaming 使用一致的评估设定
                bleu = sacrebleu.corpus_bleu(output_text_lt, [target_txt_lt], tokenize='13a')
                bleu_score = bleu.score
                logging.info(f"BLEU score: {bleu_score:.2f}")
            except Exception as e:
                logging.error(f"Failed to calculate BLEU: {e}")
                bleu_score = 0.0

    memory_stats = memory_monitor.get_stats()
    logging.info(f"Peak GPU Memory: {memory_stats['peak_memory_gb']:.2f}GB")
    logging.info(f"Average GPU Memory: {memory_stats['avg_memory_gb']:.2f}GB")

    if args.use_head_aware or args.use_haq_kv:
        if stats['cache_memory_gb']:
            avg_cache = sum(stats['cache_memory_gb']) / len(stats['cache_memory_gb'])
            peak_cache = max(stats['cache_memory_gb'])
            logging.info(f"Average Cache Memory: {avg_cache:.4f}GB")
            logging.info(f"Peak Cache Memory: {peak_cache:.4f}GB")
            if args.use_haq_kv:
                logging.info(f"HAQ-KV Quantization: Retrieval={args.retrieval_bits}bits, "
                           f"Induction={args.induction_bits}bits, Local={args.local_bits}bits")

    if args.inference_mode == "streaming":
        avg_LAAL = sum(LAAL) / len(LAAL) if LAAL else 0
        avg_AL = sum(AL) / len(AL) if AL else 0
        logging.info(f"Average AL: {avg_AL:.2f}")
        logging.info(f"Average LAAL: {avg_LAAL:.2f}")

    results = {
        'model_architecture': args.LLM_backbone,
        'model_path': args.LLM_path,
        'bleu_score': bleu_score,
        'memory_stats': memory_stats,
        'cache_stats': {
            'avg_cache_memory_gb': sum(stats['cache_memory_gb']) / len(stats['cache_memory_gb']) if stats[
                'cache_memory_gb'] else 0,
            'peak_cache_memory_gb': max(stats['cache_memory_gb']) if stats['cache_memory_gb'] else 0,
        },
        'length_stats': {
            'total_tokens': stats['total_tokens'],
            'max_length': stats['max_length'],
            'avg_length': stats['total_tokens'] / len(output_text_lt) if output_text_lt else 0,
        },
        'latency_stats': {
            'avg_inference_time': sum(stats['inference_times']) / len(stats['inference_times']) if stats[
                'inference_times'] else 0,
        },
        'streaming_stats': {
            'avg_AL': avg_AL if args.inference_mode == "streaming" else 0,
            'avg_LAAL': avg_LAAL if args.inference_mode == "streaming" else 0,
        } if args.inference_mode == "streaming" else {}
    }

    results_file = f"{args.output_dir}/results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[{args.LLM_backbone}] Results saved to {results_file}")
    print(f"[{args.LLM_backbone}] BLEU Score: {bleu_score:.2f}")
    print(f"[{args.LLM_backbone}] Peak GPU Memory: {memory_stats['peak_memory_gb']:.2f}GB")

if __name__ == "__main__":
    main()