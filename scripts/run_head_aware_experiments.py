#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Head-Aware 实验一键运行脚本
运行两个实验：
1. Head-Aware only
2. Head-Aware + Group-Aware (Full方法)
"""
import os
import sys
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def print_header(title, char="="):
    """打印标题"""
    print("\n" + char * 80)
    print(f"  {title}")
    print(char * 80)

def print_section(title):
    """打印章节标题"""
    print("\n" + "-" * 80)
    print(f"  {title}")
    print("-" * 80)

def check_environment():
    """检查环境"""
    print_header("环境检查")
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version < (3, 8):
        print("❌ 需要Python 3.8+")
        return False
    else:
        print("✅ Python版本符合要求")
    
    # 检查CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
            print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("⚠️  CUDA不可用，将使用CPU（速度较慢）")
    except ImportError:
        print("⚠️  PyTorch未安装，但可能不影响运行")
    
    # 检查必要的模块
    try:
        from StreamingLLM_GPE.models.Qwen2_5.head_aware_cache import HeadAwareDynamicCache
        print("✅ HeadAwareDynamicCache 导入成功")
    except ImportError as e:
        print(f"❌ HeadAwareDynamicCache 导入失败: {e}")
        return False
    
    try:
        from StreamingLLM_GPE.utils.head_analyzer import HeadAnalyzer
        print("✅ HeadAnalyzer 导入成功")
    except ImportError as e:
        print(f"❌ HeadAnalyzer 导入失败: {e}")
        return False
    
    print()
    return True

def check_model(model_path):
    """检查模型是否存在"""
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        return False
    
    # 检查关键文件
    config_file = os.path.join(model_path, 'config.json')
    if not os.path.exists(config_file):
        print(f"⚠️  config.json 不存在，但继续运行")
    
    print(f"✅ 模型路径有效: {model_path}")
    return True

def run_experiment(
    experiment_name,
    model_path,
    output_dir,
    max_samples,
    total_budget,
    use_group_aware=False,
    quantization="4bit",
    device=0,
    min_source_length=3000
):
    """运行单个实验"""
    print_section(f"运行实验: {experiment_name}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建命令
    cmd = [
        "python", "StreamingLLM_GPE/evaluate/multi_model_eval.py",
        "--LLM_backbone", "Qwen",
        "--LLM_path", model_path,
        "--use_head_aware",
        "--total_budget", str(total_budget),
        "--output_dir", output_dir,
        "--max_samples", str(max_samples),
        "--quantization", quantization,
        "--device", str(device),
        "--min_source_length", str(min_source_length),
        "--max_new_tokens", "512"
    ]
    
    if use_group_aware:
        cmd.append("--use_group_aware")
    
    print(f"配置:")
    print(f"  实验名称: {experiment_name}")
    print(f"  模型路径: {model_path}")
    print(f"  输出目录: {output_dir}")
    print(f"  样本数: {max_samples}")
    print(f"  预算: {total_budget} tokens/layer")
    print(f"  Group-Aware: {'是' if use_group_aware else '否'}")
    print(f"  量化: {quantization}")
    print(f"  最小源长度: {min_source_length} words")
    print()
    
    start_time = time.time()
    
    try:
        # 运行实验
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=False,  # 实时显示输出
            text=True,
            timeout=7200  # 2小时超时
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n✅ 实验 '{experiment_name}' 完成！耗时: {elapsed_time/60:.1f} 分钟")
            return True
        else:
            print(f"\n⚠️  实验 '{experiment_name}' 返回代码: {result.returncode}")
            print(f"   耗时: {elapsed_time/60:.1f} 分钟")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n⚠️  实验 '{experiment_name}' 超时（超过2小时）")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  实验 '{experiment_name}' 被用户中断")
        return False
    except Exception as e:
        print(f"\n❌ 实验 '{experiment_name}' 运行出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_results(output_dir, experiment_name):
    """分析实验结果"""
    results_file = os.path.join(output_dir, "results.json")
    
    if not os.path.exists(results_file):
        print(f"⚠️  结果文件不存在: {results_file}")
        return None
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # 提取关键指标
        metrics = {
            'experiment': experiment_name,
            'bleu': results.get('bleu_score', 0),
            'peak_memory_gb': 0,
            'avg_memory_gb': 0,
            'avg_cache_memory_gb': 0,
            'peak_cache_memory_gb': 0,
            'avg_AL': 0,
            'avg_LAAL': 0,
        }
        
        if 'memory_stats' in results:
            mem_stats = results['memory_stats']
            metrics['peak_memory_gb'] = mem_stats.get('peak_memory_gb', 0)
            metrics['avg_memory_gb'] = mem_stats.get('avg_memory_gb', 0)
        
        if 'cache_stats' in results:
            cache_stats = results['cache_stats']
            metrics['avg_cache_memory_gb'] = cache_stats.get('avg_cache_memory_gb', 0)
            metrics['peak_cache_memory_gb'] = cache_stats.get('peak_cache_memory_gb', 0)
        
        if 'streaming_stats' in results:
            stream_stats = results['streaming_stats']
            metrics['avg_AL'] = stream_stats.get('avg_AL', 0)
            metrics['avg_LAAL'] = stream_stats.get('avg_LAAL', 0)
        
        return metrics
        
    except Exception as e:
        print(f"⚠️  结果分析失败: {e}")
        return None

def print_comparison(results_list):
    """打印对比结果"""
    print_header("实验结果对比")
    
    if not results_list or len(results_list) < 2:
        print("⚠️  结果不足，无法对比")
        return
    
    print(f"{'指标':<25} {'Head-Aware':<20} {'Head-Aware+Group-Aware':<25} {'提升':<15}")
    print("-" * 85)
    
    ha_result = results_list[0]
    full_result = results_list[1]
    
    # BLEU对比
    ha_bleu = ha_result.get('bleu', 0)
    full_bleu = full_result.get('bleu', 0)
    bleu_improvement = ((full_bleu - ha_bleu) / ha_bleu * 100) if ha_bleu > 0 else 0
    print(f"{'BLEU分数':<25} {ha_bleu:<20.4f} {full_bleu:<25.4f} {bleu_improvement:+.2f}%")
    
    # 内存对比
    ha_peak = ha_result.get('peak_memory_gb', 0)
    full_peak = full_result.get('peak_memory_gb', 0)
    if ha_peak > 0 and full_peak > 0:
        mem_change = ((full_peak - ha_peak) / ha_peak * 100)
        print(f"{'峰值内存 (GB)':<25} {ha_peak:<20.2f} {full_peak:<25.2f} {mem_change:+.2f}%")
    
    # Cache内存对比
    ha_cache = ha_result.get('peak_cache_memory_gb', 0)
    full_cache = full_result.get('peak_cache_memory_gb', 0)
    if ha_cache > 0 and full_cache > 0:
        cache_change = ((full_cache - ha_cache) / ha_cache * 100)
        print(f"{'峰值Cache内存 (GB)':<25} {ha_cache:<20.4f} {full_cache:<25.4f} {cache_change:+.2f}%")
    
    # 延迟对比
    ha_al = ha_result.get('avg_AL', 0)
    full_al = full_result.get('avg_AL', 0)
    if ha_al > 0 and full_al > 0:
        al_change = ((full_al - ha_al) / ha_al * 100)
        print(f"{'平均延迟 (AL)':<25} {ha_al:<20.2f} {full_al:<25.2f} {al_change:+.2f}%")
    
    print()
    
    # 总结
    print("总结:")
    if full_bleu > ha_bleu:
        print(f"  ✅ Full方法BLEU提升 {bleu_improvement:.2f}%")
    elif full_bleu < ha_bleu:
        print(f"  ⚠️  Full方法BLEU下降 {abs(bleu_improvement):.2f}%")
    else:
        print(f"  ➡️  BLEU分数相同")
    
    if full_peak > 0 and ha_peak > 0:
        if full_peak < ha_peak:
            print(f"  ✅ Full方法内存减少 {abs(mem_change):.2f}%")
        elif full_peak > ha_peak:
            print(f"  ⚠️  Full方法内存增加 {mem_change:.2f}%")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Head-Aware 实验一键运行脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认配置（快速验证，10个样本）
  python scripts/run_head_aware_experiments.py --model-path ./models/Qwen2.5-3B-Instruct
  
  # 完整验证（100个样本）
  python scripts/run_head_aware_experiments.py --model-path ./models/Qwen2.5-3B-Instruct --max-samples 100
  
  # 自定义预算
  python scripts/run_head_aware_experiments.py --model-path ./models/Qwen2.5-3B-Instruct --total-budget 4096
        """
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="模型路径（例如: ./models/Qwen2.5-3B-Instruct）"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output_logs/head_aware_experiments",
        help="输出目录（默认: ./output_logs/head_aware_experiments）"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="最大样本数（默认: 10，用于快速验证）"
    )
    
    parser.add_argument(
        "--total-budget",
        type=int,
        default=2048,
        help="KV cache预算 per layer（默认: 2048）"
    )
    
    parser.add_argument(
        "--quantization",
        type=str,
        default="4bit",
        choices=["4bit", "8bit", "none"],
        help="量化策略（默认: 4bit）"
    )
    
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU设备ID（默认: 0）"
    )
    
    parser.add_argument(
        "--min-source-length",
        type=int,
        default=3000,
        help="最小源序列长度（words，默认: 3000）"
    )
    
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="跳过环境检查"
    )
    
    args = parser.parse_args()
    
    # 打印欢迎信息
    print_header("Head-Aware 实验一键运行脚本")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 环境检查
    if not args.skip_check:
        if not check_environment():
            print("❌ 环境检查失败，请修复后重试")
            return 1
    
    # 模型检查
    if not check_model(args.model_path):
        print("❌ 模型检查失败，请检查模型路径")
        return 1
    
    # 创建输出目录
    base_output_dir = args.output_dir
    os.makedirs(base_output_dir, exist_ok=True)
    
    # 实验配置
    experiments = [
        {
            'name': 'Head-Aware',
            'output_dir': os.path.join(base_output_dir, 'head_aware_only'),
            'use_group_aware': False
        },
        {
            'name': 'Head-Aware + Group-Aware (Full)',
            'output_dir': os.path.join(base_output_dir, 'head_aware_group_aware'),
            'use_group_aware': True
        }
    ]
    
    # 运行实验
    results_list = []
    total_start_time = time.time()
    
    for i, exp_config in enumerate(experiments, 1):
        print_header(f"实验 {i}/{len(experiments)}: {exp_config['name']}")
        
        success = run_experiment(
            experiment_name=exp_config['name'],
            model_path=args.model_path,
            output_dir=exp_config['output_dir'],
            max_samples=args.max_samples,
            total_budget=args.total_budget,
            use_group_aware=exp_config['use_group_aware'],
            quantization=args.quantization,
            device=args.device,
            min_source_length=args.min_source_length
        )
        
        # 分析结果
        metrics = analyze_results(exp_config['output_dir'], exp_config['name'])
        if metrics:
            results_list.append(metrics)
        
        # 如果不是最后一个实验，等待一下
        if i < len(experiments):
            print("\n等待5秒后继续下一个实验...")
            time.sleep(5)
    
    total_elapsed = time.time() - total_start_time
    
    # 打印对比结果
    if results_list:
        print_comparison(results_list)
        
        # 保存对比结果
        comparison_file = os.path.join(base_output_dir, 'comparison.json')
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'model_path': args.model_path,
                    'max_samples': args.max_samples,
                    'total_budget': args.total_budget,
                    'quantization': args.quantization,
                },
                'results': results_list,
                'total_time_minutes': total_elapsed / 60
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n对比结果已保存到: {comparison_file}")
    
    # 总结
    print_header("实验完成")
    print(f"总耗时: {total_elapsed/60:.1f} 分钟")
    print(f"结果目录: {base_output_dir}")
    print()
    print("实验结果:")
    for exp_config in experiments:
        results_file = os.path.join(exp_config['output_dir'], 'results.json')
        if os.path.exists(results_file):
            print(f"  ✅ {exp_config['name']}: {results_file}")
        else:
            print(f"  ⚠️  {exp_config['name']}: 结果文件不存在")
    print()
    print("下一步:")
    print("1. 查看对比结果了解两个方法的差异")
    print("2. 如果效果满意，可以运行完整实验:")
    print("   bash run_a_level_experiments.sh")
    print("   或")
    print("   python scripts/windows/run_a_level_experiments.ps1")
    print("3. 或者增加样本数进行更全面的验证:")
    print(f"   python scripts/run_head_aware_experiments.py --model-path {args.model_path} --max-samples 100")
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

