#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验结果分析和汇总脚本
用于生成论文所需的对比表格和统计信息
"""
import argparse
import json
import os
import sys
from typing import Dict, Optional

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not installed. Install with: pip install pandas")


def load_results(result_dir: str) -> Optional[Dict]:
    """加载实验结果JSON文件"""
    result_file = os.path.join(result_dir, "results.json")
    if not os.path.exists(result_file):
        print(f"Warning: Results file not found: {result_file}")
        return None
    
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {result_file}: {e}")
        return None


def extract_method_name(output_dir: str) -> str:
    """从输出目录名提取方法名称"""
    dir_name = os.path.basename(output_dir.rstrip('/\\'))
    
    # 处理预算实验
    if dir_name.startswith("budget_"):
        budget = dir_name.replace("budget_", "")
        return f"Head-Aware (Budget={budget})"
    
    # 标准方法名称映射
    method_map = {
        "baseline": "Baseline (GPE)",
        "head_aware": "Head-Aware",
        "group_aware": "Group-Aware",
        "full": "Full (Head-Aware + Group-Aware)",
    }
    
    return method_map.get(dir_name, dir_name)


def calculate_performance_drop(baseline_bleu: float, method_bleu: float) -> float:
    """计算性能下降百分比"""
    if baseline_bleu == 0:
        return 0.0
    return ((baseline_bleu - method_bleu) / baseline_bleu) * 100


def calculate_memory_reduction(baseline_memory: float, method_memory: float) -> float:
    """计算内存减少百分比"""
    if baseline_memory == 0:
        return 0.0
    return ((baseline_memory - method_memory) / baseline_memory) * 100


def analyze_experiments(base_dir: str) -> Dict:
    """分析所有实验结果"""
    results = {}
    baseline_results = None
    
    # 遍历所有子目录
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if not os.path.isdir(item_path):
            continue
        
        method_name = extract_method_name(item_path)
        result_data = load_results(item_path)
        
        if result_data is None:
            continue
        
        # 保存baseline结果用于对比
        if "baseline" in item.lower() or "Baseline" in method_name:
            baseline_results = result_data
        
        results[method_name] = {
            'method': method_name,
            'bleu_score': result_data.get('bleu_score', 0.0),
            'peak_memory_gb': result_data.get('memory_stats', {}).get('peak_memory_gb', 0.0),
            'avg_cache_memory_gb': result_data.get('cache_stats', {}).get('avg_cache_memory_gb', 0.0),
            'peak_cache_memory_gb': result_data.get('cache_stats', {}).get('peak_cache_memory_gb', 0.0),
            'avg_length': result_data.get('length_stats', {}).get('avg_length', 0.0),
            'avg_inference_time': result_data.get('latency_stats', {}).get('avg_inference_time', 0.0),
            'avg_AL': result_data.get('streaming_stats', {}).get('avg_AL', 0.0),
            'avg_LAAL': result_data.get('streaming_stats', {}).get('avg_LAAL', 0.0),
            'output_dir': item_path
        }
    
    # 计算相对指标（如果有baseline）
    if baseline_results:
        baseline_bleu = baseline_results.get('bleu_score', 0.0)
        baseline_memory = baseline_results.get('memory_stats', {}).get('peak_memory_gb', 0.0)
        
        for method_name, method_data in results.items():
            if "Baseline" not in method_name:
                method_data['performance_drop'] = calculate_performance_drop(
                    baseline_bleu, method_data['bleu_score']
                )
                method_data['memory_reduction'] = calculate_memory_reduction(
                    baseline_memory, method_data['peak_memory_gb']
                )
            else:
                method_data['performance_drop'] = 0.0
                method_data['memory_reduction'] = 0.0
    
    return results


def print_summary_table(results: Dict):
    """打印汇总表格"""
    print("\n" + "="*100)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*100)
    print()
    
    # 表头
    header = f"{'Method':<30} {'BLEU':<10} {'Memory(GB)':<15} {'Cache(GB)':<15} {'Perf.Drop(%)':<15} {'Mem.Red.(%)':<15}"
    print(header)
    print("-" * 100)
    
    # 数据行
    for method_name, data in sorted(results.items()):
        if "Baseline" in method_name:
            # Baseline行用特殊格式
            row = f"{method_name:<30} {data['bleu_score']:<10.2f} {data['peak_memory_gb']:<15.2f} {data['avg_cache_memory_gb']:<15.4f} {data.get('performance_drop', 0.0):<15.2f} {data.get('memory_reduction', 0.0):<15.2f}"
            print(row)
        else:
            row = f"{method_name:<30} {data['bleu_score']:<10.2f} {data['peak_memory_gb']:<15.2f} {data['avg_cache_memory_gb']:<15.4f} {data.get('performance_drop', 0.0):<15.2f} {data.get('memory_reduction', 0.0):<15.2f}"
            print(row)
    
    print("-" * 100)
    print()


def print_detailed_table(results: Dict):
    """打印详细表格（包含所有指标）"""
    print("\n" + "="*120)
    print("DETAILED EXPERIMENT RESULTS")
    print("="*120)
    print()
    
    # 表头
    header = f"{'Method':<30} {'BLEU':<8} {'Mem(GB)':<10} {'Cache(GB)':<12} {'AvgLen':<10} {'Time(s)':<10} {'AL':<8} {'LAAL':<8} {'Perf.Drop':<10} {'Mem.Red.':<10}"
    print(header)
    print("-" * 120)
    
    # 数据行
    for method_name, data in sorted(results.items()):
        row = (f"{method_name:<30} "
               f"{data['bleu_score']:<8.2f} "
               f"{data['peak_memory_gb']:<10.2f} "
               f"{data['avg_cache_memory_gb']:<12.4f} "
               f"{data['avg_length']:<10.1f} "
               f"{data['avg_inference_time']:<10.2f} "
               f"{data['avg_AL']:<8.2f} "
               f"{data['avg_LAAL']:<8.2f} "
               f"{data.get('performance_drop', 0.0):<10.2f} "
               f"{data.get('memory_reduction', 0.0):<10.2f}")
        print(row)
    
    print("-" * 120)
    print()


def save_results_to_csv(results: Dict, output_file: str):
    """保存结果到CSV文件"""
    if not PANDAS_AVAILABLE:
        print("Warning: pandas not available, skipping CSV export")
        return
    
    rows = []
    for method_name, data in results.items():
        rows.append({
            'Method': method_name,
            'BLEU_Score': data['bleu_score'],
            'Peak_Memory_GB': data['peak_memory_gb'],
            'Avg_Cache_Memory_GB': data['avg_cache_memory_gb'],
            'Peak_Cache_Memory_GB': data['peak_cache_memory_gb'],
            'Avg_Length': data['avg_length'],
            'Avg_Inference_Time': data['avg_inference_time'],
            'Avg_AL': data['avg_AL'],
            'Avg_LAAL': data['avg_LAAL'],
            'Performance_Drop_Percent': data.get('performance_drop', 0.0),
            'Memory_Reduction_Percent': data.get('memory_reduction', 0.0),
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Results saved to CSV: {output_file}")


def save_results_to_json(results: Dict, output_file: str):
    """保存结果到JSON文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to JSON: {output_file}")


def generate_latex_table(results: Dict, output_file: str):
    """生成LaTeX表格（用于论文）"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Comparison of different KV cache compression methods}\n")
        f.write("\\label{tab:comparison}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write("Method & BLEU & Memory (GB) & Cache (GB) & Perf. Drop (\\%) & Mem. Red. (\\%) \\\\\n")
        f.write("\\midrule\n")
        
        for method_name, data in sorted(results.items()):
            method_latex = method_name.replace('&', '\\&')
            f.write(f"{method_latex} & "
                   f"{data['bleu_score']:.2f} & "
                   f"{data['peak_memory_gb']:.2f} & "
                   f"{data['avg_cache_memory_gb']:.4f} & "
                   f"{data.get('performance_drop', 0.0):.2f} & "
                   f"{data.get('memory_reduction', 0.0):.2f} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table*}\n")
    
    print(f"LaTeX table saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Base directory containing experiment results")
    parser.add_argument("--save_csv", type=str, default=None,
                       help="Save results to CSV file")
    parser.add_argument("--save_json", type=str, default=None,
                       help="Save results to JSON file")
    parser.add_argument("--save_latex", type=str, default=None,
                       help="Save LaTeX table to file")
    parser.add_argument("--detailed", action="store_true",
                       help="Show detailed table with all metrics")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        print(f"Error: Output directory not found: {args.output_dir}")
        sys.exit(1)
    
    # 分析实验结果
    print(f"Analyzing experiments in: {args.output_dir}")
    results = analyze_experiments(args.output_dir)
    
    if not results:
        print("No results found!")
        sys.exit(1)
    
    # 打印汇总
    print_summary_table(results)
    
    if args.detailed:
        print_detailed_table(results)
    
    # 保存结果
    if args.save_csv:
        save_results_to_csv(results, args.save_csv)
    
    if args.save_json:
        save_results_to_json(results, args.save_json)
    
    if args.save_latex:
        generate_latex_table(results, args.save_latex)
    
    print("\nAnalysis completed!")


if __name__ == "__main__":
    main()

