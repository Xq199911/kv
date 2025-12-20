#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验结果可视化脚本
生成论文所需的图表
"""
import argparse
import json
import os

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not installed. Install with: pip install matplotlib")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not installed. Install with: pip install pandas")


def load_results(result_dir: str) -> dict:
    """加载实验结果"""
    result_file = os.path.join(result_dir, "results.json")
    if not os.path.exists(result_file):
        return None
    
    with open(result_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_bleu_comparison(results_dict: dict, output_file: str):
    """绘制BLEU分数对比柱状图"""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping plot")
        return
    
    methods = []
    bleu_scores = []
    
    for method_name, data in sorted(results_dict.items()):
        methods.append(method_name)
        bleu_scores.append(data['bleu_score'])
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(methods)), bleu_scores, color='steelblue', alpha=0.7)
    
    # 添加数值标签
    for i, (bar, score) in enumerate(zip(bars, bleu_scores)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{score:.2f}', ha='center', va='bottom')
    
    plt.xlabel('Method', fontsize=12)
    plt.ylabel('BLEU Score', fontsize=12)
    plt.title('BLEU Score Comparison', fontsize=14, fontweight='bold')
    plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"BLEU comparison plot saved to: {output_file}")


def plot_memory_comparison(results_dict: dict, output_file: str):
    """绘制内存使用对比图"""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping plot")
        return
    
    methods = []
    memory_usage = []
    cache_memory = []
    
    for method_name, data in sorted(results_dict.items()):
        methods.append(method_name)
        memory_usage.append(data['peak_memory_gb'])
        cache_memory.append(data['avg_cache_memory_gb'])
    
    x = np.arange(len(methods))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, memory_usage, width, label='Peak GPU Memory', alpha=0.7)
    bars2 = ax.bar(x + width/2, cache_memory, width, label='Cache Memory', alpha=0.7)
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Memory (GB)', fontsize=12)
    ax.set_title('Memory Usage Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Memory comparison plot saved to: {output_file}")


def plot_tradeoff(results_dict: dict, output_file: str):
    """绘制性能-内存权衡图"""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping plot")
        return
    
    methods = []
    bleu_scores = []
    memory_usage = []
    
    for method_name, data in sorted(results_dict.items()):
        methods.append(method_name)
        bleu_scores.append(data['bleu_score'])
        memory_usage.append(data['peak_memory_gb'])
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(memory_usage, bleu_scores, s=200, alpha=0.6, c=range(len(methods)), cmap='viridis')
    
    # 添加方法标签
    for i, method in enumerate(methods):
        plt.annotate(method, (memory_usage[i], bleu_scores[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.xlabel('Peak Memory (GB)', fontsize=12)
    plt.ylabel('BLEU Score', fontsize=12)
    plt.title('Performance-Memory Tradeoff', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.colorbar(scatter, label='Method Index')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Tradeoff plot saved to: {output_file}")


def plot_budget_impact(budget_results: dict, output_file: str):
    """绘制预算对性能的影响"""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping plot")
        return
    
    budgets = sorted([int(k.replace('budget_', '')) for k in budget_results.keys() if 'budget' in k.lower()])
    bleu_scores = []
    memory_usage = []
    
    for budget in budgets:
        key = f"Head-Aware (Budget={budget})"
        if key in budget_results:
            bleu_scores.append(budget_results[key]['bleu_score'])
            memory_usage.append(budget_results[key]['peak_memory_gb'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # BLEU vs Budget
    ax1.plot(budgets, bleu_scores, marker='o', linewidth=2, markersize=8)
    ax1.set_xlabel('Budget (tokens/layer)', fontsize=12)
    ax1.set_ylabel('BLEU Score', fontsize=12)
    ax1.set_title('BLEU Score vs Budget', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3)
    
    # Memory vs Budget
    ax2.plot(budgets, memory_usage, marker='s', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Budget (tokens/layer)', fontsize=12)
    ax2.set_ylabel('Peak Memory (GB)', fontsize=12)
    ax2.set_title('Memory Usage vs Budget', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Budget impact plot saved to: {output_file}")


def main():
    import sys
    parser = argparse.ArgumentParser(description="Visualize experiment results")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing experiment results")
    parser.add_argument("--output_dir", type=str, default="./output_logs/figures",
                       help="Output directory for figures")
    parser.add_argument("--include_budget", action="store_true",
                       help="Include budget impact analysis")
    
    args = parser.parse_args()
    
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is required for visualization")
        print("Install with: pip install matplotlib")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载结果
    from analyze_experiment_results import analyze_experiments
    results = analyze_experiments(args.results_dir)
    
    if not results:
        print("No results found!")
        return
    
    # 生成图表
    print("Generating visualizations...")
    
    plot_bleu_comparison(results, os.path.join(args.output_dir, "bleu_comparison.png"))
    plot_memory_comparison(results, os.path.join(args.output_dir, "memory_comparison.png"))
    plot_tradeoff(results, os.path.join(args.output_dir, "tradeoff.png"))
    
    if args.include_budget:
        # 尝试加载预算实验结果
        budget_dir = args.results_dir.replace("experiments", "budget_experiments")
        if os.path.exists(budget_dir):
            budget_results = analyze_experiments(budget_dir)
            if budget_results:
                plot_budget_impact(budget_results, os.path.join(args.output_dir, "budget_impact.png"))
    
    print(f"\nAll figures saved to: {args.output_dir}")


if __name__ == "__main__":
    import sys
    main()

