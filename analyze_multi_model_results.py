"""
多模型实验结果分析脚本
生成对比表格和可视化图表
用于A级会议/期刊论文
"""
import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_results(base_dir):
    """加载所有实验结果"""
    results = {}
    
    # 查找所有results.json文件
    pattern = os.path.join(base_dir, "**/results.json")
    result_files = glob.glob(pattern, recursive=True)
    
    for result_file in result_files:
        # 从路径提取模型和方法
        path_parts = Path(result_file).parts
        # 假设路径格式: base_dir/model_method/results.json
        if len(path_parts) >= 2:
            model_method = path_parts[-2]  # 倒数第二个是目录名
            parts = model_method.split('_')
            if len(parts) >= 2:
                model = parts[0]
                method = '_'.join(parts[1:])
                
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    key = (model, method)
                    results[key] = data
    
    return results


def create_comparison_table(results):
    """创建对比表格"""
    rows = []
    
    for (model, method), data in results.items():
        row = {
            'Model': model,
            'Method': method,
            'BLEU': data.get('bleu_score', 0),
            'Peak Memory (GB)': data.get('memory_stats', {}).get('peak_memory_gb', 0),
            'Cache Memory (GB)': data.get('cache_stats', {}).get('avg_cache_memory_gb', 0),
            'Max Length': data.get('length_stats', {}).get('max_length', 0),
            'Avg Length': data.get('length_stats', {}).get('avg_length', 0),
            'AL': data.get('streaming_stats', {}).get('avg_AL', 0),
            'LAAL': data.get('streaming_stats', {}).get('avg_LAAL', 0),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def generate_tables(df, output_dir):
    """生成对比表格"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 表格1: 按模型分组
    table1 = df.pivot_table(
        index='Model',
        columns='Method',
        values=['BLEU', 'Peak Memory (GB)', 'Cache Memory (GB)', 'Max Length'],
        aggfunc='mean'
    )
    
    table1_file = os.path.join(output_dir, 'table_by_model.tex')
    table1.to_latex(table1_file, float_format="%.2f")
    print(f"Saved table to {table1_file}")
    
    # 表格2: 按方法分组
    table2 = df.pivot_table(
        index='Method',
        columns='Model',
        values=['BLEU', 'Peak Memory (GB)', 'Max Length'],
        aggfunc='mean'
    )
    
    table2_file = os.path.join(output_dir, 'table_by_method.tex')
    table2.to_latex(table2_file, float_format="%.2f")
    print(f"Saved table to {table2_file}")
    
    # 保存CSV
    csv_file = os.path.join(output_dir, 'all_results.csv')
    df.to_csv(csv_file, index=False)
    print(f"Saved CSV to {csv_file}")


def generate_plots(df, output_dir):
    """生成可视化图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置样式
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    # 图1: 内存使用对比（按模型）
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1.1 峰值内存对比
    ax1 = axes[0, 0]
    memory_data = df.pivot_table(index='Model', columns='Method', values='Peak Memory (GB)', aggfunc='mean')
    memory_data.plot(kind='bar', ax=ax1, rot=45)
    ax1.set_title('Peak Memory Usage by Model and Method')
    ax1.set_ylabel('Memory (GB)')
    ax1.legend(title='Method')
    
    # 1.2 BLEU分数对比
    ax2 = axes[0, 1]
    bleu_data = df.pivot_table(index='Model', columns='Method', values='BLEU', aggfunc='mean')
    bleu_data.plot(kind='bar', ax=ax2, rot=45)
    ax2.set_title('BLEU Score by Model and Method')
    ax2.set_ylabel('BLEU Score')
    ax2.legend(title='Method')
    
    # 1.3 最大长度对比
    ax3 = axes[1, 0]
    length_data = df.pivot_table(index='Model', columns='Method', values='Max Length', aggfunc='mean')
    length_data.plot(kind='bar', ax=ax3, rot=45)
    ax3.set_title('Max Supported Length by Model and Method')
    ax3.set_ylabel('Length (tokens)')
    ax3.legend(title='Method')
    
    # 1.4 AL/LAAL对比
    ax4 = axes[1, 1]
    al_data = df[df['Method'] != 'baseline'].groupby(['Model', 'Method'])[['AL', 'LAAL']].mean().reset_index()
    al_data_melted = al_data.melt(id_vars=['Model', 'Method'], value_vars=['AL', 'LAAL'], 
                                   var_name='Metric', value_name='Value')
    sns.barplot(data=al_data_melted, x='Model', y='Value', hue='Method', ax=ax4)
    ax4.set_title('AL/LAAL by Model and Method')
    ax4.set_ylabel('Delay')
    ax4.legend(title='Method')
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'multi_model_comparison.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot_file}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='./output_logs/multi_model',
                        help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='./output_logs/analysis',
                        help='Output directory for analysis results')
    args = parser.parse_args()
    
    print("Loading results...")
    results = load_results(args.results_dir)
    print(f"Loaded {len(results)} results")
    
    print("Creating comparison table...")
    df = create_comparison_table(results)
    print(df)
    
    print("Generating tables...")
    generate_tables(df, args.output_dir)
    
    print("Generating plots...")
    generate_plots(df, args.output_dir)
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()

