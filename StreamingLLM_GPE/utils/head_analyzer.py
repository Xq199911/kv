"""
Head-Aware KV Cache Budgeting: Head功能分析模块
分析每个attention head的功能特性，用于动态分配KV cache预算
"""
from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np
import torch


class HeadAnalyzer:
    """
    分析每个attention head的功能特性
    根据attention patterns将heads分为三类：
    1. Retrieval Heads: 需要长距离依赖，保留更多历史KV
    2. Induction Heads: 需要中等距离的模式匹配
    3. Local Heads: 只关注邻近tokens，不需要长距离KV
    """
    
    def __init__(self, num_layers: int, num_heads: int, device: str = 'cuda'):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.device = device
        
        # 存储每个head的分类结果
        # {layer_idx: {head_idx: head_type}}
        self.head_profiles: Dict[int, Dict[int, str]] = defaultdict(dict)
        
        # 存储attention统计信息（用于分析）
        self.attention_stats: Dict[int, Dict[int, dict]] = defaultdict(lambda: defaultdict(dict))
        
        # 是否已经分析完成
        self.analyzed = False
        
    def analyze_head_functionality(
        self, 
        attention_weights: torch.Tensor, 
        layer_idx: int,
        cache_length: Optional[int] = None
    ) -> Dict[int, str]:
        """
        通过分析attention patterns判断head类型
        
        Args:
            attention_weights: [batch_size, num_heads, seq_len, kv_seq_len]
            layer_idx: 当前层索引
            cache_length: KV cache的长度（用于计算局部性）
        
        Returns:
            {head_idx: head_type}
        """
        batch_size, num_heads, seq_len, kv_seq_len = attention_weights.shape
        
        # 平均所有batch和sequence位置
        # [num_heads, kv_seq_len]
        avg_attention = attention_weights.mean(dim=(0, 2))
        
        head_types = {}
        
        for head_idx in range(num_heads):
            head_attn = avg_attention[head_idx].cpu().numpy()  # [kv_seq_len]
            
            # 计算统计特征
            stats = self._compute_attention_stats(head_attn, cache_length)
            self.attention_stats[layer_idx][head_idx] = stats
            
            # 根据特征分类
            head_type = self._classify_head(stats)
            head_types[head_idx] = head_type
            self.head_profiles[layer_idx][head_idx] = head_type
        
        return head_types
    
    def _compute_attention_stats(
        self, 
        attention: np.ndarray, 
        cache_length: Optional[int]
    ) -> dict:
        """
        计算attention pattern的统计特征
        """
        stats = {}
        
        # 1. Entropy（熵）：衡量attention分布的均匀性
        # 高熵 = 分布均匀 = 可能是retrieval head
        attention_sum = attention.sum()
        if attention_sum > 1e-10:
            attention_normalized = attention / attention_sum
            # 只对非零值计算熵，避免log(0)问题
            non_zero_mask = attention_normalized > 1e-10
            if np.any(non_zero_mask):
                entropy = -np.sum(attention_normalized[non_zero_mask] * np.log(attention_normalized[non_zero_mask]))
                # 处理可能的NaN或inf值
                entropy = np.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                entropy = 0.0
        else:
            # 如果attention全为0，熵为0
            attention_normalized = np.zeros_like(attention)
            entropy = 0.0
        stats['entropy'] = entropy
        
        # 2. Locality（局部性）：衡量attention是否集中在局部
        # 计算attention权重到当前位置的平均距离
        positions = np.arange(len(attention))
        weighted_distance = np.sum(attention_normalized * positions)
        stats['weighted_distance'] = weighted_distance
        stats['locality'] = 1.0 / (weighted_distance + 1.0)  # 局部性分数
        
        # 3. Concentration（集中度）：衡量attention的集中程度
        # 使用Gini系数
        sorted_attn = np.sort(attention_normalized)
        n = len(sorted_attn)
        sorted_sum = sorted_attn.sum()
        if sorted_sum > 1e-10:
            gini = (2 * np.sum((np.arange(1, n+1)) * sorted_attn)) / (n * sorted_sum) - (n+1)/n
            gini = np.nan_to_num(gini, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            gini = 0.0
        stats['concentration'] = abs(gini)
        
        # 4. Long-range ratio（长距离比例）
        if cache_length is not None:
            # 计算对前50%位置的attention权重
            mid_point = len(attention) // 2
            long_range_ratio = attention[:mid_point].sum() / (attention.sum() + 1e-10)
            stats['long_range_ratio'] = long_range_ratio
        else:
            stats['long_range_ratio'] = 0.5
        
        # 5. Peak position（峰值位置）
        peak_idx = np.argmax(attention)
        stats['peak_position'] = peak_idx / (len(attention) + 1e-10)  # 归一化
        
        return stats
    
    def _classify_head(self, stats: dict) -> str:
        """
        根据统计特征分类head类型
        
        分类规则：
        - Retrieval: 高熵 + 高长距离比例 + 低局部性
        - Induction: 中等熵 + 中等长距离比例 + 有模式
        - Local: 低熵 + 高局部性 + 高集中度
        """
        entropy = stats['entropy']
        locality = stats['locality']
        long_range_ratio = stats['long_range_ratio']
        concentration = stats['concentration']
        
        # 阈值（可以根据实验调整）
        entropy_threshold = 3.0  # 经验值
        locality_threshold = 0.3
        long_range_threshold = 0.4
        
        # 分类逻辑
        if entropy > entropy_threshold and long_range_ratio > long_range_threshold:
            # 高熵 + 长距离 = Retrieval Head
            return 'retrieval'
        elif locality > locality_threshold and concentration > 0.5:
            # 高局部性 + 高集中度 = Local Head
            return 'local'
        else:
            # 其他情况 = Induction Head
            return 'induction'
    
    def get_head_budget(
        self, 
        layer_idx: int, 
        head_idx: int, 
        total_budget: int,
        adaptive: bool = True
    ) -> int:
        """
        根据head类型返回预算分配
        
        Args:
            layer_idx: 层索引
            head_idx: head索引
            total_budget: 总预算（tokens）
            adaptive: 是否使用自适应分配
        
        Returns:
            该head分配的预算（tokens）
        """
        if not self.analyzed:
            # 如果还没分析，使用均匀分配
            return total_budget // self.num_heads
        
        head_type = self.head_profiles.get(layer_idx, {}).get(head_idx, 'induction')
        
        if adaptive:
            # 自适应分配策略
            if head_type == 'retrieval':
                # Retrieval heads需要更多预算（50%）
                return int(total_budget * 0.5 / self.num_heads * 2)  # 2倍权重
            elif head_type == 'induction':
                # Induction heads中等预算（30%）
                return int(total_budget * 0.3 / self.num_heads * 1.2)  # 1.2倍权重
            else:  # local
                # Local heads需要较少预算（20%）
                return int(total_budget * 0.2 / self.num_heads * 0.8)  # 0.8倍权重
        else:
            # 均匀分配
            return total_budget // self.num_heads
    
    def get_all_head_budgets(
        self, 
        layer_idx: int, 
        total_budget: int,
        adaptive: bool = True
    ) -> Dict[int, int]:
        """
        获取某一层所有heads的预算分配
        
        Returns:
            {head_idx: budget}
        """
        budgets = {}
        for head_idx in range(self.num_heads):
            budgets[head_idx] = self.get_head_budget(
                layer_idx, head_idx, total_budget, adaptive
            )
        
        # 归一化，确保总和等于total_budget
        total_allocated = sum(budgets.values())
        if total_allocated > 0:
            scale = total_budget / total_allocated
            budgets = {k: int(v * scale) for k, v in budgets.items()}
        
        return budgets
    
    def analyze_model(
        self, 
        model, 
        sample_inputs: List[torch.Tensor],
        num_samples: int = 10
    ):
        """
        在样本数据上分析整个模型的head特性
        
        Args:
            model: 要分析的模型
            sample_inputs: 样本输入列表
            num_samples: 分析的样本数量
        """
        model.eval()
        
        with torch.no_grad():
            for i, inputs in enumerate(sample_inputs[:num_samples]):
                # 前向传播，收集attention weights
                outputs = model(**inputs, output_attentions=True)
                
                if hasattr(outputs, 'attentions') and outputs.attentions:
                    for layer_idx, attn_weights in enumerate(outputs.attentions):
                        # attn_weights: [batch, num_heads, seq_len, seq_len]
                        cache_length = attn_weights.shape[-1]
                        self.analyze_head_functionality(
                            attn_weights, layer_idx, cache_length
                        )
        
        self.analyzed = True
        
        # 打印统计信息
        self._print_statistics()
    
    def _print_statistics(self):
        """打印head分类统计信息"""
        print("\n=== Head Analysis Statistics ===")
        
        for layer_idx in range(self.num_layers):
            layer_profiles = self.head_profiles.get(layer_idx, {})
            if not layer_profiles:
                continue
            
            retrieval_count = sum(1 for t in layer_profiles.values() if t == 'retrieval')
            induction_count = sum(1 for t in layer_profiles.values() if t == 'induction')
            local_count = sum(1 for t in layer_profiles.values() if t == 'local')
            
            print(f"\nLayer {layer_idx}:")
            print(f"  Retrieval Heads: {retrieval_count}/{self.num_heads}")
            print(f"  Induction Heads: {induction_count}/{self.num_heads}")
            print(f"  Local Heads: {local_count}/{self.num_heads}")
        
        print("=" * 30)

