"""
Head-Aware Heterogeneous Quantization Utilities
实现不同精度的量化/反量化函数，用于HAQ-KV
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional
import math


class Quantizer:
    """
    量化器基类
    支持INT4, INT2等低精度量化
    """
    
    def __init__(self, bits: int = 4):
        """
        Args:
            bits: 量化位数 (2, 4, 8等)
        """
        self.bits = bits
        self.max_val = 2 ** (bits - 1) - 1
        self.min_val = -(2 ** (bits - 1))
        
    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        量化张量
        
        Args:
            x: 输入张量 [..., dim]
        
        Returns:
            (quantized_tensor, scale): 量化后的张量和缩放因子（标量）
        """
        # 计算scale: 将输入范围映射到量化范围
        x_max = x.abs().max().item()  # 转换为Python标量
        if x_max < 1e-8:
            # 全零或接近全零，直接返回
            scale = torch.tensor(1.0, device=x.device, dtype=torch.float32)
            quantized = torch.zeros_like(x, dtype=torch.int8)
            return quantized, scale
        
        scale = torch.tensor(x_max / self.max_val, device=x.device, dtype=torch.float32)
        
        # 量化: 先除以scale，再四舍五入到整数
        quantized = torch.round(x / scale).clamp(self.min_val, self.max_val)
        quantized = quantized.to(torch.int8)  # 使用int8存储（即使bits=4）
        
        return quantized, scale
    
    def dequantize(self, quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        反量化张量
        
        Args:
            quantized: 量化后的张量
            scale: 缩放因子
        
        Returns:
            反量化后的张量
        """
        return quantized.float() * scale


class HeadAwareQuantizer:
    """
    Head-Aware异构量化器
    
    核心策略:
    - Retrieval Heads: 低精度量化 (INT4/INT2) - 语义记忆
    - Induction/Local Heads: 高精度 (FP16/BF16) - 语法缓冲
    """
    
    def __init__(
        self,
        retrieval_bits: int = 4,  # Retrieval Heads使用INT4
        induction_bits: int = 16,  # Induction Heads使用FP16
        local_bits: int = 16,      # Local Heads使用FP16
        device: str = 'cuda'
    ):
        """
        Args:
            retrieval_bits: Retrieval Heads的量化位数 (2, 4, 8)
            induction_bits: Induction Heads的量化位数 (16表示FP16)
            local_bits: Local Heads的量化位数 (16表示FP16)
            device: 设备
        """
        self.retrieval_bits = retrieval_bits
        self.induction_bits = induction_bits
        self.local_bits = local_bits
        self.device = device
        
        # 创建量化器
        if retrieval_bits < 16:
            self.retrieval_quantizer = Quantizer(bits=retrieval_bits)
        else:
            self.retrieval_quantizer = None  # 不量化
        
        if induction_bits < 16:
            self.induction_quantizer = Quantizer(bits=induction_bits)
        else:
            self.induction_quantizer = None
        
        if local_bits < 16:
            self.local_quantizer = Quantizer(bits=local_bits)
        else:
            self.local_quantizer = None
    
    def quantize_by_head_type(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        head_types: dict,  # {head_idx: 'retrieval'|'induction'|'local'}
        layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        根据head类型进行异构量化
        
        Args:
            key: [batch_size, num_heads, seq_len, head_dim]
            value: [batch_size, num_heads, seq_len, head_dim]
            head_types: {head_idx: head_type}
            layer_idx: 层索引
        
        Returns:
            (quantized_key, quantized_value, quantization_info)
            quantization_info包含scale信息，用于反量化
        """
        bsz, num_heads, seq_len, head_dim = key.shape
        
        # 存储量化信息: {head_idx: {'key_scale': ..., 'value_scale': ...}}
        quant_info = {}
        
        # 分别处理每个head
        quantized_keys = []
        quantized_values = []
        
        for head_idx in range(num_heads):
            head_type = head_types.get(head_idx, 'retrieval')  # 默认retrieval
            
            # 提取当前head的KV
            head_key = key[:, head_idx:head_idx+1, :, :]  # [bsz, 1, seq_len, head_dim]
            head_value = value[:, head_idx:head_idx+1, :, :]
            
            # 根据head类型选择量化策略
            if head_type == 'retrieval':
                # Retrieval Heads: 低精度量化
                if self.retrieval_quantizer is not None:
                    q_key, key_scale = self.retrieval_quantizer.quantize(head_key)
                    q_value, value_scale = self.retrieval_quantizer.quantize(head_value)
                    quant_info[head_idx] = {
                        'key_scale': key_scale,
                        'value_scale': value_scale,
                        'bits': self.retrieval_bits
                    }
                else:
                    # 不量化
                    q_key, q_value = head_key, head_value
                    quant_info[head_idx] = {'bits': 16}
            
            elif head_type == 'induction':
                # Induction Heads: 高精度
                if self.induction_quantizer is not None:
                    q_key, key_scale = self.induction_quantizer.quantize(head_key)
                    q_value, value_scale = self.induction_quantizer.quantize(head_value)
                    quant_info[head_idx] = {
                        'key_scale': key_scale,
                        'value_scale': value_scale,
                        'bits': self.induction_bits
                    }
                else:
                    q_key, q_value = head_key, head_value
                    quant_info[head_idx] = {'bits': 16}
            
            else:  # local
                # Local Heads: 高精度
                if self.local_quantizer is not None:
                    q_key, key_scale = self.local_quantizer.quantize(head_key)
                    q_value, value_scale = self.local_quantizer.quantize(head_value)
                    quant_info[head_idx] = {
                        'key_scale': key_scale,
                        'value_scale': value_scale,
                        'bits': self.local_bits
                    }
                else:
                    q_key, q_value = head_key, head_value
                    quant_info[head_idx] = {'bits': 16}
            
            quantized_keys.append(q_key)
            quantized_values.append(q_value)
        
        # 拼接所有heads
        quantized_key = torch.cat(quantized_keys, dim=1)  # [bsz, num_heads, seq_len, head_dim]
        quantized_value = torch.cat(quantized_values, dim=1)
        
        return quantized_key, quantized_value, quant_info
    
    def dequantize_by_head_type(
        self,
        quantized_key: torch.Tensor,
        quantized_value: torch.Tensor,
        quant_info: dict,
        head_types: dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据量化信息反量化
        
        Args:
            quantized_key: 量化后的key
            quantized_value: 量化后的value
            quant_info: 量化信息
            head_types: head类型
        
        Returns:
            (dequantized_key, dequantized_value)
        """
        bsz, num_heads, seq_len, head_dim = quantized_key.shape
        
        dequantized_keys = []
        dequantized_values = []
        
        for head_idx in range(num_heads):
            head_type = head_types.get(head_idx, 'retrieval')
            info = quant_info.get(head_idx, {'bits': 16})
            
            # 提取当前head
            head_key = quantized_key[:, head_idx:head_idx+1, :, :]
            head_value = quantized_value[:, head_idx:head_idx+1, :, :]
            
            # 如果bits=16，说明没有量化，直接返回
            if info.get('bits', 16) == 16:
                dequantized_keys.append(head_key)
                dequantized_values.append(head_value)
            else:
                # 反量化
                key_scale = info.get('key_scale', torch.tensor(1.0, device=head_key.device))
                value_scale = info.get('value_scale', torch.tensor(1.0, device=head_value.device))
                
                # 选择对应的量化器
                if head_type == 'retrieval' and self.retrieval_quantizer:
                    dq_key = self.retrieval_quantizer.dequantize(head_key, key_scale)
                    dq_value = self.retrieval_quantizer.dequantize(head_value, value_scale)
                elif head_type == 'induction' and self.induction_quantizer:
                    dq_key = self.induction_quantizer.dequantize(head_key, key_scale)
                    dq_value = self.induction_quantizer.dequantize(head_value, value_scale)
                elif head_type == 'local' and self.local_quantizer:
                    dq_key = self.local_quantizer.dequantize(head_key, key_scale)
                    dq_value = self.local_quantizer.dequantize(head_value, value_scale)
                else:
                    # 没有量化器，直接返回
                    dq_key, dq_value = head_key, head_value
                
                dequantized_keys.append(dq_key)
                dequantized_values.append(dq_value)
        
        dequantized_key = torch.cat(dequantized_keys, dim=1)
        dequantized_value = torch.cat(dequantized_values, dim=1)
        
        return dequantized_key, dequantized_value
    
    def estimate_memory_saving(
        self,
        num_heads: int,
        head_types: dict,
        seq_len: int,
        head_dim: int
    ) -> dict:
        """
        估算内存节省
        
        Returns:
            {
                'original_memory_gb': ...,
                'quantized_memory_gb': ...,
                'saving_ratio': ...,
                'effective_length_multiplier': ...
            }
        """
        # 原始内存 (FP16, 2 bytes per element)
        original_memory = num_heads * seq_len * head_dim * 2 * 2  # key + value
        
        # 计算量化后的内存
        quantized_memory = 0
        retrieval_count = 0
        
        for head_idx in range(num_heads):
            head_type = head_types.get(head_idx, 'retrieval')
            
            if head_type == 'retrieval':
                bits = self.retrieval_bits
                retrieval_count += 1
            elif head_type == 'induction':
                bits = self.induction_bits
            else:
                bits = self.local_bits
            
            # 量化后的字节数 (加上scale的存储)
            if bits < 16:
                bytes_per_element = bits / 8
                # scale存储 (FP32, 4 bytes)
                scale_overhead = 4 * 2  # key_scale + value_scale
            else:
                bytes_per_element = 2  # FP16
                scale_overhead = 0
            
            head_memory = seq_len * head_dim * bytes_per_element * 2 + scale_overhead
            quantized_memory += head_memory
        
        saving_ratio = 1 - (quantized_memory / original_memory)
        
        # 有效长度倍数: 在相同内存下可以存储的倍数
        if quantized_memory > 0:
            effective_multiplier = original_memory / quantized_memory
        else:
            effective_multiplier = 1.0
        
        return {
            'original_memory_gb': original_memory / (1024 ** 3),
            'quantized_memory_gb': quantized_memory / (1024 ** 3),
            'saving_ratio': saving_ratio,
            'effective_length_multiplier': effective_multiplier,
            'retrieval_heads_count': retrieval_count
        }

