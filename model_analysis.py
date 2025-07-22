#!/usr/bin/env python3
"""
模型分析工具 - 分析模型参数量、显存占用和计算复杂度
"""

import torch
import torch.nn as nn
from src.models import ConvLSTM, CNN3D_LSTM, CNN3D_LSTM_Lite, CNN3D_LSTM_Micro, get_model
from src.utils import load_config
import numpy as np

def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def estimate_memory_usage(model, input_shape, batch_size=1):
    """估算模型显存占用"""
    # 创建输入张量
    x = torch.randn(batch_size, *input_shape)
    
    # 计算模型参数占用的显存（假设float32，每个参数4字节）
    param_memory = sum(p.numel() for p in model.parameters()) * 4 / (1024**2)  # MB
    
    # 计算输入数据占用的显存
    input_memory = x.numel() * 4 / (1024**2)  # MB
    
    # 估算激活值占用的显存（粗略估计为输入的5-10倍）
    activation_memory = input_memory * 8  # 经验值
    
    # 估算梯度占用的显存（与参数量相同）
    gradient_memory = param_memory
    
    total_memory = param_memory + input_memory + activation_memory + gradient_memory
    
    return {
        'param_memory_mb': param_memory,
        'input_memory_mb': input_memory, 
        'activation_memory_mb': activation_memory,
        'gradient_memory_mb': gradient_memory,
        'total_memory_mb': total_memory,
        'total_memory_gb': total_memory / 1024
    }

def analyze_model_layers(model):
    """分析模型各层的参数量"""
    print("=== Layer-wise Parameter Analysis ===")
    total_params = 0
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 叶子节点
            layer_params = sum(p.numel() for p in module.parameters())
            if layer_params > 0:
                print(f"{name:30s}: {layer_params:>10,d} parameters")
                total_params += layer_params
    
    print(f"{'Total':30s}: {total_params:>10,d} parameters")
    return total_params

def compare_models():
    """比较不同模型的规模"""
    config = load_config()
    input_dim = len(config["use_channels"])
    input_days = config["input_days"]
    
    # 假设空间分辨率
    h, w = 181, 360  # 常见的全球气象数据分辨率
    input_shape = (input_days, input_dim, h, w)
    
    print("=== Model Comparison ===")
    print(f"Input shape: {input_shape}")
    print(f"Batch size for analysis: 4")
    print()
    
    models_to_compare = [
        ('ConvLSTM', 'convlstm', {}),
        ('3DCNN-LSTM (Original)', '3dcnn_lstm', {}),
        ('3DCNN-LSTM (Lite)', '3dcnn_lstm_lite', {}),
        ('3DCNN-LSTM (Micro)', '3dcnn_lstm_micro', {}),
        ('3DCNN-LSTM (Custom Small)', '3dcnn_lstm', {'hidden_dim': 32, 'lstm_hidden_dim': 16}),
        ('3DCNN-LSTM (Custom Tiny)', '3dcnn_lstm', {'hidden_dim': 16, 'lstm_hidden_dim': 8}),
    ]
    
    results = []
    
    for i, (name, model_type, kwargs) in enumerate(models_to_compare, 1):
        print(f"{i}. {name}:")
        
        try:
            model = get_model(model_type, input_dim, **kwargs)
            params, trainable = count_parameters(model)
            memory = estimate_memory_usage(model, input_shape, batch_size=4)
            
            print(f"   Parameters: {params:,d} ({params/1e6:.2f}M)")
            print(f"   Estimated GPU memory: {memory['total_memory_gb']:.2f} GB")
            
            if i > 2:  # 比较相对于原始3DCNN-LSTM的减少量
                original_params = results[1][1] if len(results) > 1 else params
                reduction = (original_params - params) / original_params * 100
                print(f"   Parameter reduction: {reduction:.1f}%")
            
            results.append((name, params, memory))
            
            # 只为前3个模型显示详细层分析
            if i <= 3:
                analyze_model_layers(model)
            
            print()
            
        except Exception as e:
            print(f"   Error creating model: {e}")
            print()
    
    # 显存占用对比表
    print("=== Memory Usage Summary ===")
    print(f"{'Model':<25} {'Parameters':<12} {'GPU Memory (GB)':<15} {'Reduction':<12}")
    print("-" * 70)
    
    original_params = results[1][1] if len(results) > 1 else 0
    
    for name, params, memory in results:
        reduction = ""
        if original_params > 0 and params != original_params:
            reduction = f"{(original_params - params) / original_params * 100:.1f}%"
        
        print(f"{name:<25} {params/1e6:>8.2f}M {memory['total_memory_gb']:>12.2f} {reduction:>12}")
    
    return results

def get_optimization_recommendations():
    """提供优化建议"""
    print("\n=== Optimization Recommendations ===")
    
    print("\n1. 模型结构优化:")
    print("   - 减少hidden_dim: 64 → 32 → 16")
    print("   - 减少lstm_hidden_dim: 32 → 16 → 8")
    print("   - 减少3D卷积层数: 3层 → 2层")
    print("   - 使用深度可分离卷积代替标准3D卷积")
    
    print("\n2. 训练优化:")
    print("   - 减少batch_size: 4 → 2 → 1")
    print("   - 使用梯度累积模拟大batch_size")
    print("   - 使用混合精度训练(fp16)")
    print("   - 使用梯度检查点(gradient checkpointing)")
    
    print("\n3. 数据优化:")
    print("   - 降低空间分辨率")
    print("   - 减少输入时间步长")
    print("   - 数据预处理时进行降采样")
    
    print("\n4. 硬件优化:")
    print("   - 使用更大显存的GPU")
    print("   - 使用CPU进行数据预处理")
    print("   - 考虑模型并行或数据并行")

if __name__ == "__main__":
    try:
        compare_models()
        get_optimization_recommendations()
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Note: This analysis uses estimated input shapes and may not reflect actual memory usage.")
