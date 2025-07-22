#!/usr/bin/env python3
"""
简单的模型测试脚本
"""

import torch
import sys
import os

# 添加项目路径
sys.path.append('/home/jebediahc/tum/soil-moisture-prediction-dl')

def test_model_forward():
    """测试模型前向传播"""
    from src.models import CNN3D_LSTM_Lite
    from src.utils import load_config
    
    print("Testing CNN3D_LSTM_Lite model...")
    
    # 配置
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建模型
    input_dim = len(config["use_channels"])
    model = CNN3D_LSTM_Lite(input_dim=input_dim).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建测试输入 (使用较小的尺寸)
    batch_size = 1
    input_days = 10  # 减少输入天数
    h, w = 32, 64    # 使用较小的空间分辨率
    
    x = torch.randn(batch_size, input_days, input_dim, h, w, device=device)
    print(f"Input shape: {x.shape}")
    
    try:
        # 测试前向传播
        model.eval()
        with torch.no_grad():
            output = model(x)
            print(f"Output shape: {output.shape}")
            print("✅ Forward pass successful!")
            
        # 测试训练模式
        model.train()
        x.requires_grad_(True)
        output = model(x)
        loss = output.mean()
        loss.backward()
        print("✅ Backward pass successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_forward()
    if success:
        print("\n🎉 Test passed! The model should work in training now.")
    else:
        print("\n⚠️ Test failed. Please check the errors above.")
