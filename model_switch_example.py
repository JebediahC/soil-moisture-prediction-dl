#!/usr/bin/env python3
"""
模型切换示例脚本
演示如何在ConvLSTM和3DCNN-LSTM之间切换
"""

import yaml
from src.models import get_model
from src.utils import load_config

def switch_model_in_config(model_type):
    """
    在配置文件中切换模型类型
    
    Args:
        model_type: 要切换到的模型类型 ('convlstm' 或 '3dcnn_lstm')
    """
    config_path = "config/config.yaml"
    
    # 读取现有配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 更新模型类型
    config['model_type'] = model_type
    
    # 保存配置
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Model type switched to: {model_type}")

def demo_model_creation():
    """
    演示创建不同类型的模型
    """
    config = load_config()
    input_dim = len(config["use_channels"])
    
    print("=== Model Creation Demo ===")
    
    models_to_demo = [
        ('convlstm', 'ConvLSTM'),
        ('3dcnn_lstm', '3DCNN-LSTM (Original)'),
        ('3dcnn_lstm_lite', '3DCNN-LSTM (Lite)'),
        ('3dcnn_lstm_micro', '3DCNN-LSTM (Micro)')
    ]
    
    for model_type, model_name in models_to_demo:
        print(f"\n{len([m for m in models_to_demo if models_to_demo.index(m) <= models_to_demo.index((model_type, model_name))])}. Creating {model_name}:")
        try:
            model = get_model(model_type, input_dim)
            params = sum(p.numel() for p in model.parameters())
            print(f"   Parameters: {params:,d} ({params/1e6:.2f}M)")
            print(f"   Model structure preview:")
            print(f"   {str(model)[:200]}..." if len(str(model)) > 200 else f"   {model}")
        except Exception as e:
            print(f"   Error: {e}")
    
    print("\n=== Memory Usage Recommendations ===")
    print("根据GPU显存选择合适的模型:")
    print("- 2-4GB GPU: 使用 convlstm 或 3dcnn_lstm_micro")
    print("- 4-8GB GPU: 使用 3dcnn_lstm_lite") 
    print("- 8GB+ GPU: 可以尝试 3dcnn_lstm")
    print("\n=== 性能预期 ===")
    print("- convlstm: 最快训练，基础效果")
    print("- 3dcnn_lstm_micro: 快速训练，改进的时空特征提取")
    print("- 3dcnn_lstm_lite: 平衡的训练速度和效果") 
    print("- 3dcnn_lstm: 最佳效果但训练较慢")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Model switching utility")
    parser.add_argument("--switch", type=str, 
                       choices=['convlstm', '3dcnn_lstm', '3dcnn_lstm_lite', '3dcnn_lstm_micro'], 
                       help="Switch to specified model type")
    parser.add_argument("--demo", action='store_true', help="Run model creation demo")
    
    args = parser.parse_args()
    
    if args.switch:
        switch_model_in_config(args.switch)
    
    if args.demo:
        demo_model_creation()
    
    if not args.switch and not args.demo:
        print("Usage:")
        print("  python model_switch_example.py --switch convlstm         # Switch to ConvLSTM")
        print("  python model_switch_example.py --switch 3dcnn_lstm       # Switch to 3DCNN-LSTM") 
        print("  python model_switch_example.py --switch 3dcnn_lstm_lite  # Switch to 3DCNN-LSTM Lite")
        print("  python model_switch_example.py --switch 3dcnn_lstm_micro # Switch to 3DCNN-LSTM Micro")
        print("  python model_switch_example.py --demo                    # Show model demo")
