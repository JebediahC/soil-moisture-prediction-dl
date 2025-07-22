# 显存优化完整指南

## 问题分析

根据模型分析结果，3DCNN-LSTM模型的显存占用主要来自以下几个方面：

### 1. 模型参数量对比
- **ConvLSTM**: 11,601 参数 (0.01M) 
- **3DCNN-LSTM (Original)**: 114,833 参数 (0.11M)
- **3DCNN-LSTM (Lite)**: 8,777 参数 (0.01M) - **92.4%减少**
- **3DCNN-LSTM (Micro)**: 2,617 参数 (0.00M) - **97.7%减少**

### 2. 主要显存消耗源
1. **激活值存储** (~636MB): 最大的显存消耗
2. **输入数据** (~79MB): 取决于batch_size和输入分辨率
3. **模型参数** (~0.4MB): 相对较小
4. **梯度存储** (~0.4MB): 与参数量相同

## 优化策略

### 方案1: 使用轻量级模型（推荐）

#### 选择合适的模型变体:

```bash
# 极度内存受限 (2-4GB GPU)
python model_switch_example.py --switch 3dcnn_lstm_micro

# 中等内存 (4-8GB GPU)  
python model_switch_example.py --switch 3dcnn_lstm_lite

# 充足内存 (8GB+ GPU)
python model_switch_example.py --switch 3dcnn_lstm
```

#### 模型特点对比:

| 模型 | 参数量 | 特点 | 适用场景 |
|------|--------|------|----------|
| `3dcnn_lstm_micro` | 2.6K | 单层3D CNN + 小LSTM | 显存极度受限 |
| `3dcnn_lstm_lite` | 8.8K | 双层3D CNN + 中等LSTM | 平衡性能和显存 |
| `3dcnn_lstm` | 114.8K | 三层3D CNN + 大LSTM | 追求最佳性能 |

### 方案2: 训练参数优化

#### 减小batch_size
```yaml
# config/config.yaml
batch_size: 1  # 从4减少到1，显存减少75%
```

#### 降低输入分辨率
```python
# 在数据预处理时添加降采样
def downsample_data(data, factor=2):
    """降低空间分辨率"""
    return data[:, :, ::factor, ::factor]
```

#### 减少输入时间步
```yaml
# config/config.yaml
input_days: 10  # 从20减少到10，显存减少50%
```

### 方案3: 高级优化技术

#### 混合精度训练
```python
from torch.cuda.amp import autocast, GradScaler

# 在训练循环中使用
scaler = GradScaler()
with autocast():
    pred = model(x)
    loss = criterion(pred, y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### 梯度累积
```python
# 模拟大batch_size而不增加显存
accumulation_steps = 4
for i, (x, y) in enumerate(train_loader):
    pred = model(x)
    loss = criterion(pred, y) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### 梯度检查点
```python
import torch.utils.checkpoint as checkpoint

# 在模型forward中使用
def forward(self, x):
    # 使用checkpoint减少中间激活值存储
    x = checkpoint.checkpoint(self.conv3d1, x)
    return x
```

## 实际使用建议

### 第一步: 快速测试
```bash
# 使用最小模型验证流程
python model_switch_example.py --switch 3dcnn_lstm_micro
python train.py
```

### 第二步: 逐步升级
如果micro版本能正常运行：
```bash
# 尝试lite版本
python model_switch_example.py --switch 3dcnn_lstm_lite
python train.py
```

### 第三步: 性能对比
记录不同模型的：
- 训练时间
- 显存占用
- 验证损失
- 最终预测效果

## 显存监控

### 使用nvidia-smi监控
```bash
# 实时监控GPU使用情况
watch -n 1 nvidia-smi
```

### 在代码中监控
```python
import torch

def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated")
        print(f"GPU Memory: {torch.cuda.memory_reserved()/1e9:.2f}GB reserved")

# 在训练循环中调用
print_gpu_memory()
```

## 性能预期

### 显存使用预期 (batch_size=4, 原始分辨率)
- **ConvLSTM**: ~0.7GB
- **3DCNN-LSTM Micro**: ~0.7GB  
- **3DCNN-LSTM Lite**: ~0.8GB
- **3DCNN-LSTM Original**: ~1.2GB+

### 训练速度预期
- **Micro版本**: 最快，接近ConvLSTM
- **Lite版本**: 中等，约为原版的1.5-2倍速度
- **Original版本**: 最慢，但可能效果最好

## 故障排除

### CUDA Out of Memory错误
1. 减小batch_size到1
2. 切换到更小的模型变体
3. 降低输入分辨率
4. 减少input_days

### 训练太慢
1. 使用3dcnn_lstm_lite而不是original
2. 增加batch_size（如果显存允许）
3. 使用混合精度训练

### 效果不佳
1. 先确保micro/lite版本能正常收敛
2. 逐步增加模型复杂度
3. 调整学习率和其他超参数

## 推荐配置

### 对于大多数用户 (推荐)
```yaml
model_type: "3dcnn_lstm_lite"
batch_size: 2
input_days: 15
```

### 对于显存受限用户
```yaml
model_type: "3dcnn_lstm_micro"  
batch_size: 1
input_days: 10
```

### 对于高端GPU用户
```yaml
model_type: "3dcnn_lstm"
batch_size: 4
input_days: 20
```
