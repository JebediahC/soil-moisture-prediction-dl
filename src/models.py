import torch
import torch.nn as nn
from . import utils

config = utils.load_config()

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):  # hidden_dim可调小
        super().__init__()
        self.cell = ConvLSTMCell(input_dim, hidden_dim)
        self.out = nn.Conv2d(hidden_dim, 1, 1)

    def forward(self, x_seq):
        b, t, c, h, w = x_seq.size()
        h_t = torch.zeros((b, 16, h, w), device=x_seq.device)
        c_t = torch.zeros((b, 16, h, w), device=x_seq.device)
        for t_step in range(t):
            h_t, c_t = self.cell(x_seq[:, t_step], h_t, c_t)
        out = self.out(h_t)
        return out.unsqueeze(1).repeat(1, config["predict_days"], 1, 1, 1)


class CNN3D_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, lstm_hidden_dim=32):
        super().__init__()
        
        # 3D CNN部分用于提取时空特征
        self.conv3d1 = nn.Conv3d(input_dim, hidden_dim//4, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(hidden_dim//4)
        self.conv3d2 = nn.Conv3d(hidden_dim//4, hidden_dim//2, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(hidden_dim//2)
        self.conv3d3 = nn.Conv3d(hidden_dim//2, hidden_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(hidden_dim)
        
        # 自适应池化，将时间维度减少到合适的大小
        self.adaptive_pool = nn.AdaptiveAvgPool3d((8, None, None))  # 时间维度压缩到8
        
        # LSTM部分
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True, bidirectional=True)
        
        # 输出层
        self.conv_out = nn.Conv2d(lstm_hidden_dim * 2, hidden_dim//2, kernel_size=3, padding=1)
        self.bn_out = nn.BatchNorm2d(hidden_dim//2)
        self.final_out = nn.Conv2d(hidden_dim//2, 1, kernel_size=1)
        
        self.dropout = nn.Dropout3d(0.1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x_seq):
        # x_seq shape: (batch, time, channels, height, width)
        b, t, c, h, w = x_seq.size()
        
        # 转换为3D卷积的输入格式: (batch, channels, time, height, width)
        x = x_seq.permute(0, 2, 1, 3, 4)
        
        # 3D CNN特征提取
        x = self.relu(self.bn1(self.conv3d1(x)))
        x = self.dropout(x)
        
        x = self.relu(self.bn2(self.conv3d2(x)))
        x = self.dropout(x)
        
        x = self.relu(self.bn3(self.conv3d3(x)))
        x = self.dropout(x)
        
        # 自适应池化压缩时间维度
        x = self.adaptive_pool(x)  # (batch, hidden_dim, 8, h, w)
        
        # 重新排列为LSTM输入格式 - 改进的方法
        _, feature_dim, new_t, h, w = x.size()
        
        # 将空间维度平铺，时间维度保持
        x = x.permute(0, 2, 3, 4, 1)  # (batch, new_t, h, w, feature_dim)
        x = x.contiguous().view(b, new_t, h * w, feature_dim)  # (batch, new_t, h*w, feature_dim)
        
        # 重塑为LSTM期望的格式
        x = x.view(b * h * w, new_t, feature_dim)  # (batch*h*w, new_t, feature_dim)
        
        # LSTM处理
        lstm_out, _ = self.lstm(x)  # (batch*h*w, new_t, lstm_hidden_dim*2)
        
        # 取最后一个时间步的输出
        final_lstm_out = lstm_out[:, -1, :]  # (batch*h*w, lstm_hidden_dim*2)
        
        # 重新整形回空间维度
        final_features = final_lstm_out.view(b, h, w, -1)  # (batch, h, w, lstm_hidden_dim*2)
        final_features = final_features.permute(0, 3, 1, 2)  # (batch, lstm_hidden_dim*2, h, w)
        
        # 输出层
        out = self.relu(self.bn_out(self.conv_out(final_features)))
        out = self.final_out(out)  # (batch, 1, h, w)
        
        # 重复预测天数
        out = out.unsqueeze(1).repeat(1, config["predict_days"], 1, 1, 1)
        
        return out


class CNN3D_LSTM_Lite(nn.Module):
    """
    轻量版3DCNN-LSTM模型，专为显存优化设计
    """
    def __init__(self, input_dim, hidden_dim=32, lstm_hidden_dim=16):
        super().__init__()
        
        # 更轻量的3D CNN部分
        self.conv3d1 = nn.Conv3d(input_dim, hidden_dim//4, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(hidden_dim//4)
        self.conv3d2 = nn.Conv3d(hidden_dim//4, hidden_dim//2, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(hidden_dim//2)
        # 去掉第三层3D卷积，减少参数量
        
        # 更激进的时间维度压缩
        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, None, None))  # 时间维度压缩到4
        
        # 更小的LSTM
        self.lstm = nn.LSTM(hidden_dim//2, lstm_hidden_dim, batch_first=True, bidirectional=True)
        
        # 简化的输出层
        self.final_out = nn.Conv2d(lstm_hidden_dim * 2, 1, kernel_size=1)
        
        self.dropout = nn.Dropout3d(0.15)  # 稍微增加dropout防止过拟合
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x_seq):
        # x_seq shape: (batch, time, channels, height, width)
        b, t, c, h, w = x_seq.size()
        
        # 转换为3D卷积的输入格式: (batch, channels, time, height, width)
        x = x_seq.permute(0, 2, 1, 3, 4)
        
        # 轻量3D CNN特征提取
        x = self.relu(self.bn1(self.conv3d1(x)))
        x = self.dropout(x)
        
        x = self.relu(self.bn2(self.conv3d2(x)))
        x = self.dropout(x)
        
        # 自适应池化压缩时间维度
        x = self.adaptive_pool(x)  # (batch, hidden_dim//2, 4, h, w)
        
        # 重新排列为LSTM输入格式 - 改进的方法
        _, feature_dim, new_t, h, w = x.size()
        
        # 将空间维度平铺，时间维度保持
        # 形状变换: (batch, feature_dim, new_t, h, w) -> (batch, new_t, h*w, feature_dim)
        x = x.permute(0, 2, 3, 4, 1)  # (batch, new_t, h, w, feature_dim)
        x = x.contiguous().view(b, new_t, h * w, feature_dim)  # (batch, new_t, h*w, feature_dim)
        
        # 重塑为LSTM期望的格式: (batch, seq_len, input_size)
        # 我们将每个像素位置作为一个独立的序列处理
        x = x.view(b * h * w, new_t, feature_dim)  # (batch*h*w, new_t, feature_dim)
        
        # LSTM处理 - batch_first=True
        lstm_out, _ = self.lstm(x)  # (batch*h*w, new_t, lstm_hidden_dim*2)
        
        # 取最后一个时间步的输出
        final_lstm_out = lstm_out[:, -1, :]  # (batch*h*w, lstm_hidden_dim*2)
        
        # 重新整形回空间维度
        final_features = final_lstm_out.view(b, h, w, -1)  # (batch, h, w, lstm_hidden_dim*2)
        final_features = final_features.permute(0, 3, 1, 2)  # (batch, lstm_hidden_dim*2, h, w)
        
        # 直接输出，减少计算
        out = self.final_out(final_features)  # (batch, 1, h, w)
        
        # 重复预测天数
        out = out.unsqueeze(1).repeat(1, config["predict_days"], 1, 1, 1)
        
        return out


class CNN3D_LSTM_Micro(nn.Module):
    """
    微型版3DCNN-LSTM模型，极度显存优化
    """
    def __init__(self, input_dim, hidden_dim=16, lstm_hidden_dim=8):
        super().__init__()
        
        # 最小的3D CNN
        self.conv3d1 = nn.Conv3d(input_dim, hidden_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(hidden_dim)
        
        # 强时间压缩
        self.adaptive_pool = nn.AdaptiveAvgPool3d((2, None, None))
        
        # 最小LSTM
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, batch_first=True)
        
        # 直接输出
        self.final_out = nn.Conv2d(lstm_hidden_dim, 1, kernel_size=1)
        
        self.dropout = nn.Dropout3d(0.2)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x_seq):
        b, t, c, h, w = x_seq.size()
        x = x_seq.permute(0, 2, 1, 3, 4)
        
        # 单层3D CNN
        x = self.relu(self.bn1(self.conv3d1(x)))
        x = self.dropout(x)
        
        # 压缩时间维度
        x = self.adaptive_pool(x)  # (batch, hidden_dim, 2, h, w)
        
        # LSTM处理 - 改进的方法
        _, feature_dim, new_t, h, w = x.size()
        
        # 重新排列为LSTM期望的格式
        x = x.permute(0, 2, 3, 4, 1)  # (batch, new_t, h, w, feature_dim)
        x = x.contiguous().view(b, new_t, h * w, feature_dim)  # (batch, new_t, h*w, feature_dim)
        x = x.view(b * h * w, new_t, feature_dim)  # (batch*h*w, new_t, feature_dim)
        
        # LSTM处理
        lstm_out, _ = self.lstm(x)  # (batch*h*w, new_t, lstm_hidden_dim)
        
        # 取最后一个时间步的输出
        final_lstm_out = lstm_out[:, -1, :]  # (batch*h*w, lstm_hidden_dim)
        
        # 重整并输出
        final_features = final_lstm_out.view(b, h, w, -1).permute(0, 3, 1, 2)  # (batch, lstm_hidden_dim, h, w)
        
        out = self.final_out(final_features)
        out = out.unsqueeze(1).repeat(1, config["predict_days"], 1, 1, 1)
        
        return out


def get_model(model_type, input_dim, **kwargs):
    """
    模型工厂函数，用于创建不同类型的模型
    
    Args:
        model_type: 模型类型，可选:
                   - 'convlstm': 原始ConvLSTM模型
                   - '3dcnn_lstm': 原始3DCNN-LSTM模型
                   - '3dcnn_lstm_lite': 轻量版3DCNN-LSTM模型 
                   - '3dcnn_lstm_micro': 微型版3DCNN-LSTM模型
        input_dim: 输入通道数
        **kwargs: 其他模型参数
    
    Returns:
        相应的模型实例
    """
    if model_type.lower() == 'convlstm':
        hidden_dim = kwargs.get('hidden_dim', 16)
        return ConvLSTM(input_dim, hidden_dim)
    elif model_type.lower() == '3dcnn_lstm':
        hidden_dim = kwargs.get('hidden_dim', 64)
        lstm_hidden_dim = kwargs.get('lstm_hidden_dim', 32)
        return CNN3D_LSTM(input_dim, hidden_dim, lstm_hidden_dim)
    elif model_type.lower() == '3dcnn_lstm_lite':
        hidden_dim = kwargs.get('hidden_dim', 32)
        lstm_hidden_dim = kwargs.get('lstm_hidden_dim', 16)
        return CNN3D_LSTM_Lite(input_dim, hidden_dim, lstm_hidden_dim)
    elif model_type.lower() == '3dcnn_lstm_micro':
        hidden_dim = kwargs.get('hidden_dim', 16)
        lstm_hidden_dim = kwargs.get('lstm_hidden_dim', 8)
        return CNN3D_LSTM_Micro(input_dim, hidden_dim, lstm_hidden_dim)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose from 'convlstm', '3dcnn_lstm', '3dcnn_lstm_lite', or '3dcnn_lstm_micro'")