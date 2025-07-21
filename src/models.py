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