import torch
from torch.utils.data import Dataset, DataLoader

class SoilDataset(Dataset):
    def __init__(self, data, input_days, pred_days):
        self.data = data
        self.input_days = input_days
        self.pred_days = pred_days
        self.length = data.shape[0] - input_days - pred_days + 1
        print(f"Available samples: {self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.input_days]  # (input_days, C, H, W)
        y = self.data[idx + self.input_days:idx + self.input_days + self.pred_days, 0:1]  # only swvl1
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)