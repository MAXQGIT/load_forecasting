import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len*2

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len, :]
        y = self.data[idx + self.seq_len:idx + self.seq_len*2, -1]
        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)
