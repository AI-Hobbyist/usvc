import torch.nn as nn


class BiGRU(nn.Module):
    def __init__(self, i, h, n): super().__init__(); self.gru = nn.GRU(i, h, num_layers=n, batch_first=True, bidirectional=True)
    def forward(self, x): out, _ = self.gru(x); return out


class BiLSTM(nn.Module):
    def __init__(self, i, h, n): super().__init__(); self.lstm = nn.LSTM(i, h, num_layers=n, batch_first=True, bidirectional=True)
    def forward(self, x): out, _ = self.lstm(x); return out