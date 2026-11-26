import torch
from torch import nn

class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layer, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        gru, _ = self.gru(x)
        y = self.linear(gru)

        return y
