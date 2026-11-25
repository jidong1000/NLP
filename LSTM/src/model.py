import torch
from torch import nn
from config import *

class MyModel(nn.Module):
    def __init__(self, vocab_size, padding_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=padding_idx)
        self.lstm = nn.LSTM(input_size=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, batch_first=True)
        self.linear = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        embed = self.embedding(x)
        lstm, (_, _) = self.lstm(embed)
        # 获取每个样本真实的最后一个token的隐藏状态
        batch_indexes = torch.arange(0, lstm.shape[0])
        lengths = (x != self.embedding.padding_idx).sum(dim=1)
        output = self.linear(lstm[batch_indexes, lengths - 1])

        return output
