from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from config import *

class MyDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_json(path, lines=True, orient='records').to_dict(orient='records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_tensor = torch.tensor(self.data[idx]['review'], dtype=torch.long)
        target_tensor = torch.tensor(self.data[idx]['label'], dtype=torch.float32)

        return input_tensor, target_tensor

def get_dataloader(train=True):
    dataset = MyDataset(PROCESSED_DATA_DIR / 'train.jsonl' if train == True else PROCESSED_DATA_DIR / 'test.jsonl')

    return DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

if __name__ == "__main__":
    pass