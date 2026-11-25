import torch
from dataset import *
from tokenizer import *
from config import *
from model import *
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

def train(epochs):
    device = torch.device('cuda')
    dataloader = get_dataloader()
    tokenizer = JiebaTokenizer.from_vocab(PROCESSED_DATA_DIR / 'vocab.txt')
    model = MyModel(tokenizer.vocab_size, tokenizer.pad_token_idx).to(device)
    model.train()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    log_dir = LOGS_DIR / datetime.now().strftime('%Y%m%d-%H%M%S-%f')

    with SummaryWriter(log_dir=str(log_dir), flush_secs=10) as writer:
        for epoch in range(epochs):
            total_loss = 0.0
            with tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as progress:
                for x, y in progress:
                    x, y = x.to(device), y.to(device)
                    y_pred = model(x).squeeze(-1)
                    loss = criterion(y_pred, y)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    batch_loss = loss.item()
                    total_loss += batch_loss * y.size(0)
                    progress.set_postfix(batch_loss=f"{batch_loss:.4f}", refresh=False)

            this_loss = total_loss / len(dataloader.dataset)
            writer.add_scalar("Loss/train", this_loss, epoch + 1)

if __name__ == "__main__":
    train(epochs=EPOCHS)
