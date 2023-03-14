import torch
from torch import nn
from models.sample_fns import sample
from datasets import WordDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import re


text = open("./data/100-0.txt", "r", encoding="utf-8-sig").read()

block_size = 128
train_dataset = WordDataset(text, block_size)

batch_size = 64
train_loader = DataLoader(
    train_dataset, shuffle=True, pin_memory=True, batch_size=batch_size
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

model = (
    DecoderOnlyTransformer(
        num_layers=8,
        num_heads=8,
        vocab_size=train_dataset.vocab_size,
        hidden_size=512,
        max_pos_embeddings=train_dataset.block_size,
        dropout=0.1,
    )
    .to(device)
    .train()
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=6e-4)

max_epochs = 15
for epoch in range(max_epochs):
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for it, (x, y) in pbar:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()

        optimizer.step()

        pbar.set_description(f"epoch {epoch} iter {it}: train loss {loss.item():.5f}")

    # eval:
    model.eval()
    with torch.no_grad():
        context = "O God, O God!"
        x = torch.tensor(
            [train_dataset.stoi[s] for s in re.split(r"\b", context)], dtype=torch.long
        )[None, ...].to(device)
        y = sample(model, x, 200, temperature=1.0, sample=True, top_k=10)[0]
        completion = "".join([train_dataset.itos[int(i)] for i in y])
        print(completion)

    # save model
    print("saving model")
    ckpt_path = os.path.join(os.getcwd(), "model.pt")
    torch.save(model.state_dict(), ckpt_path)
    model.train()

import gzip
import random
import tqdm
import numpy as np

import torch
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn
from palm_rlhf_pytorch import PaLM
from accelerate import Accelerator
import csv
import math
import os
# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 2
GRADIENT_ACCUMULATE_EVERY = 1
LEARNING_RATE = 2e-4
VALIDATE_EVERY = 20
PRIME_LENGTH = 128
GENERATE_EVERY = 200
GENERATE_LENGTH = 512
SEQ_LEN = 1024

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(112, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))


# accelerator

accelerator = Accelerator()
device = accelerator.device

# instantiate palm
model = (PaLM
    ( 
       num_tokens=10000,
        dim=512,
        depth=1,
        causal= True,
        dim_head = 64,
        heads= 8,
        ff_mult= 4,
        attn_dropout= 0.1,
        ff_dropout= 0.1,
        lora_r=8,
        rotary_xpos_scale_base= 512,
        cross_entropy_ignore_index = 0)
    .to(device)
    .train())
        
    




print("no. of parameters:  ", sum(p.numel() for p in model.parameters()))
# prepare merged data

with open("data/output1.txt") as file:
    X = np.fromstring(file.read(int(1e9)), dtype=np.uint8)
    trX, vaX = np.split(X, [math.floor(int(X.shape[0])*0.8)])
    print(trX.shape)
    print(vaX.shape)
    data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)


class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq.to(device)

    def __len__(self):
        return self.data.size(0) // self.seq_len

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))
val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE))

# print(train_dataset.__len__())

# for i in range(10):
#     print(train_dataset.__getitem__(i))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=6e-4)

max_epochs = 15
for epoch in range(max_epochs):
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for it, (x, y) in pbar:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()

        optimizer.step()

        


    # save model
    print("saving model")
    ckpt_path = os.path.join(os.getcwd(), "model.pt")
    torch.save(model.state_dict(), ckpt_path)
    model.train()
