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
    return str(chr(max(200, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))


# accelerator

accelerator = Accelerator()
device = accelerator.device

# instantiate palm
model = (PaLM
    ( 
       num_tokens=5000,
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

with open("output1.txt") as file:
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

max_epochs = 5
for max_epochs in tqdm.tqdm(range(NUM_BATCHES), mininterval=1.0, desc="training"):




    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        loss = model(next(train_loader), return_loss = True)
        accelerator.backward(loss / GRADIENT_ACCUMULATE_EVERY)

  

    if max_epochs % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            loss = model(next(val_loader), return_loss = True)
            accelerator.print(f"validation loss: {loss.item()}")

    if max_epochs % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        prime = decode_tokens(inp)
        # accelerator.print(f"%s \n\n %s", (prime, "*" * 100))
        accelerator.print(f"\n{prime}\n\n", "*" * 10)
        sample = model.generate(GENERATE_LENGTH, inp[None, ...])
        output_str = decode_tokens(sample[0])
        accelerator.print("\n", output_str, "\n")

       



        
        
# save model
print("saving model")
ckpt_path = os.path.join(os.getcwd(), "model.pt")
torch.save(model.state_dict(), ckpt_path)
model.train()      
