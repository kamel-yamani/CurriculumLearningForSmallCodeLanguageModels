import os
import time
import datetime
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import numpy as np
import pandas as pd
from tqdm import tqdm

from model import GPT

# To train the complex model or without output, just change the data path
DATA_DIR = 'data/TinyPy_processed/all'
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Check if MODELS_DIR exists, if not, create it
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# Hyperparameters
batch_size = 64  
block_size = 256 
max_iters = 120000  
miles = [int(max_iters * m) for m in [0.7, 0.8, 0.9]]
eval_interval = 10000
learning_rate = 1e-3 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_iters = 500

dropout = 0

compile = True 

# wandb 
wandb_log = False # disabled by default
wandb_project = 'WANDB_PROJECT_NAME'

# For logging purposes
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}

# Set random seed
torch.manual_seed(1337)

train_data = np.memmap(os.path.join(DATA_DIR,  'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(DATA_DIR, 'val.bin'), dtype=np.uint16, mode='r')

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(DATA_DIR, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# Get random batch of data
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Estimate loss on train and test splits
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters) 
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Model
model = GPT()
m = model.to(device)
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0
    
def human_readable(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

num_parameters_hr = human_readable(num_parameters)
print(f'The model has {num_parameters_hr} trainable parameters')

# Get current date and hour
now = datetime.datetime.now()
date_hour = now.strftime("%Y-%m-%d_%H-%M")

# Construct wandb_run_name
wandb_run_name = f'TLM_RUN_{num_parameters_hr}_{date_hour}'

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Scheduler
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=miles, gamma=0.1)

# Train
if wandb_log :
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# Calculate training time 
start_time = time.time()

for iter in range(max_iters):

    # evaluate the model on the train and val splits and log the losses
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'iter {iter:5d} | train loss {losses["train"]:.4f} | val loss {losses["val"]:.4f}')
        if wandb_log:
            wandb.log({
                "iter": iter,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": scheduler.get_last_lr()[0],
            })

    # train the model for one iteration
    xb, yb = get_batch('train')

    # forward pass
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # Step the scheduler
    scheduler.step()

end_time = time.time()
print(f'Training time: {(end_time - start_time) / 60}  min')

torch.save(model.state_dict(), f"{MODELS_DIR}/{num_parameters_hr}_{date_hour}.pth")