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

DATA_DIR = 'data/TinyPy_processed/sequential'
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Check if MODELS_DIR exists, if not, create it
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# Hyperparameters
batch_size = 64  
block_size = 256 
eval_interval = 10000
learning_rate = 1e-3 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_iters = 500
dropout = 0
compile = True 

# Stages for Curriculum Learning

stages = ['LEVEL1', 'LEVEL2', 'LEVEL3']

stage_iterations = [40000, 40000, 40000]

# wandb 
wandb_log = False # disabled by default
wandb_project = 'WANDB_PROJECT_NAME'

# For logging purposes
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}

# Set random seed
torch.manual_seed(1337)

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

def human_readable(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


# Model
model = GPT()
m = model.to(device)
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0
    


num_parameters_hr = human_readable(num_parameters)
print(f'The model has {num_parameters_hr} trainable parameters')

# Get current date and hour
now = datetime.datetime.now()
date_hour = now.strftime("%Y-%m-%d_%H-%M")

# Construct wandb_run_name
wandb_run_name = f'TLM_RUN_{num_parameters_hr}_{date_hour}'


# Train
if wandb_log :
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

total_training_time = 0

# Train for each stage
for i, dataset in enumerate(stages):

    
    data_dir = os.path.join( DATA_DIR, dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    max_iters = stage_iterations[i]
    miles = [int(max_iters * m) for m in [0.7, 0.8, 0.9]]
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=miles, gamma=0.1)
    
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
    
    stage_training_time = (end_time - start_time) / 60
    total_training_time += stage_training_time
    
    print(f'Training time for dataset {dataset}: {stage_training_time}  min')
    torch.save(model.state_dict(), f"{MODELS_DIR}/{num_parameters_hr}_{date_hour}_TEMP{dataset}.pth")
    
    

print(f'Total training time: {total_training_time} min')

# Save the final model after all stages
torch.save(model.state_dict(), f"{MODELS_DIR}/{num_parameters_hr}_{date_hour}_final.pth")
