# Curriculum Learning for Code Language Models

This repository contains code, data and models_checkpoint for the paper "Curriculum Learning for Code Language Models". It implements training and evaluation of code language models on the TinyPy dataset using various curriculum learning techniques.

## Directory Structure

```
├── README.md
├── code
│   ├── evaluate_*.py - Evaluation scripts
│   ├── model.py - Model definition
│   ├── train_*.py - Training scripts for different techniques
│   └── pycache
├── data
│   ├── TinyPy - Raw TinyPy dataset
│   │   ├── [Split] - Data splits
│   └── TinyPy_processed - Preprocessed binary data  
├── models_checkpoint - Saved model checkpoints
```

## Models

The following models are included:

- `baseline` - Baseline model trained on all data
- `hybrid_cl` - Hybrid curriculum learning 
- `incremental_cl` - Incremental curriculum learning
- `sequential_cl` - Sequential curriculum learning

## Dependencies

- numpy
- pandas 
- torch
- tqdm
- wandb (optional)
- fuzzywuzzy

## Usage

Train a model:

```bash
python train_baseline.py
``` 

Evaluate a model:

```bash
python evaluate_code_execution.py
```
