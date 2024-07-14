# Curriculum Learning for Small Code Language Models

This repository contains code and models_checkpoint for the paper "Curriculum Learning for Small Code Language Models". It implements training and evaluation of code language models on the TinyPy dataset using various curriculum learning techniques.

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

- `baseline_model_1M.pth` - Baseline model trained on all data
- `hybrid_cl_model_1M.pth` - Hybrid curriculum learning 
- `incremental_cl_model_1M.pth` - Incremental curriculum learning
- `sequential_cl_model_1M.pth` - Sequential curriculum learning

## Data

The `data` folder should be added manually by downloading the dataset from Kaggle:

1. Create a new folder named `data` inside the `code` folder.
2. Go to the Kaggle dataset [TinyPy for Curriculum Learning](https://www.kaggle.com/datasets/kamelmohammedyamani/tinypy-for-curriculum-learning).
3. Download the dataset.
5. Extract the contents and place them in the `data` folder.

## Dependencies

- numpy
- pandas 
- torch
- tqdm
- wandb (optional)
- fuzzywuzzy

## Usage

### Train a model

To train a model, use one of the training scripts. For example, to train the baseline model:

```bash
python train_baseline.py
```

To train with hybrid curriculum learning:

```bash
python code/train_hybrid_cl.py
```

To train with incremental curriculum learning:

```bash
python code/train_incremental_cl.py
```

To train with sequential curriculum learning:

```bash
python code/train_sequential_cl.py
```

### Evaluate a model

To evaluate a model, use one of the evaluation scripts. For example, to evaluate the code execution:

```bash
python evaluate_code_execution.py
```

To evaluate code completion at the token level:

```bash
python code/evaluate_code_completion_tokenlevel.py
```

To evaluate code completion at the line level:

```bash
python code/evaluate_code_completion_linelevel.py
```

## Acknowledgement

This work was supported in part through the NYU IT High Performance Computing resources, services, and staff expertise
