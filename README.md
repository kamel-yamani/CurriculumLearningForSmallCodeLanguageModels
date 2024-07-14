# Curriculum Learning for Code Language Models

This repository contains code, data and models_checkpoint for the paper "Curriculum Learning for Small Code Language Models". It implements training and evaluation of code language models on the TinyPy dataset using various curriculum learning techniques.

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

1. Go to the Kaggle dataset [TinyPy for Curriculum Learning](https://www.kaggle.com/datasets/kamelmohammedyamani/tinypy-for-curriculum-learning).
2. Download the dataset.
3. Extract the contents and place them in the `data` folder in the following structure:

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
python train_baseline.py
```

To train with incremental curriculum learning:

```bash
python train_baseline.py
```

To train with sequential curriculum learning:

```bash
python train_baseline.py
```

### Evaluate a model

To evaluate a model, use one of the evaluation scripts. For example, to evaluate the code execution:

```bash
python evaluate_code_execution.py
```

To evaluate code completion at the line level:

```bash
python code/evaluate_code_completion_linelevel.py
```

## Acknowledgement

This work was supported in part through the NYU IT High Performance Computing resources, services, and staff expertise"
