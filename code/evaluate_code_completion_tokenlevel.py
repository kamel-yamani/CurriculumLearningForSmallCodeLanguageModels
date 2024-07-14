import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import re
import pandas as pd
from model import GPT

class ScriptEvaluator:
    
    EVAL_DATA_DIR= 'data/TinyPy_processed/all'
    MODELS_DIR = os.path.join('models_checkpoint')
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results') 
    
    # Check if MODELS_DIR exists, if not, create it
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    # Check if RESULTS_DIR exists, if not, create it
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # Constants for dataset and file paths
    MODEL_FILE = 'hybrid_cl_model_1M.pth'
    ACCURACY_FILE = os.path.join(RESULTS_DIR,  '1M_accuracy_code_completion_tokenlevel_hybrid.txt')
    RESULTS_FILE = os.path.join(RESULTS_DIR,  '1M_results_code_completion_tokenlevel_hybrid.csv')
    
    
    def __init__(self):
        self.dataset = self.EVAL_DATA_DIR
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(1337)
        self.test_data, self.meta = self.load_dataset()
        self.m = self.load_model()

    def load_dataset(self):
        
        test_data = np.memmap(os.path.join(self.dataset, 'test.bin'), dtype=np.uint16, mode='r')
        meta_path = os.path.join(self.dataset, 'meta.pkl')
        meta_vocab_size = None
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            meta_vocab_size = meta['vocab_size']
            print(f"found vocab_size = {meta_vocab_size} (inside {meta_path}")

        return test_data, meta

    def load_model(self):
        model = GPT()
        print("Compiling model...")
        model = torch.compile(model)  
        model_path = os.path.join(self.MODELS_DIR, self.MODEL_FILE)
        model.load_state_dict(torch.load(model_path))
        m = model.to(self.device)
        return m

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

    def main(self):
        # Extracting stoi and itos from meta
        self.stoi = self.meta['stoi']
        self.itos = self.meta['itos']

        examples = self.decode(self.test_data).split("\n\n")
        examples = [example for example in examples if example]

        correct_predictions = 0
        total_predictions = 0

        results = []

        for code_snippet in tqdm(examples):

            tokens = torch.tensor(self.encode(code_snippet), dtype=torch.long).unsqueeze(0).to(self.device)

            for i in range(1, tokens.shape[1]):

                context = tokens[:, :i]
                actual_next_token = tokens[:, i].item()
                predicted_next_token = self.m.generate(context, max_new_tokens=1)
                predicted_next_token = predicted_next_token[:, -1].item()
                is_correct = (predicted_next_token == actual_next_token)

                if is_correct:
                    correct_predictions += 1
                results.append({
                    'context': context.cpu(),
                    'actual_next_token': actual_next_token,
                    'predicted_next_token': predicted_next_token,
                    'is_correct': is_correct
                })

                total_predictions += 1

        accuracy = (correct_predictions / total_predictions)
        
        print(f"Accuracy: {accuracy * 100:.2f}%")
        
        # Store accuracy in a file
        with open(self.ACCURACY_FILE, 'w') as f:
            f.write(f"Accuracy: {accuracy * 100:.2f}%\n")

        # Store results in a CSV file
        df = pd.DataFrame(results)
        df.to_csv(self.RESULTS_FILE, index=False)


if __name__ == "__main__":

    evaluator = ScriptEvaluator()

    evaluator.main()
