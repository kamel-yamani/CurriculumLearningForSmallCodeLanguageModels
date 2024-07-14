import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import re
import pandas as pd
from fuzzywuzzy import fuzz

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
    ACCURACY_FILE = os.path.join(RESULTS_DIR,  '1M_accuracy_code_completion_linelevel_hybrid.txt')
    RESULTS_FILE = os.path.join(RESULTS_DIR,  '1M_results_code_completion_linelevel_hybrid.csv')
    
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
        
        total_similarity = 0
        total_comparisons = 0

        results = []

        for code_snippet in tqdm(examples):

            lines = code_snippet.split('\n')
            for i in range(1, len(lines)):

                context_lines = lines[:i]
                actual_next_line = lines[i]

                context_tokens = torch.tensor(self.encode('\n'.join(context_lines) + '\n'), dtype=torch.long).unsqueeze(0).to(self.device)
                actual_next_line_tokens = torch.tensor(self.encode(actual_next_line), dtype=torch.long).unsqueeze(0).to(self.device)

                n = actual_next_line_tokens.shape[1]  # Limit to length of actual next line
                predicted_next_line_tokens = self.m.generate(context_tokens, max_new_tokens=n)
                predicted_next_line_tokens = predicted_next_line_tokens[:, -n:]
                is_correct = torch.equal(predicted_next_line_tokens, actual_next_line_tokens)
                
                decoded_predicted_next_line_tokens = self.decode(predicted_next_line_tokens.tolist()[0])
                decoded_actual_next_line_tokens = self.decode(actual_next_line_tokens.tolist()[0])
                
                similarity = fuzz.ratio(decoded_predicted_next_line_tokens, decoded_actual_next_line_tokens)
                total_similarity += similarity
                total_comparisons += 1
              
                if is_correct:
                    correct_predictions += 1
                results.append({
                    'context': context_tokens.cpu(),
                    'actual_next_line': actual_next_line_tokens.cpu(),
                    'predicted_next_line': predicted_next_line_tokens.cpu(),
                    'is_correct': is_correct,
                    'edit_similarity': similarity
                })

                total_predictions += 1

        accuracy = (correct_predictions / total_predictions)
        
        print(f"Accuracy: {accuracy * 100:.2f}%")
        
        average_similarity = total_similarity / total_comparisons
        
        print(f"Edit Similarity: {average_similarity :.2f}")
        
        # Store accuracy in a file
        with open(self.ACCURACY_FILE, 'w') as f:
            f.write(f"Accuracy: {accuracy * 100:.2f}%\nEdit Similarity: {average_similarity :.2f}")

        # Store results in a CSV file
        df = pd.DataFrame(results)
        df.to_csv(self.RESULTS_FILE, index=False)


if __name__ == "__main__":

    evaluator = ScriptEvaluator()

    evaluator.main()
