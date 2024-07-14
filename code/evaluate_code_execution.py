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
    ACCURACY_FILE = os.path.join(RESULTS_DIR,  '1M_accuracy_code_execution_hybrid.txt')
    RESULTS_FILE = os.path.join(RESULTS_DIR,  '1M_results_code_execution_hybrid.csv')
    
    
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

    def evaluate_example(self, example, max_new_tokens=30):
        splited_example = example.split("# output")
        if not ("for" in splited_example[0]) :
            max_new_tokens = 20
        encoded_example = torch.tensor(self.encode(splited_example[0] + "# output"), dtype=torch.long).unsqueeze(0).to(self.device)

        prompt_text = splited_example[0] + "# output"
        result_example = splited_example[-1]

        real_results = [float(match.group()) for match in re.finditer(r"\d+\.\d+|\d+|-\d+|-\d+\.\d+", result_example.split('\n\n')[0].replace("\n", ""))]

        response = self.decode(self.m.generate(encoded_example, max_new_tokens=max_new_tokens)[0].tolist())
        splited_response = response.split("# output")
        result_response = splited_response[-1]
        generated_results = [float(match.group()) for match in re.finditer(r"\d+\.\d+|\d+|-\d+|-\d+\.\d+", result_response.split('\n\n')[0].replace("\n", ""))]

        return prompt_text, real_results, generated_results

    def write_results_to_file(self, output_file, prompt, real_results, generated_results):
        df = pd.DataFrame({
            'Prompt': prompt,
            'Real_Results': real_results,
            'Generated_Results': generated_results
        })
        df.to_csv(output_file, index=False)

    def main(self):
        # Extracting stoi and itos from meta
        self.stoi = self.meta['stoi']
        self.itos = self.meta['itos']

        examples = self.decode(self.test_data).split("\n\n")
        examples = [example for example in examples if example]

        prompt = []
        real_results = []
        generated_results = []

        for example in tqdm(examples):
            prompt_text, real_result, result = self.evaluate_example(example)
            prompt.append(prompt_text)
            real_results.append(real_result)
            generated_results.append(result)

        correct_count = sum(1 for real, generated in zip(real_results, generated_results) if real == generated)
        accuracy = correct_count / len(generated_results)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        
        # Store accuracy in a file
        with open(self.ACCURACY_FILE, 'w') as f:
            f.write(f"Accuracy: {accuracy * 100:.2f}%\n")

        # Store results in a CSV file
        self.write_results_to_file(self.RESULTS_FILE, prompt, real_results, generated_results)


if __name__ == "__main__":

    evaluator = ScriptEvaluator()

    evaluator.main()
