import os
import sys
import json
import requests
import concurrent.futures
from time import sleep
from tqdm import tqdm
import argparse
import torch
from typing import Dict, List
from utils import load_dataset_from_file, save_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class ArmoRMPipeline:
    def __init__(self, model_id, device_map="auto", torch_dtype=torch.bfloat16, truncation=True, trust_remote_code=False, max_length=4096):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, float]:
        """
        messages: OpenAI chat messages to be scored
        Note: no batching since due to length differences, the model will have to pad to the max length which is not efficient
        Returns: a dictionary with the score between 0 and 1
        """
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
            score = output.score.float().item()
        return {"score": score}

################
# Configurations
################
def get_args():
    # Experiment Settings
    parser = argparse.ArgumentParser(description="Tagging Manager.")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Input dataset file name")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of samples per batch.")
    parser.add_argument("--checkpoint_every", type=int, default=5000, help="Save checkpoint every n batches")

    # Generation Configs
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--top_k", type=int, default=1) # 1 means greedy decoding
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--stop_tokens", type=str, default="<|eot_id|>", help="Stop token")

    return parser.parse_args()
args = get_args()

checkpoint_every = args.checkpoint_every
batch_size = args.batch_size

# Generate outputs, update dataset in batches, and overwrite checkpoint
def generate_and_update(dataset, model, checkpoint_file, checkpoint_every = 20):
    if os.path.exists(checkpoint_file):
        last_checkpoint_idx = len(load_dataset_from_file(checkpoint_file))
        print(f"Checkpoint file found. Resuming from last checkpoint with index {last_checkpoint_idx}.")
        dataset[:last_checkpoint_idx] = load_dataset_from_file(checkpoint_file)
        num_batches = (len(dataset) - last_checkpoint_idx + batch_size - 1) // batch_size
        print(f"Remaining number of batches: {num_batches}")
    else:
        last_checkpoint_idx = 0
        num_batches = (len(dataset) + batch_size - 1) // batch_size  # Calculate total number of batches
        print(f"Total number of batches: {num_batches}")

    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size + last_checkpoint_idx
        end_idx = min((i + 1) * batch_size + last_checkpoint_idx, len(dataset))
        batch = dataset[start_idx:end_idx]

        for item in batch:
            rewards = []
            instruction = item['instruction']
            for response in item['responses']:
                reward_armorm = model([{"role": "user", "content": instruction}, {"role": "assistant", "content": response}])
                rewards.append(reward_armorm)

            item['rewards_armorm'] = rewards

        dataset[start_idx:end_idx] = batch
        # Overwrite the same checkpoint file every checkpoint_every batches

        if (i + 1) % checkpoint_every == 0:
            save_dataset(dataset[:end_idx], checkpoint_file)
            print(f"Dataset checkpoint saved after batch {i + 1}.")
        i += 1

    return dataset

# main
input_file = args.input_file
output_file = f"{input_file[:input_file.rfind('.')]}_armorm.json"
checkpoint_file = f"{input_file[:input_file.rfind('.')]}_armorm_checkpoint.json"
dataset = load_dataset_from_file(input_file)

model = ArmoRMPipeline("RLHFlow/ArmoRM-Llama3-8B-v0.1", trust_remote_code=True, device_map=f"cuda:{args.device}")

updated_dataset = generate_and_update(dataset, model, checkpoint_file, checkpoint_every)

save_dataset(updated_dataset, output_file, convert_to_jsonl=False)

# Remove the checkpoint file after completion
if os.path.exists(checkpoint_file):
    os.remove(checkpoint_file)
    print("Final dataset saved. Checkpoint removed.")