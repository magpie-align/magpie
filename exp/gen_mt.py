import torch
import os
import sys
import argparse
import json
import requests
import concurrent.futures
from time import sleep
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import load_dataset_from_file, save_dataset, get_conversation_template
from vllm import LLM, SamplingParams

################
# Configurations
################
def get_args():
    # Experiment Settings
    parser = argparse.ArgumentParser(description="Response Generation Manager.")
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="We will support more models in the future.")
    parser.add_argument("--input_file", type=str, default=None, help="Input dataset file name")
    parser.add_argument("--num_turns", type=int, default=2, help="Number of turns for each conversation.")
    parser.add_argument("--batch_size", type=int, default=128, help="Number of samples per batch")
    parser.add_argument("--checkpoint_every", type=int, default=20, help="Save checkpoint every n batches")

    # Generation Parameters
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs to use for tensor parallelism. Only used for Llama 70B models.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--early_stopping", type=bool, default=True, help="Stop generation when the \n is generated.")
    parser.add_argument("--disable_early_stopping", action="store_false", dest="early_stopping", help="Disable early stopping.")
    parser.add_argument("--instruction_temperature", type=float, default=1.0)
    parser.add_argument("--instruction_top_p", type=float, default=1.0)
    parser.add_argument("--instruction_repetition_penalty", type=float, default=1.0)
    parser.add_argument("--response_temperature", type=float, default=0)
    parser.add_argument("--response_top_p", type=float, default=1.0)
    parser.add_argument("--response_repetition_penalty", type=float, default=1.0)
    parser.add_argument("--tokenizer_template", type=bool, default=False, help="Use tokenizer template for generating the response.")
    parser.add_argument("--use_tokenizer_template", action="store_true", dest="tokenizer_template")
    parser.add_argument("--debug", type=bool, default=False, help="Debug mode.")

    return parser.parse_args()

args = get_args()
print(f"Response Generation Manager. Arguments: {args}") # For logging

mt_system_prompt = "You are a helpful Al assistant. The user will engage in a multi-round conversation with you, asking initial questions and following up with additional related questions. Your goal is to provide thorough,relevant and insightful responses to help the user with their queries."

if args.input_file is None:
    raise ValueError("Please specify the input file path.")
if args.num_turns < 2:
    raise ValueError("Please specify a number of turns greater than 1.")

# Constants for the local vllm engine
MODEL_NAME = args.model_path
INPUT_FILE_NAME = args.input_file 
BATCH_SIZE = args.batch_size
CHECKPOINT_FILE = f"{INPUT_FILE_NAME[:INPUT_FILE_NAME.rfind('.')]}_mt_checkpoint.json"
CHECKPOINT_EVERY = args.checkpoint_every
SAVED_FILE = f"{INPUT_FILE_NAME[:INPUT_FILE_NAME.rfind('.')]}_mt.json"

# Obtain config from configs/model_configs.json
with open("../configs/model_configs.json", "r") as f:
    model_configs = json.load(f)
    model_config = model_configs[args.model_path]
    stop_tokens = model_config["stop_tokens"]
    stop_token_ids = model_config["stop_token_ids"]
    mt_append_template = model_config["mt_append_template"]

    if args.early_stopping:
        stop_tokens.append("\n")

# Process a batch of data using local vllm engine
def process_batch(batch, llm, instruction_params, response_params, tokenizer=None):
    # user_instructions = [item['instruction'] for item in batch]
    for turn in range(2, args.num_turns+1):
        print(f"Processing turn {turn}...")
        # Generate Instructions
        print(f"Generating instructions for turn {turn}...")
        prompts = []
        for item in batch:
            if not args.tokenizer_template:
                conv = get_conversation_template(MODEL_NAME)
                conv.system_message = mt_system_prompt
                if turn == 2:
                    conv.append_message(conv.roles[0], item[f'instruction'])
                    conv.append_message(conv.roles[1], item[f'response'])
                else:
                    conv.append_message(conv.roles[0], item[f'instruction'])
                    conv.append_message(conv.roles[1], item[f'response'])
                    for i in range(2, turn):
                        conv.append_message(conv.roles[0], item[f'instruction_{i}'])
                        conv.append_message(conv.roles[1], item[f'response_{i}'])
                template = conv.get_prompt() + mt_append_template
            else:
                chat = []
                chat.append({"role": "system", "content": mt_system_prompt})
                if turn == 2:
                    chat.append({"role": "user", "content": item[f'instruction']})
                    chat.append({"role": "assistant", "content": item[f'response']})
                else:
                    chat.append({"role": "user", "content": item[f'instruction']})
                    chat.append({"role": "assistant", "content": item[f'response']})
                    for i in range(2, turn):
                        chat.append({"role": "user", "content": item[f'instruction_{i}']})
                        chat.append({"role": "assistant", "content": item[f'response_{i}']})
                template = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False) + mt_append_template
            prompts.append(template)
        outputs = llm.generate(prompts, instruction_params)
        for i, item in enumerate(batch):
            item[f'instruction_{turn}'] = outputs[i].outputs[0].text.strip()

        # Generate Responses
        print(f"Generating responses for turn {turn}...")
        prompts = []
        for item in batch:
            if not args.tokenizer_template:
                conv = get_conversation_template(MODEL_NAME)
                if turn == 2:
                    conv.append_message(conv.roles[0], item[f'instruction'])
                    conv.append_message(conv.roles[1], item[f'response'])
                else:
                    conv.append_message(conv.roles[0], item[f'instruction'])
                    conv.append_message(conv.roles[1], item[f'response'])
                    for i in range(2, turn):
                        conv.append_message(conv.roles[0], item[f'instruction_{i}'])
                        conv.append_message(conv.roles[1], item[f'response_{i}'])
                conv.append_message(conv.roles[0], item[f'instruction_{turn}'])
                conv.append_message(conv.roles[1], None)
                template = conv.get_prompt() 
            else:
                chat = []
                if turn == 2:
                    chat.append({"role": "user", "content": item[f'instruction']})
                    chat.append({"role": "assistant", "content": item[f'response']})
                else:
                    chat.append({"role": "user", "content": item[f'instruction']})
                    chat.append({"role": "assistant", "content": item[f'response']})
                    for i in range(2, turn):
                        chat.append({"role": "user", "content": item[f'instruction_{i}']})
                        chat.append({"role": "assistant", "content": item[f'response_{i}']})
                chat.append({"role": "user", "content": item[f'instruction_{turn}']})
                template = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            prompts.append(template)
        outputs = llm.generate(prompts, response_params)
        for i, item in enumerate(batch):
            item[f'response_{turn}'] = outputs[i].outputs[0].text.strip()

    return batch

# Generate outputs, update dataset in batches, and overwrite checkpoint
def generate_and_update(dataset, llm=None, instruction_params=None, response_params=None, tokenizer=None):

    # Intialize the dataset with the checkpoint file (if it exists)
    if os.path.exists(CHECKPOINT_FILE):
        last_checkpoint_idx = len(load_dataset_from_file(CHECKPOINT_FILE))
        print(f"Checkpoint file found. Resuming from last checkpoint with index {last_checkpoint_idx}.")
        dataset[:last_checkpoint_idx] = load_dataset_from_file(CHECKPOINT_FILE)
        # Calculate total number of batches
        num_batches = (len(dataset) - last_checkpoint_idx + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"Remaining number of batches: {num_batches}")
    else:
        last_checkpoint_idx = 0
        # Calculate total number of batches
        num_batches = (len(dataset) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Total number of batches: {num_batches}")

    for i in tqdm(range(num_batches)):
        start_idx = i * BATCH_SIZE + last_checkpoint_idx
        end_idx = min((i + 1) * BATCH_SIZE + last_checkpoint_idx, len(dataset))
        batch = dataset[start_idx:end_idx]
        batch = process_batch(batch, llm, instruction_params, response_params, tokenizer)
        
        dataset[start_idx:end_idx] = batch
        # Overwrite the same checkpoint file after serveral batches
        if i % CHECKPOINT_EVERY == 0:
            save_dataset(dataset[:end_idx], CHECKPOINT_FILE)
            print(f"Dataset checkpoint saved after batch {i + 1}.")

    return dataset

# Main function to control workflow
def main():
    # Load instructions from the input file
    dataset = load_dataset_from_file(INPUT_FILE_NAME)
    if args.debug:
        dataset = dataset[:1] # For debugging
    
    # Set the device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    print("Start Local vllm engine...")
    llm =  LLM(model=MODEL_NAME, 
        dtype=args.dtype,
        trust_remote_code=True,
        max_model_len = args.max_model_len, # limited by kv-cache 
        tensor_parallel_size = args.tensor_parallel_size,
        gpu_memory_utilization = args.gpu_memory_utilization)

    instruction_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.instruction_temperature,
        top_p=args.instruction_top_p,
        repetition_penalty=args.instruction_repetition_penalty,
        stop_token_ids=stop_token_ids,
        stop=stop_tokens,
        )
    
    response_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.response_temperature,
        top_p=args.response_top_p,
        repetition_penalty=args.response_repetition_penalty,
        stop_token_ids=stop_token_ids,
        )

    updated_dataset = generate_and_update(dataset, llm, instruction_params, response_params, tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME))

    # Save final dataset
    save_dataset(updated_dataset, SAVED_FILE)

    # Optionally remove the checkpoint file after completion
    os.remove(CHECKPOINT_FILE)
    print("Final dataset saved. Checkpoint removed.")

# Run the main function
if __name__ == "__main__":
    main()