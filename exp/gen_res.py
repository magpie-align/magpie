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
from utils import load_dataset_from_file, save_dataset, make_api_request_with_retry, get_conversation_template
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM

################
# Configurations
################
def get_args():
    # Experiment Settings
    parser = argparse.ArgumentParser(description="Response Generation Manager.")
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="We will support more models in the future.")
    parser.add_argument("--input_file", type=str, default=None, help="Input dataset file name")
    parser.add_argument("--batch_size", type=int, default=128, help="Number of samples per batch")
    parser.add_argument("--checkpoint_every", type=int, default=20, help="Save checkpoint every n batches")
    parser.add_argument("--api_url", type=str, default="https://api.together.xyz/v1/chat/completions", help="API URL")
    parser.add_argument("--api_key", type=str, default=None, help="Together API Key")
    parser.add_argument("--offline", action="store_true", help="Use local engine")

    # Generation Parameters
    parser.add_argument('--engine', default="vllm", type=str, choices=["vllm", "hf", "together"])
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs to use for tensor parallelism. Only used for Llama 70B models.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--tokenizer_template", type=bool, default=False, help="Use tokenizer template for generating the response.")
    parser.add_argument("--use_tokenizer_template", action="store_true", dest="tokenizer_template")

    return parser.parse_args()

args = get_args()
print(f"Response Generation Manager. Arguments: {args}") # For logging

if args.input_file is None:
    raise ValueError("Please specify the input file path.")

# Constants for the local vllm engine
MODEL_NAME = args.model_path
INPUT_FILE_NAME = args.input_file 
BATCH_SIZE = args.batch_size
CHECKPOINT_FILE = f"{INPUT_FILE_NAME[:INPUT_FILE_NAME.rfind('.')]}_res_checkpoint.json"
CHECKPOINT_EVERY = args.checkpoint_every
SAVED_FILE = f"{INPUT_FILE_NAME[:INPUT_FILE_NAME.rfind('.')]}_res.json"

# Obtain config from configs/model_configs.json
with open("../configs/model_configs.json", "r") as f:
    model_configs = json.load(f)
    model_config = model_configs[args.model_path]
    stop_tokens = model_config["stop_tokens"]
    stop_token_ids = model_config["stop_token_ids"]

# API Setups
if args.engine == "together":
    # Change name for API (Together Naming Convention)
    if MODEL_NAME == "meta-llama/Meta-Llama-3-8B-Instruct":
        api_model_name = "meta-llama/Llama-3-8b-chat-hf"
    elif MODEL_NAME == "meta-llama/Meta-Llama-3-70B-Instruct":
        api_model_name = "meta-llama/Llama-3-70b-chat-hf"
    else:
        api_model_name = MODEL_NAME

    # Constants for the API
    API_ENDPOINT = args.api_url
    API_HEADERS = {
        "Authorization": args.api_key,
    }
    API_PARAMS = {
        "model": api_model_name,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "stop": stop_tokens
    }

# Process a batch of data using the API
def process_batch_with_api(batch):
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_item = {
            executor.submit(
                make_api_request_with_retry, 
                [{'content': item['instruction'], 'role': 'user'}],
                API_PARAMS,
                API_ENDPOINT,
                API_HEADERS,
            ): item 
            for item in batch
        }

        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            try:
                api_response = future.result()
                item['response'] = api_response.strip()
                item['gen_response_configs'] = {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "repetition_penalty": args.repetition_penalty,
                    "max_tokens": args.max_tokens,
                    "stop_tokens": stop_tokens,
                    "output_generator": MODEL_NAME,
                    "engine": api_model_name,
                }
            except Exception as e:
                print(f"Failed to process item: {item} with error: {str(e)}")
                item['response'] = ""
                
    return batch

# Process a batch of data using local vllm engine
def process_batch(batch, llm, params, tokenizer=None):
    user_instructions = [item['instruction'] for item in batch]
    prompts = []
    for instruction in user_instructions:
        if not args.tokenizer_template:
            conv = get_conversation_template(MODEL_NAME)
            conv.append_message(conv.roles[0], instruction)
            conv.append_message(conv.roles[1], None)
            template = conv.get_prompt()
        else:
            chat = [{"role": "user", "content": instruction}]
            template = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompts.append(template)
    if args.engine == "vllm":
        outputs = llm.generate(prompts, params)
    elif args.engine == "hf":
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(torch.cuda.current_device())
        gen_do_sample = False if args.temperature == 0 else True
        outputs = llm.generate(**inputs,
                tokenizer=tokenizer, 
                do_sample=gen_do_sample, 
                temperature=args.temperature if gen_do_sample else None, # To avoid temperature` (=0) has to be a strictly positive float
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty, 
                max_length=args.max_tokens,
                )
        outputs = tokenizer.batch_decode(outputs[i][len(inputs[i]):] for i in range(len(outputs)))
        # Setting stop tokens seems not working for Gemma, so we manually truncate the outputs
        for i, completion in enumerate(outputs):
            for stop_token in stop_tokens:
                if stop_token in completion:
                    outputs[i] = completion[:completion.index(stop_token)]

    for i, item in enumerate(batch):
        if args.engine == "vllm":
            item['response'] = outputs[i].outputs[0].text.strip()
        elif args.engine == "hf":
            item['response'] = outputs[i].strip()
        item['gen_response_configs'] = {
            "prompt": prompts[i],
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "max_tokens": args.max_tokens,
            "stop_tokens": stop_tokens,
            "output_generator": MODEL_NAME,
            "engine": args.engine,
        }
    return batch

# Generate outputs, update dataset in batches, and overwrite checkpoint
def generate_and_update(dataset, llm=None, params=None, tokenizer=None):
    # Initialize tokenizer
    if tokenizer is not None:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if "gemma-2" in args.model_path.lower():
            tokenizer.padding_side = "right"

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
        if args.engine == "together":
            batch = process_batch_with_api(batch)
        else:
            batch = process_batch(batch, llm, params, tokenizer)
        
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
    
    if args.engine == "together":
        print("Start together API engine...")
        llm = None
        params = None
        tokenizer = None
    elif args.engine == "vllm":
        # Set the device
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        print("Start Local vllm engine...")
        llm = LLM(model=MODEL_NAME, 
            dtype=args.dtype,
            trust_remote_code=True,
            max_model_len = args.max_model_len, # limited by kv-cache 
            tensor_parallel_size = args.tensor_parallel_size,
            gpu_memory_utilization = args.gpu_memory_utilization)
    
        params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stop_token_ids=stop_token_ids,
            )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    elif args.engine == "hf":
        print("Start Hugging Face engine...")
        params = None
        # Load the model and tokenizer
        llm = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map={'':torch.cuda.current_device()},
            torch_dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    else:
        raise ValueError("Invalid engine type.")

    updated_dataset = generate_and_update(dataset, llm, params, tokenizer=tokenizer)

    # Save final dataset
    save_dataset(updated_dataset, SAVED_FILE)

    # Optionally remove the checkpoint file after completion
    os.remove(CHECKPOINT_FILE)
    print("Final dataset saved. Checkpoint removed.")

# Run the main function
if __name__ == "__main__":
    main()