import os
import json
import warnings
import concurrent.futures
from tqdm import tqdm
import argparse
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, pipeline
from utils import load_jsonl_to_list, load_dataset_from_file, save_dataset
from utils import make_api_request_with_retry
from str_utils import input_difficulty_rating, input_classification, input_quality_rating
from utils import get_conversation_template
from lingua import Language, LanguageDetectorBuilder

################
# Configurations
################
def get_args():
    # Experiment Settings
    parser = argparse.ArgumentParser(description="Unified Tagging Manager.")
    parser.add_argument("--tag_mission", type=str, default="quality", help="The tagging mission.", choices=["difficulty", "quality", "classification", "safety", "reward", "language"])
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Tag Model.")
    parser.add_argument("--guard_model_path", type=str, default="meta-llama/Meta-Llama-Guard-2-8B", help="Guard Model.")
    parser.add_argument("--reward_model_path", type=str, default="sfairXC/FsfairX-LLaMA3-RM-v0.1", help="Reward Model.")
    parser.add_argument("--input_file", type=str, default=None, help="Input dataset file name")
    parser.add_argument("--batch_size", type=int, default=1000, help="Number of samples per batch. Online <100, Offline <200.")
    parser.add_argument("--checkpoint_every", type=int, default=40, help="Save checkpoint every n batches")
    parser.add_argument("--api", type=bool, default=False, help="Use API to generate responses")
    parser.add_argument("--offline", action="store_false", dest="api", help="Use local vllm engine")
    parser.add_argument("--online", action="store_true", dest="api", help="Use Together API engine")
    parser.add_argument("--api_url", type=str, default="https://api.together.xyz/v1/chat/completions", help="API URL")
    parser.add_argument("--api_key", type=str, default=None, help="Together API Key")
    parser.add_argument("--debug", type=bool, default=False, help="Debug mode. Only process the first 100 samples.")
    parser.add_argument("--save_as", type=str, default="json", choices=["json", "jsonl"], help="Save the generated responses as a what kind of file")

    # vllm Configs
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--quantization", type=str, default="fp8", choices=["fp8", "awq", "gptq", None])
    parser.add_argument("--kv_cache_dtype", type=str, default="auto", choices=["auto", "fp8"])
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    
    # Tagging Generation Configs
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    return parser.parse_args()

args = get_args()
print(f"[unitag.py] Unified Tagging Manager. Arguments: {args}") # For logging

MODEL_NAME = args.model_path
checkpoint_every = args.checkpoint_every if args.tag_mission != "reward" else args.checkpoint_every*100
batch_size = args.batch_size
mission = args.tag_mission

# Change name for API (Together Naming)
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
    "repetition_penalty": args.repetition_penalty,
    "stop": ["}"]
}

def template_generator(input, mission):
    if mission == "difficulty":
        return input_difficulty_rating(input)
    elif mission == "quality":
        return input_quality_rating(input)
    elif mission == "classification":
        return input_classification(input)
    else:
        raise ValueError("Invalid mission. Available missions: difficulty, quality, classification")

# Process item
def process_engine_responses(response, item, mission):
    try:
        tags_json = json.loads(response)
        if mission == "difficulty":
            item['intent'] = tags_json['intent']
            item['knowledge'] = tags_json['knowledge']
            item['difficulty'] = tags_json['difficulty']
            item['difficulty_generator'] = MODEL_NAME
        elif mission == "quality":
            item['input_quality'] = tags_json["input_quality"]
            item['quality_explanation'] = tags_json["explanation"]
            item['quality_generator'] = MODEL_NAME
        elif mission == "classification":
            item['task_category'] = tags_json['primary_tag']
            item['other_task_category'] = tags_json['other_tags']
            item['task_category_generator'] = MODEL_NAME
    except Exception as e:
        print(f"[unitag.py] Failed to process item with error: {str(e)}")
        print(f"[unitag.py] Raw response from LLM tagger: {response}")
        if mission == "difficulty":
            item['intent'] = None
            item['knowledge'] = None
            item['difficulty'] = None
            item['difficulty_generator'] = None
        elif mission == "quality":
            item['input_quality'] = None
            item['quality_explanation'] = None
            item['quality_generator'] = None
        elif mission == "classification":
            item['task_category'] = None
            item['other_task_category'] = None
            item['task_category_generator'] = None

    return item


# Process a batch of data using the API
def process_batch_with_api(batch, mission):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_item = {
            executor.submit(
                make_api_request_with_retry,
                [
                    {'role': 'user', 'content': template_generator(item['instruction'], mission)},
                    {'role': 'assistant', 'content': "{"}
                ],
                API_PARAMS,
                API_ENDPOINT,
                API_HEADERS
            ): item
            for item in batch
        }
        
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            api_response = future.result()
            # Append prefilling prefix
            api_response = "{" + api_response + "}"
            item = process_engine_responses(api_response, item, mission)
                
    return batch

# Process a batch of data
def process_batch(batch, llm, params, mission, tokenizer=None):
    prompts = []
    for i, item in enumerate(batch):
        if mission == "safety":
            chat = [
                {"role": "user", "content": item['instruction']},
                {"role": "assistant", "content": item['response']},
            ]
            template = tokenizer.apply_chat_template(chat, tokenize=False)
        else:
            conv = get_conversation_template(MODEL_NAME)
            conv.append_message(conv.roles[0], template_generator(item['instruction'], mission))
            conv.append_message(conv.roles[1], None)
            template = conv.get_prompt()
            template += "{"  # Prefilling for better generation
        prompts.append(template)

    outputs = llm.generate(prompts, params)

    for i, item in enumerate(batch):
        if mission == "safety":
            model_response = outputs[i].outputs[0].text.strip()
            item['llama_guard_2'] = model_response
        else:
            model_response = "{\n" + outputs[i].outputs[0].text.strip()
            # Remove additional information at the end of the response
            model_response = model_response[:model_response.rfind("}")+1]

            item = process_engine_responses(model_response, item, mission)
    
    return batch

def process_batch_with_reward_model(batch, rm_pipe, rm_pipe_kwargs):
    prompts = []
    for i, item in enumerate(batch):
        input = item['instruction']
        output = item['response']
        chat = [
            {"role": "user", "content": f"{input}"},
            {"role": "assistant", "content": f"{output}"},
        ]
        template = rm_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False).replace(rm_tokenizer.bos_token, "")
        prompts.append(template)
    
    outputs = rm_pipe(prompts, **rm_pipe_kwargs)
    scores = [output[0]["score"] for output in outputs]

    for i, item in enumerate(batch):
        try:
            item['instruct_reward'] = scores[i]
            item['reward_model'] = args.reward_model_path
        except Exception as e:
            print(f"Failed to process item: {item} with error: {str(e)}")

            item['instruct_reward'] = None
            item['reward_model'] = args.reward_model_path
    return batch

# Generate outputs, update dataset in batches, and overwrite checkpoint
def generate_and_update(dataset, mission, llm, params, api, rm_pipe, rm_pipe_kwargs, batch_size, checkpoint_file, checkpoint_every = 20):
    if os.path.exists(checkpoint_file):
        last_checkpoint_idx = len(load_dataset_from_file(checkpoint_file))
        print(f"[unitag.py] Checkpoint file found. Resuming from last checkpoint with index {last_checkpoint_idx}.")
        dataset[:last_checkpoint_idx] = load_dataset_from_file(checkpoint_file)
        num_batches = (len(dataset) - last_checkpoint_idx + batch_size - 1) // batch_size
        print(f"[unitag.py] Remaining number of batches: {num_batches}")
    else:
        last_checkpoint_idx = 0
        num_batches = (len(dataset) + batch_size - 1) // batch_size  # Calculate total number of batches
        print(f"[unitag.py] Total number of batches: {num_batches}")

    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size + last_checkpoint_idx
        end_idx = min((i + 1) * batch_size + last_checkpoint_idx, len(dataset))
        batch = dataset[start_idx:end_idx]

        if api:
            batch = process_batch_with_api(batch, mission)
        elif mission == "reward":
            batch = process_batch_with_reward_model(batch, rm_pipe, rm_pipe_kwargs)
        elif mission == "safety":
            tokenizer = AutoTokenizer.from_pretrained(args.guard_model_path)
            batch = process_batch(batch, llm, params, mission, tokenizer)
        else:
            batch = process_batch(batch, llm, params, mission)

        dataset[start_idx:end_idx] = batch
        # Overwrite the same checkpoint file every checkpoint_every batches
        if (i + 1) % checkpoint_every == 0:
            save_dataset(dataset[:end_idx], checkpoint_file)
            print(f"[unitag.py] Dataset checkpoint saved after batch {i + 1}.")

    return dataset


if __name__ == "__main__":
    input_file = args.input_file
    # Mission Settings
    if mission == "difficulty":
        output_file = f"{input_file[:input_file.rfind('.')]}_difficulty.jsonl"
        checkpoint_file = f"{input_file[:input_file.rfind('.')]}_difficulty_checkpoint.json"
    elif mission == "quality":
        output_file = f"{input_file[:input_file.rfind('.')]}_quality.jsonl"
        checkpoint_file = f"{input_file[:input_file.rfind('.')]}_quality_checkpoint.json"
    elif mission == "classification":
        output_file = f"{input_file[:input_file.rfind('.')]}_category.jsonl"
        checkpoint_file = f"{input_file[:input_file.rfind('.')]}_category_checkpoint.json"
    elif mission == "safety":
        output_file = f"{input_file[:input_file.rfind('.')]}_safety.jsonl"
        checkpoint_file = f"{input_file[:input_file.rfind('.')]}_safety_checkpoint.json"
    elif mission == "reward":
        rm_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path)
        output_file = f"{input_file[:input_file.rfind('.')]}_reward.jsonl"
        checkpoint_file = f"{input_file[:input_file.rfind('.')]}_reward_checkpoint.json"
    elif mission == "language":
        output_file = f"{input_file[:input_file.rfind('.')]}_language.jsonl"
        checkpoint_file = f"{input_file[:input_file.rfind('.')]}_language_checkpoint.json"
    else:
        raise ValueError("Invalid mission. Available missions: difficulty, quality, classification, safety, reward, language")
    # Change jsonl to json if args.save_as is json
    if args.save_as == "json":
        output_file = f"{output_file[:output_file.rfind('.')]}.json"

    # Load dataset
    if not args.debug:
        dataset = load_dataset_from_file(input_file)
    else:
        warnings.warn("Debug mode enabled. Only processing the first 100 samples.")
        dataset = load_dataset_from_file(input_file)[:100]
        # output_file = f"{output_file[:output_file.rfind('.')]}_debug.jsonl"
        # checkpoint_file = f"{output_file[:output_file.rfind('.')]}_debug_checkpoint.json"

    if mission != "language":
        if args.api:
            if args.tag_mission not in ["difficulty", "quality", "classification"]:
                raise ValueError("Safety and reward mission is not supported by API.")

            print("[unitag.py] Start together API engine...")
            llm = None
            params = None
            rm_pipe = None
            rm_pipe_kwargs = None
        else:
            if args.tag_mission == "reward":
                # Currently vllm do not support reward model inference, use transformer pipeline instead
                rm_pipe = pipeline(
                    "sentiment-analysis",
                    model=args.reward_model_path,
                    device=int(args.device),
                    tokenizer=rm_tokenizer,
                    model_kwargs={"torch_dtype": torch.float16}
                )
                rm_pipe_kwargs = {
                    "return_all_scores": True,
                    "function_to_apply": "none",
                    "batch_size": batch_size,
                }
                llm = None
                params = None
            else:
                print("[unitag.py] Start Local vllm engine...")
                os.environ["CUDA_VISIBLE_DEVICES"] = args.device

                llm = LLM(model=MODEL_NAME if not args.tag_mission == "safety" else args.guard_model_path,
                            dtype=args.dtype,
                            quantization=args.quantization,
                            kv_cache_dtype=args.kv_cache_dtype,
                            max_model_len=args.max_model_len,
                            tensor_parallel_size=args.tensor_parallel_size,
                            gpu_memory_utilization=args.gpu_memory_utilization,
                            trust_remote_code=True,
                            enable_prefix_caching=True,)
                
                params = SamplingParams(
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                            repetition_penalty=args.repetition_penalty,
                            stop=["}"],
                            include_stop_str_in_output=True,
                            )
                rm_pipe = None
                rm_pipe_kwargs = None

        updated_dataset = generate_and_update(dataset, mission, llm, params, args.api, rm_pipe, rm_pipe_kwargs, batch_size, checkpoint_file, checkpoint_every)
    
    else:
        # Language Detection using lingua
        print("[unitag.py] Start language detection engine...")
        detector = LanguageDetectorBuilder.from_all_languages().build()
        for item in tqdm(dataset):
            if item['instruction'] != "":
                try:
                    item['language'] = detector.detect_language_of(item['instruction']).iso_code_639_1.name
                except Exception as e:
                    print(f"Failed to process item with error: {str(e)}")
                    item['language'] = None
            else:
                item['language'] = None
        updated_dataset = dataset

    if args.save_as == "json":
        save_dataset(updated_dataset, output_file, convert_to_jsonl=False)
    else:
        save_dataset(updated_dataset, output_file, convert_to_jsonl=True)

    # Remove the checkpoint file after completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("[unitag.py] Final dataset saved. Checkpoint removed.")