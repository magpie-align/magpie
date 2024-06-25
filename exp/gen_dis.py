from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import faiss
import argparse
import json
from tqdm import tqdm
from utils import load_dataset_from_file

################
# Configurations
################
def get_args():
    # Experiment Settings
    parser = argparse.ArgumentParser(description="Similarity Calculation Manager.")
    parser.add_argument("--sentence_model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--input_file", type=str, default=None, help="Input dataset file name")
    parser.add_argument("--encoding_batch_size", type=int, default=65536, help="Batch size for encoding the sentences.")
    parser.add_argument("--distance_distance_threshold", type=float, default=0.05, help="distance_threshold for the similarity search.")
    parser.add_argument("--search_space_size", type=int, default=500, help="Number of examples to search for similarity.")
    parser.add_argument("--search_batch_size", type=int, default=1024, help="Batch size for searching for similarity.")

    # System Settings
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--save_faiss_index", type=bool, default=True, help="Save the Faiss index.")
    
    return parser.parse_args()

args = get_args()

sentence_model = args.sentence_model
dataset_path = args.input_file
dataset_name = dataset_path[dataset_path.rfind('/')+1:dataset_path.rfind('.')]
output_file = f"../data/{dataset_name}_distance.jsonl"

################
# Step 1 - Load the Dataset and Build the Faiss Index
################
# Load the dataset
dataset = load_dataset("json", data_files=dataset_path)
print(dataset)
inputs = dataset["train"]["instruction"]
print(f"The second instruction in the dataset is: {inputs[1]}")

model = SentenceTransformer(sentence_model)
model.to(device=f'cuda:{args.device}', dtype=torch.float32)
print(f"The model is loaded on device: {model.device}")

# Encode the sentences in the dataset into vectors
encoding_batch_size = args.encoding_batch_size  # Adjust the batch size based on available memory
embeddings = []
for i in range(0, len(inputs), encoding_batch_size):
    batch_sentences = inputs[i:i+encoding_batch_size]
    batch_embeddings = model.encode(batch_sentences, convert_to_tensor=True, show_progress_bar=True)
    embeddings.append(batch_embeddings.cpu().numpy())

# Concatenate the embeddings
embeddings = np.concatenate(embeddings, axis=0)

# Print out the shape of the concatenated embeddings to verify the results
print(f"The shape of the concatenated embeddings is: {embeddings.shape}")

# Add the encoded vectors to the dataset
print("Adding the embeddings to the dataset...")
dataset["train"] = dataset["train"].add_column("embeddings", embeddings.tolist())

# Build the Faiss index on the dataset
print("Building the Faiss index...")
dataset["train"].add_faiss_index(column="embeddings")

# Save the Faiss index
if args.save_faiss_index:
    print("Saving the Faiss index...")
    index = dataset["train"].get_index("embeddings")
    faiss_index = index.faiss_index
    index_file = f"../data/{dataset_name}.faiss"
    faiss.write_index(faiss_index, index_file)

################
# Step 2 - Find Similar Examples
################
distance_threshold = args.distance_distance_threshold
search_space_size = args.search_space_size
search_batch_size = args.search_batch_size
n_batches = (len(dataset["train"]) + search_batch_size - 1) // search_batch_size
print(f"Number of batches: {n_batches}")

# load the dataset in jsonl format
unfilled_dataset = load_dataset_from_file(dataset_path)

with open(output_file, 'a') as file:
    for batch_idx in tqdm(range(n_batches)):
        start_idx = batch_idx * search_batch_size
        end_idx = min((batch_idx + 1) * search_batch_size, len(dataset["train"]))

        batch_indices = list(range(start_idx, end_idx))
        
        # Obtain the embeddings for the current batch
        batch_embeddings = embeddings[batch_indices]
        
        # Search for the most similar examples
        search_results = dataset["train"].search_batch(queries=batch_embeddings, k=search_space_size, index_name="embeddings")
        total_scores = search_results.total_scores
        total_indices = search_results.total_indices

        for i in range(len(total_indices)):
            scores = total_scores[i]
            indices = total_indices[i]
            min_distance = float(scores[1]) # should exclude itself
            dataset["train"][start_idx + i]["min_distance"] = min_distance

            filtered_indices = [index for index, score in zip(indices, scores) if score < distance_threshold]
            # Should remove itself
            filtered_indices = [index for index in filtered_indices if index != start_idx + i]

            if len(filtered_indices) == 0:
                repeat_count = 0
                min_similar_conversation_id = None

                dataset["train"][start_idx + i]["repeat_count"] = repeat_count
                dataset["train"][start_idx + i]["min_similar_conversation_id"] = min_similar_conversation_id
            else:
                min_similar_conversation_idx = int(min(filtered_indices))
                if min_similar_conversation_idx >= start_idx + i:
                    min_similar_conversation_id = dataset["train"][start_idx + i]["conversation_id"]
                else: 
                    min_similar_conversation_id = dataset["train"][min_similar_conversation_idx]["conversation_id"]
                
                repeat_count = len(filtered_indices)

                dataset["train"][start_idx + i]["repeat_count"] = repeat_count
                dataset["train"][start_idx + i]["min_similar_conversation_id"] = min_similar_conversation_id

            # save the updated dataset
            line = unfilled_dataset[start_idx + i]
            line["min_neighbor_distance"] = min_distance
            line["repeat_count"] = repeat_count
            line["min_similar_conversation_id"] = min_similar_conversation_id
            file.write(json.dumps(line) + '\n')
            
        print(f"Batch {batch_idx} is saved to the output file")

print("Distance calculation is completed.")