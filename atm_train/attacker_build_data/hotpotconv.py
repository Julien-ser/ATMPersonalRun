from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import sys
sys.path.append("/home/julien/contriever")  # Adjust the path as necessary
from src.contriever import Contriever
from transformers import AutoTokenizer
from tqdm import tqdm
import json
from pathlib import Path

contriever = Contriever.from_pretrained("facebook/contriever") 
tokenizer = AutoTokenizer.from_pretrained("facebook/contriever") #Load the associated tokenizer:

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = AutoModel.from_pretrained("facebook/contriever").to(device)

def contriever_score(query, context):
    # Tokenize the input sentences
    inputs = tokenizer([query, context], padding=True, truncation=True, return_tensors="pt").to(device)
    
    embeddings = model(**inputs).last_hidden_state  # Shape: [batch_size, seq_length, hidden_dim]
    
    # Extract the [CLS] token embeddings for both query and context (first token in each sequence)
    query_embedding = embeddings[0, 0]  # [CLS] token for the query sentence
    context_embedding = embeddings[1, 0]  # [CLS] token for the context fragment/passage
    
    # Compute the dot product similarity
    score = torch.dot(query_embedding, context_embedding)
    return score.item()

# Load the input JSONL file
input_file = Path("datasets/hotpot_train_v1.1.jsonl")  # Path to your JSONL input file

output_list = []

# Get total number of lines for tqdm
with open(input_file, 'r') as f:
    total_lines = sum(1 for _ in f)

# Read and process the JSONL file
with open(input_file, 'r') as f:
    for i, line in tqdm(enumerate(f), desc="Processing examples", unit="example", total=500):
        if i >= 500:
            break  # Stop after processing 500 lines
        try:
            example = json.loads(line.strip())
            answer = example["answer"]
            question = example["question"]
            qid = example["_id"]
            supporting_facts = example["supporting_facts"]
            #aliases = list(set(answers["aliases"] + answers["normalized_aliases"]))
            titles = [example['context'][i][0] for i in range(len(example['context']))]
            text = ["".join(example['context'][i][1]) for i in range(len(example['context']))]
            contexts = []
            for j in range(0, len(titles)):
                score = contriever_score(question, text[j]) + 1  # Adding 1 to the score for adjustment
                if titles[j] in supporting_facts:
                    score += 0.75
                val = {
                    "hasanswer": True,
                    "id": f"{qid}_{j}",
                    "score": str(score),
                    "text": text[j],
                    "title": titles[j]
                }
                contexts.append(val)
            # Build the final output example in the required format
            output_example = {
                "query": example["question"],
                "answers": [answer],
                "ctxs": contexts
            }
            output_list.append(json.dumps(output_example))
        except KeyError as e:
            print(f"Missing key at index {i}: {e}")
        except TypeError as e:
            print(f"Type error at index {i}: {e}")
        except Exception as e:
            print(f"An error occurred at index {i}: {e}")

# Write the processed examples to a new JSONL file
output_path = Path("datasets/hotpot_style.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    for line in output_list:
        f.write(line + '\n')

print(f"Successfully created {output_path} with {len(output_list)} examples.")