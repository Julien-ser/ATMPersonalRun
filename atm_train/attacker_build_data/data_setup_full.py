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
torch.cuda.empty_cache()
# Initialize Contriever model and tokenizer
contriever = Contriever.from_pretrained("facebook/contriever") 
tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")  # Load the associated tokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModel.from_pretrained("facebook/contriever").to(device)

# Function to compute Contriever score
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

# Load the dataset
ds = load_dataset("mandarjoshi/trivia_qa", "rc")

output_path = Path("triviaqa.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)

# Check how many examples have already been processed
if output_path.exists():
    with open(output_path, "r") as f:
        processed = sum(1 for _ in f)
else:
    processed = 0

print(f"Resuming from index {processed}")

output_list = []
save_every = 1000  # Save every 1000 examples
# Process the dataset
for i in tqdm(range(processed, len(ds['train'])), desc="Processing examples", unit="example"):
    try:
        example = ds["train"][i]

        answers = example["answer"]
        question = example["question"]
        qid = example["question_id"]
        aliases = list(set(answers["aliases"] + answers["normalized_aliases"]))
        titles = example['search_results']['title']
        text = example['search_results']['search_context']

        contexts = []
        for j in range(len(titles)):
            score = contriever_score(question, text[j]) + 1
            val = {
                "hasanswer": True,
                "id": f"{qid}_{j}",
                "score": str(score),
                "text": text[j],
                "title": titles[j]
            }
            contexts.append(val)

        output_example = {
            "query": question,
            "answers": aliases,
            "ctxs": contexts
        }

        output_list.append(json.dumps(output_example))

        # Periodically write to file
        if len(output_list) >= save_every:
            with open(output_path, "a") as f:
                for line in output_list:
                    f.write(line + '\n')
            print(f"Saved {len(output_list)} examples to {output_path} (up to index {i})")
            output_list = []

    except KeyError as e:
        print(f"Missing key at index {i}: {e}")
    except TypeError as e:
        print(f"Type error at index {i}: {e}")
    except Exception as e:
        print(f"An error occurred at index {i}: {e}")

# Write any remaining lines
if output_list:
    with open(output_path, "a") as f:
        for line in output_list:
            f.write(line + '\n')
    print(f"Saved remaining {len(output_list)} examples to {output_path}")

print(f"Finished processing. Total examples in {output_path}: {processed + len(output_list)}")
