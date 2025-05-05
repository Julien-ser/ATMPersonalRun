from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import sys
sys.path.append("/home/julien/contriever")  # Adjust the path as necessary
from src.contriever import Contriever
from transformers import AutoTokenizer
from tqdm import tqdm

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


ds = load_dataset("mandarjoshi/trivia_qa", "rc")

import json
from pathlib import Path

output_list = []
max_examples = 5  # Number of examples you want to save
#range(len(ds['train']))
for i in tqdm(range(len(ds['train'])), desc="Processing examples", unit="example"):
    try:
        example = ds["train"][i]

        answers = example["answer"]

        question = example["question"]

        qid = example["question_id"]

        aliases = list(set(answers["aliases"] + answers["normalized_aliases"]))
        
        titles = example['search_results']['title']

        text = example['search_results']['search_context']

        contexts = []

        for j in range(0, len(titles)):
            score = contriever_score(question, text[j]) + 1
            val = {"hasanswer":True, "id":f"{qid}_{j}","score":str(score),"text":text[j], "title":titles[j]}
            contexts.append(val)
        # Build final output example in the required format
        output_example = {
            "query": example["question"],
            "answers": aliases,
            "ctxs": contexts
        }

        output_list.append(json.dumps(output_example))

    except KeyError as e:
        print(f"Missing key at index {i}: {e}")
    except TypeError as e:
        print(f"Type error at index {i}: {e}")
    except Exception as e:
        print(f"An error occurred at index {i}: {e}")

# Write to .jsonl file
output_path = Path("triviaqa.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    for line in output_list:
        f.write(line + '\n')

print(f"Successfully created {output_path} with {len(output_list)} examples.")
