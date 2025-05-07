from prompting_for_rag import get_prompt
from datasets import load_dataset, Dataset
import json
import argparse
from vllm import LLM, SamplingParams
import pandas as pd
import numpy as np
from pathlib import Path
import torch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
import os
import re

example_format = 'TITLE {title} # TEXT {text}'

NUM_DUPS = 5
# NUM_DUPS = 10
def format_row(example):
    prompts = []
    for i in range(NUM_DUPS):
        item = {}
        try:
            item['example'] = example_format.format_map(example['passages']['passage_text'][i])
        except:
            try:
                item['example'] = example_format.format_map(example['passages']['passage_text'][0])
            except:
                item['example'] = example_format.format_map({
                    "title": "<title>",
                    "text": "<text>",
                })


        item['question'] = example['query']
        item['answers'] = example['answers']
        try:
            prompts.append(get_prompt('atm_data_attacker', item))
        except:
            prompts.append("")
    
    return {'prompt': prompts}
    
    
def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--ds_name", default='nq-train', type=str)
    parser.add_argument("--model_name", default='/path/to/input/pretrained_models/Mixtral-8x7B-Instruct-v0.1/', type=str)
    parser.add_argument("--world_size", default=4, type=int)
    parser.add_argument("--max_new_tokens", default=512, type=int)
    parser.add_argument("--dest_dir", required=True, type=str)
    
    
    args = parser.parse_args()
    return args

def is_valid_prompt_list(prompt_list):
    return (
        isinstance(prompt_list, list) and
        len(prompt_list) == NUM_DUPS and
        all(isinstance(p, str) and 0 < len(p.strip()) < 20000 and all(ord(c) < 10000 for c in p)
            for p in prompt_list)
    )

def clean_prompt(prompt):
    """Clean prompt by removing non-printable, non-ASCII, and weird characters, and standardizing whitespace."""
    if not isinstance(prompt, str):
        return ''
    # Remove non-printable characters
    prompt = re.sub(r'[^\x20-\x7E\n\r\t]', '', prompt)
    # Replace multiple whitespace with single space
    prompt = re.sub(r'\s+', ' ', prompt)
    # Strip leading/trailing whitespace
    prompt = prompt.strip()
    return prompt

def call_model_dup(prompts, model, max_new_tokens=50, num_dups=1):
    error_log_path = 'prefix_errors.txt'
    filtered_prompts = []
    error_count = 0
    with open(error_log_path, 'a') as error_log:
        for p in prompts:
            # Clean each prompt in the group
            cleaned = [clean_prompt(x) for x in p]
            if is_valid_prompt_list(cleaned):
                filtered_prompts.append(cleaned)
            else:
                error_count += 1
                error_log.write(f"[INVALID PROMPT GROUP] {json.dumps(cleaned, ensure_ascii=False)}\n")

    print(f"[ℹ️] Filtered out {len(prompts) - len(filtered_prompts)} malformed prompt groups.")
    if error_count > 0:
        print(f"[!] {error_count} prompt groups written to {error_log_path}")

    if not filtered_prompts:
        print("No valid prompts to process.")
        return pd.DataFrame(columns=[f'output_{idx}' for idx in range(num_dups)])

    prompts = np.array(filtered_prompts).reshape((-1, num_dups))
    pdf = pd.DataFrame(prompts, columns=[f'input_{idx}' for idx in range(num_dups)])

    sampling_params = SamplingParams(
        temperature=0.8, top_p=0.95, max_tokens=max_new_tokens)

    odf = pd.DataFrame(columns=[f'output_{idx}' for idx in range(num_dups)])
    for idx in range(num_dups):
        try:
            preds = model.generate(pdf[f'input_{idx}'].tolist(), sampling_params)
            preds = [pred.outputs[0].text for pred in preds]
            odf[f'output_{idx}'] = preds
        except Exception as e:
            # Log the error and the prompts that caused it
            with open(error_log_path, 'a') as error_log:
                error_log.write(f"[MODEL ERROR] input_{idx}: {str(e)}\n")
                for prompt in pdf[f'input_{idx}'].tolist():
                    error_log.write(f"[PROMPT] {prompt}\n")
            print(f"[ERROR] in model.generate for input_{idx}: {e}")
            odf[f'output_{idx}'] = [f"[ERROR] {str(e)}"] * len(pdf)

    odf.to_csv("test_fab.csv", index=False)
    return odf

                                                    
    
if __name__ == '__main__':
    args = parse_args()
    ds_name = args.ds_name
    ds = load_dataset('json', data_files=f'datasets/{ds_name}.jsonl', split='train')
    
    # Limit to 10,000 examples if dataset is larger
    if len(ds) > 10000:
        ds = ds.shuffle(seed=42).select(range(10000))
    
    ds = ds.map(format_row, num_proc=8, remove_columns=ds.column_names)
    
    model = LLM(model=args.model_name, tensor_parallel_size=args.world_size, trust_remote_code=True, swap_space=4)

    preds = call_model_dup(ds['prompt'], model, max_new_tokens=args.max_new_tokens, num_dups=NUM_DUPS)

    '''dest_dir = Path(args.dest_dir)
    if not dest_dir.exists():
        dest_dir.mkdir()'''

    model_name = Path(args.model_name).name

    file_path = f'{ds_name}_fab.csv'

    # Save to CSV using the file path string
    preds.to_csv(file_path, index=False)
    #absolute_dest_dir = os.path.abspath(args.dest_dir)
    #print(f"Saving to: {absolute_dest_dir}")

    # Save to CSV using the file path string
    #preds.to_csv((args.dest_dir / f'{ds_name}_fab.csv').resolve(), index=False)

