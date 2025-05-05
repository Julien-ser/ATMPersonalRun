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
from tqdm import tqdm

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
        prompts.append(get_prompt('atm_data_attacker', item))
    
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

from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from pathlib import Path

def call_model_dup(prompts, model, max_new_tokens=50, num_dups=1, checkpoint_interval=1000, existing_rows=0):
    prompts = np.array(prompts)
    prompts = prompts.reshape((-1, num_dups))
    pdf = pd.DataFrame(prompts, columns=[f'input_{idx}' for idx in range(num_dups)])

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=max_new_tokens)

    # Set up output DataFrame
    odf = pd.DataFrame(columns=[f'output_{idx}' for idx in range(num_dups)])

    total_rows = len(prompts)

    # File path for saving predictions
    file_path = f"{ds_name}_fab.csv"

    # Create or append to CSV with the header check
    file_exists = Path(file_path).exists()

    with tqdm(total=total_rows, desc="Generating and saving predictions", unit="prompt") as pbar:
        for i in range(total_rows):
            item = {f'input_{idx}': pdf.iloc[i, idx] for idx in range(num_dups)}

            # Generate prediction for each prompt
            preds = model.generate([item[f'input_{idx}'] for idx in range(num_dups)], sampling_params)
            outputs = [pred.outputs[0].text for pred in preds]

            # Add predictions to the output dataframe
            for idx in range(num_dups):
                odf.at[i, f'output_{idx}'] = outputs[idx]

            # Every `checkpoint_interval` instances, save the current predictions
            if (i + 1) % checkpoint_interval == 0 or (i + 1) == total_rows:
                chunk = odf.iloc[i - checkpoint_interval + 1:i + 1]  # Extract chunk of 1000 predictions
                chunk.to_csv(file_path, mode='a', header=not file_exists, index=False)
                file_exists = True  # After the first write, the file exists, so headers won't be written again

            pbar.update(1)  # Update the progress bar by 1 per processed prompt

    return odf

def get_existing_rows(file_path):
    # Check if the file exists
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return len(df)  # Return the number of rows already saved
    else:
        return 0  # If the file doesn't exist, return 0                         
    
if __name__ == '__main__':
    args = parse_args()
    ds_name = args.ds_name
    ds = load_dataset('json', data_files=f'datasets/{ds_name}.jsonl', split='train')
    
    ds = ds.map(format_row, num_proc=8, remove_columns=ds.column_names)
    
    model = LLM(model=args.model_name, tensor_parallel_size=args.world_size, trust_remote_code=True, swap_space=4)

    # Get the number of rows already saved in the CSV (for resuming)
    existing_rows = get_existing_rows(f"{ds_name}_fab.csv")

    # Call model and get predictions, starting from the last saved row
    preds = call_model_dup(ds['prompt'], model, max_new_tokens=args.max_new_tokens, num_dups=NUM_DUPS, existing_rows=existing_rows)

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

