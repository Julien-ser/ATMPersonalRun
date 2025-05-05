'''import json

def show_jsonl_structure(file_path):
    structure = []  # To store the structure information
    
    try:
        with open(file_path, 'r') as file:
            for line_number, line in enumerate(file, 1):
                try:
                    # Load the JSON object from the line
                    json_obj = json.loads(line.strip())
                    
                    # Create a dictionary to store the structure of the current line
                    line_structure = {}
                    for key, value in json_obj.items():
                        line_structure[key] = type(value).__name__
                    
                    # Add the structure of the current line to the list
                    structure.append(line_structure)
                
                except json.JSONDecodeError:
                    print(f"Line {line_number}: Failed to decode JSON")
    
        # Convert the structure to JSON and print
        print(json.dumps(structure, indent=4))
        
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Provide the path to your .jsonl file
    jsonl_file_path = 'triviaqa_small_fab.jsonl'
    
    # Show the structure of the .jsonl file
    show_jsonl_structure(jsonl_file_path)
'''

import json

# Load dpo_data (prompt + chosen)
dpo_entries = []
with open("_dpo.jsonl", "r") as f:
    for line in f:
        entry = json.loads(line)
        dpo_entries.append({
            "prompt": entry["prompt"],
            "adv_prompt": entry["chosen"]  # Treat 'chosen' as adversarial prompt
        })

# Load query_data (answers)
query_answers = []
with open("triviaqa_small_fab.jsonl", "r") as f:
    for line in f:
        entry = json.loads(line)
        if entry["answers"]:  # Ensure answers exist
            query_answers.append(entry["answers"][0])  # Take first answer

assert len(dpo_entries) == len(query_answers), "Datasets must have the same length!"

mito_data = []
for dpo_entry, answer in zip(dpo_entries, query_answers):
    mito_data.append({
        "prompt": dpo_entry["prompt"],
        "adv_prompt": dpo_entry["adv_prompt"],
        "answer": answer  # From query_data
    })


with open("mito_merged.jsonl", "w") as f:
    for entry in mito_data:
        f.write(json.dumps(entry) + "\n")