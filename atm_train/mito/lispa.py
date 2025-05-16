import json

def json_tree(json_obj):
    """Recursively generate a tree of the keys in a JSON object."""
    if isinstance(json_obj, dict):
        return {key: json_tree(json_obj[key]) for key in json_obj}
    elif isinstance(json_obj, list):
        return [json_tree(item) for item in json_obj]
    else:
        return None  # Base case for non-dict/list items

def load_jsonl_and_generate_tree(file_path):
    """Load a JSONL file and generate a JSON tree from its structure."""
    with open(file_path, 'r') as f:
        # Process only the first line (remove this if you want to process all lines)
        first_line = json.loads(f.readline())
        return json_tree(first_line)
# Replace with your file path

tree = load_jsonl_and_generate_tree('mito_merged.jsonl')


# Print the tree in JSON format
print(json.dumps(tree, indent=2))