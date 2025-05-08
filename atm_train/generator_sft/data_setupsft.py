import json

def transform_entry(entry):
    return {
        "query": entry.get("query", ""),
        "answers": entry.get("answers", []),
        "ctxs": entry.get("ctxs", []),
        "passages": [
            {
                "passage_id": ctx.get("id", ""),
                "is_selected": ctx.get("has_answer", 0),
                "passage_text": ctx.get("text", ""),
                "relevance_score": ctx.get("score", "")
            } for ctx in entry.get("ctxs", [])
        ]
    }

def process_jsonl(input_path, output_path):
    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for line in infile:
            if not line.strip():
                continue  # Skip empty lines
            entry = json.loads(line)
            transformed = transform_entry(entry)
            outfile.write(json.dumps(transformed) + "\n")

# Example usage:
process_jsonl("hotpot_style_fab.jsonl", "hinarr.jsonl")
