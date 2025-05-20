import json
from typing import List, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

import re
from collections import Counter
from peft import PeftModel

def normalize_text(text: str) -> str:
    """Lowercase, remove punctuation, and extra whitespace."""
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)  # Remove articles
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

def compute_subspan_exact_match(prediction: str, ground_truths: List[str]) -> bool:
    """Check if the normalized prediction is a substring of the normalized ground truth."""
    if normalize_text(prediction) == "":
        return False
    else:
        return any(((normalize_text(prediction) in normalize_text(gt)) or (normalize_text(gt) in normalize_text(prediction)))for gt in ground_truths)

def compute_exact_match(prediction: str, ground_truths: List[str]) -> bool:
    """Check if the normalized prediction is a substring of the normalized ground truth."""
    return any(normalize_text(prediction) == normalize_text(gt) for gt in ground_truths)

def compute_f1(prediction: str, ground_truths: List[str]) -> float:
    """Return the max F1 score over all ground truths."""
    def single_f1(pred, gt):
        pred_tokens = normalize_text(pred).split()
        gt_tokens = normalize_text(gt).split()
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        return 2 * (precision * recall) / (precision + recall)
    return max((single_f1(prediction, gt) for gt in ground_truths), default=0.0)

class MITOModelEvaluator:
    def __init__(self, base_model_path: str, adapter_path: str = None, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the evaluator with the trained model and optional LoRA adapter.
        
        Args:
            base_model_path: Path to the base model
            adapter_path: Optional path to the LoRA adapter checkpoint
            device: Device to run evaluation on
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path).to(self.device)
        
        if adapter_path:
            # Load LoRA adapter on top of base model
            self.model = PeftModel.from_pretrained(base_model, adapter_path).to(self.device)
        else:
            self.model = base_model
        
        self.model.eval()
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def prepare_input(self, query: str, contexts: List[dict]) -> str:
        """
        Formats the input using the [INST] ... [/INST] format expected by Mistral-Instruct models.
        
        Args:
            query: The input question/query.
            contexts: List of context dicts with 'text' field.
            
        Returns:
            Formatted input string.
        """
        system_prompt = (
            "<<SYS>>\n"
            "You are a helpful assistant. Use the provided context to answer the question.\n"
            "If the answer is not in the context, say you don't know.\n"
            "<</SYS>>\n"
        )

        # Join all context passages
        context_str = "\n".join([f"{ctx['text']}" for ctx in contexts])
        
        # Construct the full prompt
        prompt = (
            f"[INST] {system_prompt}"
            f"Context:\n{context_str}\n\n"
            f"Question: {query}"
            f" [/INST]\n"
            f"Answer:"
        )

        return prompt
        
    def generate_response(self, input_text: str, max_length: int = 512) -> str:
        """
        Generate a response from the model given input text.
        
        Args:
            input_text: Formatted input text
            max_length: Maximum length of generated response
            
        Returns:
            Generated answer text
        """
        self.tokenizer.truncation_side = "left"  # Set truncation side to left
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        # Decode only the generated part (skip input)
        input_length = inputs.input_ids.shape[1]
        generated = outputs[:, input_length:]
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)
    
    def evaluate_single_example(self, example: Dict) -> Dict:
        """
        Evaluate a single example from the dataset.

        Args:
            example: Dictionary containing 'query', 'ctxs', and 'ground_truth'

        Returns:
            Dictionary with original query, contexts, generated answer, and metrics
        """
        query = example.get("query", "")
        contexts = example.get("ctxs", [])
        ground_truths = example.get("answers", [])
        #ground_truth = example.get("ground_truth", "")  # Add ground truth to the dataset
        if not query or not contexts:
            print("We got a poopy")
            return {
                "query": query,
                "contexts": contexts,
                "answer": "",
                "ground_truth": ground_truths,
                "has_answer": False,
                "subspan_exact_match": False,
                "exact_match": False,
                "f1_score": 0.0
            }
        
        input_text = self.prepare_input(query, contexts)
        answer = self.generate_response(input_text)
        
        # Compute metrics
        has_answer = bool(answer.strip())
        subspan_exact_match = compute_subspan_exact_match(answer, ground_truths)
        exact_match = compute_exact_match(answer, ground_truths)
        f1_score = compute_f1(answer, ground_truths)
        
        return {
            "query": query,
            "contexts": contexts,
            "answer": answer,
            "ground_truth": ground_truths,
            "has_answer": has_answer,
            "subspan_exact_match": subspan_exact_match,
            "exact_match": exact_match,
            "f1_score": f1_score
        }
    
    def evaluate_dataset(self, data: List[Dict], output_file: str = None) -> Dict:
        """
        Evaluate a full dataset and compute cumulative metrics.

        Args:
            data: List of examples to evaluate
            output_file: Optional path to save results

        Returns:
            Dictionary with evaluation results and cumulative metrics
        """
        results = []
        total_f1 = 0.0
        total_exact_match = 0
        total_subspan_exact_match = 0

        for example in tqdm(data, desc="Evaluating examples"):
            result = self.evaluate_single_example(example)
            results.append(result)

            # Accumulate metrics
            total_f1 += result["f1_score"]
            total_exact_match += int(result["exact_match"])
            total_subspan_exact_match += int(result["subspan_exact_match"])

            # Optionally print some examples
            if len(results) <= 3:  # Print first few examples
                print("\nExample Evaluation:")
                print(f"Query: {result['query']}")
                print(f"Answer: {result['answer']}")
                print(f"Ground Truths: {result['ground_truth']}")
                print(f"F1 Score: {result['f1_score']}")
                print(f"Exact Match: {result['exact_match']}")
                print(f"Subspan Exact Match: {result['subspan_exact_match']}")
                print("-" * 50)

        # Compute cumulative metrics
        num_examples = len(data)
        cumulative_f1 = total_f1 / num_examples if num_examples > 0 else 0.0
        cumulative_exact_match = total_exact_match / num_examples if num_examples > 0 else 0.0
        cumulative_subspan_exact_match = total_subspan_exact_match / num_examples if num_examples > 0 else 0.0

        # Save results to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

        # Return results and cumulative metrics
        return {
            "results": results,
            "cumulative_f1": cumulative_f1,
            "cumulative_exact_match": cumulative_exact_match,
            "cumulative_subspan_exact_match": cumulative_subspan_exact_match,
        }


if __name__ == "__main__":
    # Example usage
    import warnings
    base_model_path = "/home/Mahdiyar/Research/Julien/mito/newruns/model_final"
    #"/home/Mahdiyar/Research/Julien/mito/experiments/checkpoint-31"#"/home/julien/ATM-RAG/atm_train/generator_sft/experiments/model_final"
    #adapter_path = "/home/Mahdiyar/Research/Julien/mito/experiments/checkpoint-31"
    adapter_path = None
    warnings.filterwarnings("ignore", message="Found missing adapter keys while loading the checkpoint")
    evaluator = MITOModelEvaluator(base_model_path, adapter_path)
    
    # Load your evaluation data
    with open("hotpot_style.jsonl") as f:
        eval_data = [json.loads(line) for line in f if line.strip()]
    
    # Evaluate - this could be a single example or a list
    if isinstance(eval_data, dict):
        eval_data = [eval_data]  # Convert single example to list
    
    evaluation_results = evaluator.evaluate_dataset(eval_data, output_file="mitodpofull_eval.json")
    
    print("\nEvaluation complete. Results saved to mitodpofull_eval.json")
    print(f"Cumulative F1 Score: {evaluation_results['cumulative_f1']:.4f}")
    print(f"Cumulative Exact Match: {evaluation_results['cumulative_exact_match']:.4f}")
    print(f"Cumulative Subspan Exact Match: {evaluation_results['cumulative_subspan_exact_match']:.4f}")