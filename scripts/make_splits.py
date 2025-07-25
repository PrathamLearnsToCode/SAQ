#!/usr/bin/env python3
"""
Download HumanEval dataset and create splits for SAQ evaluation.
"""
import json
import os
from pathlib import Path
from datasets import load_dataset
import argparse


def download_humaneval(output_dir: str = "splits", sample_size: int = None):
    """
    Download HumanEval dataset and create JSONL splits.
    
    Args:
        output_dir: Directory to save splits
        sample_size: If specified, sample this many examples
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading HumanEval dataset...")
    
    # Load the dataset from Hugging Face
    dataset = load_dataset("openai_humaneval", split="test")
    
    print(f"Loaded {len(dataset)} examples from HumanEval")
    
    # Convert to our format
    examples = []
    for i, item in enumerate(dataset):
        example = {
            "task_id": item["task_id"],
            "prompt": item["prompt"], 
            "canonical_solution": item["canonical_solution"],
            "test": item["test"],
            "entry_point": item["entry_point"],
            "docstring": item.get("docstring", ""),
            "index": i
        }
        examples.append(example)
    
    # Sample if requested
    if sample_size and sample_size < len(examples):
        import random
        random.seed(42)  # For reproducibility
        examples = random.sample(examples, sample_size)
        print(f"Sampled {sample_size} examples")
    
    # Save full dev split
    dev_file = os.path.join(output_dir, "dev_humaneval.jsonl")
    with open(dev_file, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")
    
    print(f"Saved {len(examples)} examples to {dev_file}")
    
    # Create a smaller test split (first 20 examples for quick testing)
    test_examples = examples[:min(20, len(examples))]
    test_file = os.path.join(output_dir, "test_humaneval.jsonl")
    with open(test_file, "w") as f:
        for example in test_examples:
            f.write(json.dumps(example) + "\n")
    
    print(f"Saved {len(test_examples)} examples to {test_file}")
    
    # Print some stats
    print(f"\nDataset Statistics:")
    print(f"Total examples: {len(examples)}")
    print(f"Average prompt length: {sum(len(ex['prompt']) for ex in examples) / len(examples):.1f} chars")
    
    # Show first example
    print(f"\nFirst example:")
    print(f"Task ID: {examples[0]['task_id']}")
    print(f"Prompt preview: {examples[0]['prompt'][:200]}...")
    
    return len(examples)


def main():
    parser = argparse.ArgumentParser(description="Download and prepare HumanEval splits")
    parser.add_argument("--output_dir", default="splits", help="Output directory")
    parser.add_argument("--sample_size", type=int, help="Sample size (default: use all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    import random
    random.seed(args.seed)
    
    # Download and create splits
    num_examples = download_humaneval(args.output_dir, args.sample_size)
    
    print(f"\n✅ Successfully created HumanEval splits with {num_examples} examples")
    print(f"Files created in {args.output_dir}/:")
    print(f"  - dev_humaneval.jsonl ({num_examples} examples)")
    print(f"  - test_humaneval.jsonl ({min(20, num_examples)} examples)")


if __name__ == "__main__":
    main() 