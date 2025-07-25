#!/usr/bin/env python3
"""
Evaluate compile-pass@k for code LLMs on HumanEval.
"""
import json
import argparse
import os
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils.py_compile_check import compile_ok, detailed_compile_check


def load_model_and_tokenizer(model_path: str, device: str = "auto", load_in_4bit: bool = False):
    """Load model and tokenizer with optional quantization."""
    
    print(f"Loading model from: {model_path}")
    
    # Configure quantization if requested
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print("Using 4-bit NF4 quantization")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map=device,
        torch_dtype=torch.float16 if not load_in_4bit else None,
        trust_remote_code=True
    )
    
    print(f"Model loaded on device: {model.device}")
    return model, tokenizer


def generate_code(model, tokenizer, prompt: str, max_new_tokens: int = 256, temperature: float = 0.0):
    """Generate code completion for a given prompt."""
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        if temperature == 0.0:
            # Greedy decoding
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        else:
            # Sampling
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    
    # Decode generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (remove input prompt)
    completion = generated_text[len(prompt):]
    
    return completion


def evaluate_compile_pass(
    model_path: str,
    split_file: str,
    device: str = "auto",
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    load_in_4bit: bool = False,
    out_file: str = None,
    max_examples: int = None
):
    """Evaluate compile-pass@1 on HumanEval split."""
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path, device, load_in_4bit)
    
    # Load evaluation data
    print(f"Loading evaluation data from: {split_file}")
    examples = []
    with open(split_file, 'r') as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    
    if max_examples:
        examples = examples[:max_examples]
    
    print(f"Evaluating on {len(examples)} examples")
    
    # Evaluation results
    results = []
    compile_pass_count = 0
    
    # Process each example
    for i, example in enumerate(tqdm(examples, desc="Evaluating")):
        task_id = example["task_id"]
        prompt = example["prompt"]
        
        # Generate completion
        try:
            completion = generate_code(
                model, tokenizer, prompt, 
                max_new_tokens=max_new_tokens, 
                temperature=temperature
            )
            
            # Combine prompt + completion for full code
            full_code = prompt + completion
            
            # Check compilation
            compiles = compile_ok(full_code)
            compile_pass_count += int(compiles)
            
            # Get detailed error info if compilation fails
            _, error_msg = detailed_compile_check(full_code)
            
            result = {
                "task_id": task_id,
                "index": i,
                "prompt": prompt,
                "completion": completion,
                "full_code": full_code,
                "compiles": compiles,
                "error_msg": error_msg if not compiles else None,
                "prompt_length": len(prompt),
                "completion_length": len(completion),
            }
            
            results.append(result)
            
            # Print progress every 20 examples
            if (i + 1) % 20 == 0:
                current_pass_rate = compile_pass_count / (i + 1) * 100
                print(f"Progress: {i+1}/{len(examples)}, Current pass rate: {current_pass_rate:.2f}%")
        
        except Exception as e:
            print(f"Error processing example {i} ({task_id}): {e}")
            result = {
                "task_id": task_id,
                "index": i,
                "prompt": prompt,
                "completion": "",
                "full_code": "",
                "compiles": False,
                "error_msg": f"Generation error: {str(e)}",
                "prompt_length": len(prompt),
                "completion_length": 0,
            }
            results.append(result)
    
    # Calculate final metrics
    total_examples = len(results)
    compile_pass_rate = compile_pass_count / total_examples * 100
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Model: {model_path}")
    print(f"Split: {split_file}")
    print(f"Total examples: {total_examples}")
    print(f"Compile pass: {compile_pass_count}/{total_examples}")
    print(f"Compile-pass@1: {compile_pass_rate:.2f}%")
    print(f"Temperature: {temperature}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"4-bit quantization: {load_in_4bit}")
    
    # Save detailed results
    if out_file:
        # Save CSV summary
        csv_file = out_file.replace('.json', '.csv') if out_file.endswith('.json') else out_file + '.csv'
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['task_id', 'index', 'compiles', 'error_msg', 'prompt_length', 'completion_length'])
            for result in results:
                writer.writerow([
                    result['task_id'], result['index'], result['compiles'], 
                    result['error_msg'], result['prompt_length'], result['completion_length']
                ])
        
        # Save detailed JSON
        json_file = out_file if out_file.endswith('.json') else out_file + '.json'
        with open(json_file, 'w') as f:
            json.dump({
                'metadata': {
                    'model_path': model_path,
                    'split_file': split_file,
                    'timestamp': datetime.now().isoformat(),
                    'total_examples': total_examples,
                    'compile_pass_count': compile_pass_count,
                    'compile_pass_rate': compile_pass_rate,
                    'temperature': temperature,
                    'max_new_tokens': max_new_tokens,
                    'load_in_4bit': load_in_4bit,
                },
                'results': results
            }, f, indent=2)
        
        print(f"\nResults saved to:")
        print(f"  CSV: {csv_file}")
        print(f"  JSON: {json_file}")
    
    # Save summary to reports
    os.makedirs("reports", exist_ok=True)
    summary_file = f"reports/baseline_metrics.json"
    
    # Load existing summary or create new
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary_data = json.load(f)
    else:
        summary_data = {"experiments": []}
    
    # Add this experiment
    experiment = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "split_file": split_file,
        "total_examples": total_examples,
        "compile_pass_count": compile_pass_count,
        "compile_pass_rate": compile_pass_rate,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "load_in_4bit": load_in_4bit,
    }
    summary_data["experiments"].append(experiment)
    
    # Save updated summary
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"Summary added to: {summary_file}")
    
    return compile_pass_rate, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate compile-pass@k on HumanEval")
    parser.add_argument("--model_path", required=True, help="Path to model (local or HF hub)")
    parser.add_argument("--split_file", default="splits/dev_humaneval.jsonl", help="JSONL file with evaluation data")
    parser.add_argument("--device", default="auto", help="Device for model")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature (0.0 = greedy)")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit")
    parser.add_argument("--out_file", help="Output file prefix for results")
    parser.add_argument("--max_examples", type=int, help="Limit number of examples (for testing)")
    
    args = parser.parse_args()
    
    # Generate default output file name if not provided
    if not args.out_file:
        model_name = os.path.basename(args.model_path).replace("/", "_")
        quant_suffix = "_4bit" if args.load_in_4bit else "_fp16"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_file = f"results_{model_name}{quant_suffix}_{timestamp}"
    
    # Run evaluation
    pass_rate, results = evaluate_compile_pass(
        model_path=args.model_path,
        split_file=args.split_file,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        load_in_4bit=args.load_in_4bit,
        out_file=args.out_file,
        max_examples=args.max_examples
    )
    
    return pass_rate


if __name__ == "__main__":
    main() 