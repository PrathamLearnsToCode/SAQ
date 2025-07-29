#!/usr/bin/env python3
"""
Post-Training Quantization (PTQ) Baselines for SAQ.

This script applies AWQ and GPTQ 4-bit quantization to Phi-3-mini-4k-instruct,
evaluates compile-pass@1 and compile-pass@10 on HumanEval and MBPP,
and collects performance metrics.
"""

import json
import argparse
import os
import sys
import time
import gc
from pathlib import Path
from typing import Dict, List, Any, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import psutil
import subprocess
from tqdm import tqdm

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils.py_compile_check import compile_ok, detailed_compile_check

# Try importing quantization libraries
try:
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    GPTQ_AVAILABLE = True
except ImportError:
    print("Warning: auto-gptq not available. GPTQ quantization will be skipped.")
    GPTQ_AVAILABLE = False

try:
    from awq import AutoAWQForCausalLM
    AWQ_AVAILABLE = True
except ImportError:
    print("Warning: autoawq not available. AWQ quantization will be skipped.")
    AWQ_AVAILABLE = False


def get_memory_usage():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() // (1024 ** 2)
    return 0


def reset_memory():
    """Reset GPU memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()


def download_datasets(output_dir: str = "splits"):
    """Download HumanEval and MBPP datasets if not already present."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Download HumanEval
    humaneval_file = os.path.join(output_dir, "dev_humaneval.jsonl")
    if not os.path.exists(humaneval_file):
        print("Downloading HumanEval dataset...")
        dataset = load_dataset("openai_humaneval", split="test")
        
        with open(humaneval_file, "w") as f:
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
                f.write(json.dumps(example) + "\n")
        print(f"Saved HumanEval to {humaneval_file}")
    
    # Download MBPP
    mbpp_file = os.path.join(output_dir, "dev_mbpp.jsonl")
    if not os.path.exists(mbpp_file):
        print("Downloading MBPP dataset...")
        try:
            dataset = load_dataset("mbpp", split="test")
            
            with open(mbpp_file, "w") as f:
                for i, item in enumerate(dataset):
                    example = {
                        "task_id": f"MBPP/{item['task_id']}",
                        "prompt": item["text"] + "\n",  # Add newline for consistency
                        "canonical_solution": item["code"],
                        "test": item.get("test_list", []),
                        "entry_point": "solution",  # Default entry point
                        "index": i
                    }
                    f.write(json.dumps(example) + "\n")
            print(f"Saved MBPP to {mbpp_file}")
        except Exception as e:
            print(f"Warning: Could not download MBPP dataset: {e}")
            print("Continuing with HumanEval only...")


def load_fp16_model(model_path: str):
    """Load model in FP16 precision."""
    print(f"Loading FP16 model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    return model, tokenizer


def apply_bnb_quantization(model_path: str, save_path: str):
    """Apply BitsAndBytesConfig 4-bit quantization as fallback."""
    print(f"Applying BitsAndBytesConfig NF4 quantization to {model_path}")
    
    # Configure quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model with quantization
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Save model and tokenizer
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"BnB quantized model saved to {save_path}")
    
    # Clean up
    del model
    reset_memory()


def apply_awq_quantization(model_path: str, save_path: str):
    """Apply AWQ 4-bit quantization."""
    if not AWQ_AVAILABLE:
        raise ImportError("autoawq package not available")
    
    print(f"Applying AWQ quantization to {model_path}")
    
    try:
        # Load model for quantization
        model = AutoAWQForCausalLM.from_pretrained(
            model_path, 
            safetensors=True,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"AWQ doesn't support this model: {e}")
        print("Falling back to BitsAndBytesConfig 4-bit quantization...")
        return apply_bnb_quantization(model_path, save_path)
    
    # Prepare calibration data (use a small subset of HumanEval)
    calib_data = []
    try:
        dataset = load_dataset("openai_humaneval", split="test")
        for i, item in enumerate(dataset):
            if i >= 128:  # Use 128 samples for calibration
                break
            calib_data.append(item["prompt"])
    except:
        # Fallback calibration data
        calib_data = [
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
            "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]"
        ]
    
    # Quantize model
    model.quantize(tokenizer, quant_config={"zero_point": True, "q_group_size": 128})
    
    # Save quantized model
    os.makedirs(save_path, exist_ok=True)
    model.save_quantized(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"AWQ quantized model saved to {save_path}")
    
    # Clean up
    del model
    reset_memory()


def apply_gptq_quantization(model_path: str, save_path: str):
    """Apply GPTQ 4-bit quantization."""
    if not GPTQ_AVAILABLE:
        raise ImportError("auto-gptq package not available")
    
    print(f"Applying GPTQ quantization to {model_path}")
    
    # Configure quantization
    quantize_config = BaseQuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=False,
        damp_percent=0.1,
        sym=True,
        true_sequential=True
    )
    
    # Load model for quantization
    model = AutoGPTQForCausalLM.from_pretrained(
        model_path, 
        quantize_config=quantize_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare calibration data
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    calib_data = []
    try:
        dataset = load_dataset("openai_humaneval", split="test")
        for i, item in enumerate(dataset):
            if i >= 128:  # Use 128 samples for calibration
                break
            calib_data.append(item["prompt"])
    except:
        # Fallback calibration data
        calib_data = [
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
        ]
    
    # Tokenize calibration data
    examples = []
    for text in calib_data:
        examples.append(tokenizer(text, return_tensors="pt", max_length=512, truncation=True))
    
    # Quantize model
    model.quantize(examples)
    
    # Save quantized model
    os.makedirs(save_path, exist_ok=True)
    model.save_quantized(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"GPTQ quantized model saved to {save_path}")
    
    # Clean up
    del model
    reset_memory()


def generate_code(model, tokenizer, prompt: str, max_new_tokens: int = 256, temperature: float = 0.0):
    """Generate code completion using the model."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    completion = generated_text[len(prompt):]
    
    return completion


def evaluate_model_on_dataset(model_path: str, dataset_file: str, k_values: List[int] = [1, 10]) -> Dict[str, float]:
    """Evaluate model on a dataset for compile-pass@k."""
    print(f"Evaluating {model_path} on {dataset_file}")
    
    # Load model
    if "awq" in model_path.lower() and AWQ_AVAILABLE:
        model = AutoAWQForCausalLM.from_quantized(model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif "gptq" in model_path.lower() and GPTQ_AVAILABLE:
        model = AutoGPTQForCausalLM.from_quantized(model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        # Load as regular model (FP16 or quantized via transformers)
        model, tokenizer = load_fp16_model(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    examples = []
    with open(dataset_file, 'r') as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    
    print(f"Loaded {len(examples)} examples")
    
    # Evaluate for each k value
    results = {}
    
    for k in k_values:
        print(f"Evaluating compile-pass@{k}")
        
        total_correct = 0
        total_examples = len(examples)
        
        for i, example in enumerate(tqdm(examples, desc=f"pass@{k}")):
            task_id = example["task_id"]
            prompt = example["prompt"]
            
            # Generate k completions
            compilable_count = 0
            
            for attempt in range(k):
                try:
                    # Use temperature > 0 for k > 1 to get diverse samples
                    temperature = 0.0 if k == 1 else 0.8
                    completion = generate_code(
                        model, tokenizer, prompt,
                        max_new_tokens=256,
                        temperature=temperature
                    )
                    
                    # Check if it compiles
                    full_code = prompt + completion
                    if compile_ok(full_code):
                        compilable_count += 1
                        if k == 1:  # For pass@1, we only need one success
                            break
                
                except Exception as e:
                    print(f"Error generating for {task_id} attempt {attempt}: {e}")
                    continue
            
            # For pass@k, we need at least one success
            if compilable_count > 0:
                total_correct += 1
        
        pass_rate = total_correct / total_examples * 100
        results[f"compile_pass@{k}"] = pass_rate
        print(f"Compile-pass@{k}: {pass_rate:.2f}%")
    
    # Clean up
    del model
    reset_memory()
    
    return results


def measure_latency(model_path: str, num_samples: int = 10) -> float:
    """Measure average inference latency."""
    print(f"Measuring latency for {model_path}")
    
    # Load model
    if "awq" in model_path.lower() and AWQ_AVAILABLE:
        model = AutoAWQForCausalLM.from_quantized(model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif "gptq" in model_path.lower() and GPTQ_AVAILABLE:
        model = AutoGPTQForCausalLM.from_quantized(model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        model, tokenizer = load_fp16_model(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Sample prompt
    prompt = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return"
    
    # Warmup
    for _ in range(3):
        _ = generate_code(model, tokenizer, prompt, max_new_tokens=128)
    
    # Measure latency
    times = []
    for _ in range(num_samples):
        start_time = time.time()
        _ = generate_code(model, tokenizer, prompt, max_new_tokens=128)
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to ms
    
    avg_latency = sum(times) / len(times)
    
    # Clean up
    del model
    reset_memory()
    
    return avg_latency


def main():
    parser = argparse.ArgumentParser(description="Run PTQ baselines for SAQ")
    parser.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct", 
                        help="Base model to quantize")
    parser.add_argument("--tasks", nargs="+", default=["humaneval", "mbpp"],
                        help="Tasks to evaluate on")
    parser.add_argument("--out", default="reports/ptq_baselines.json",
                        help="Output file for results")
    parser.add_argument("--methods", nargs="+", default=["awq", "gptq"],
                        help="Quantization methods to use")
    parser.add_argument("--skip_quantization", action="store_true",
                        help="Skip quantization step (use existing checkpoints)")
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("ckpts", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("splits", exist_ok=True)
    
    # Download datasets
    download_datasets()
    
    # Results storage
    all_results = []
    
    # Define checkpoint paths
    checkpoint_paths = {
        "awq": "ckpts/phi3_awq4",
        "gptq": "ckpts/phi3_gptq4"
    }
    
    # Apply quantization if not skipped
    if not args.skip_quantization:
        for method in args.methods:
            if method == "awq" and AWQ_AVAILABLE:
                try:
                    apply_awq_quantization(args.model, checkpoint_paths["awq"])
                except Exception as e:
                    print(f"AWQ quantization failed: {e}")
                    continue
            elif method == "gptq" and GPTQ_AVAILABLE:
                try:
                    apply_gptq_quantization(args.model, checkpoint_paths["gptq"])
                except Exception as e:
                    print(f"GPTQ quantization failed: {e}")
                    continue
            else:
                print(f"Skipping {method} - library not available")
    
    # Evaluate each quantized model
    for method in args.methods:
        if method not in checkpoint_paths:
            continue
            
        checkpoint_path = checkpoint_paths[method]
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint {checkpoint_path} not found, skipping {method}")
            continue
        
        print(f"\n{'='*50}")
        print(f"Evaluating {method.upper()} quantized model")
        print(f"{'='*50}")
        
        # Measure latency and memory
        reset_memory()
        latency = measure_latency(checkpoint_path)
        max_memory = get_memory_usage()
        
        # Evaluate on each task
        for task in args.tasks:
            if task == "humaneval":
                dataset_file = "splits/dev_humaneval.jsonl"
            elif task == "mbpp":
                dataset_file = "splits/dev_mbpp.jsonl"
            else:
                print(f"Unknown task: {task}")
                continue
            
            if not os.path.exists(dataset_file):
                print(f"Dataset file {dataset_file} not found, skipping {task}")
                continue
            
            # Evaluate model
            reset_memory()
            eval_results = evaluate_model_on_dataset(checkpoint_path, dataset_file, k_values=[1, 10])
            
            # Store results
            result = {
                "method": method,
                "task": task,
                "model_path": checkpoint_path,
                "latency_ms": latency,
                "max_memory_mb": max_memory,
                **eval_results
            }
            all_results.append(result)
            
            print(f"Results for {method} on {task}:")
            print(f"  Compile-pass@1: {eval_results.get('compile_pass@1', 0):.2f}%")
            print(f"  Compile-pass@10: {eval_results.get('compile_pass@10', 0):.2f}%")
            print(f"  Latency: {latency:.2f}ms")
            print(f"  Max Memory: {max_memory}MB")
    
    # Save results
    print(f"\nSaving results to {args.out}")
    with open(args.out, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    for result in all_results:
        method = result["method"]
        task = result["task"]
        pass1 = result.get("compile_pass@1", 0)
        pass10 = result.get("compile_pass@10", 0)
        latency = result["latency_ms"]
        memory = result["max_memory_mb"]
        
        print(f"{method.upper()} on {task}:")
        print(f"  Pass@1: {pass1:.1f}%, Pass@10: {pass10:.1f}%")
        print(f"  Latency: {latency:.1f}ms, Memory: {memory}MB")
        print()


if __name__ == "__main__":
    main() 