#!/usr/bin/env python3
"""
Quantize models using BitsAndBytes 4-bit NF4 quantization.
"""
import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datetime import datetime


def quantize_model_nf4(
    model_path: str,
    output_path: str,
    device: str = "auto",
    compute_dtype: str = "float16"
):
    """
    Quantize a model using 4-bit NF4 quantization and save it.
    
    Args:
        model_path: Path to the original model (local or HF hub)
        output_path: Path to save the quantized model
        device: Device for loading
        compute_dtype: Compute dtype for quantization
    """
    
    print(f"Quantizing model: {model_path}")
    print(f"Output path: {output_path}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Configure quantization
    compute_dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype_map[compute_dtype],
        bnb_4bit_use_double_quant=True,  # Nested quantization for memory efficiency
    )
    
    print(f"Quantization config:")
    print(f"  - 4-bit NF4 quantization")
    print(f"  - Compute dtype: {compute_dtype}")
    print(f"  - Double quantization: True")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Load and quantize model
    print("Loading and quantizing model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map=device,
        trust_remote_code=True
    )
    
    print(f"Model loaded on device: {model.device}")
    
    # Print model info
    print(f"\nModel information:")
    print(f"  - Model type: {type(model).__name__}")
    print(f"  - Device: {model.device}")
    
    # Calculate approximate memory usage
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  - Parameters: {param_count:,}")
    
    # Save quantized model
    print(f"\nSaving quantized model to: {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Save quantization info
    quant_info = {
        "timestamp": datetime.now().isoformat(),
        "original_model": model_path,
        "quantization_method": "4-bit NF4 (BitsAndBytes)",
        "compute_dtype": compute_dtype,
        "double_quantization": True,
        "parameter_count": param_count,
        "output_path": output_path,
    }
    
    import json
    with open(os.path.join(output_path, "quantization_info.json"), "w") as f:
        json.dump(quant_info, f, indent=2)
    
    print(f"✅ Quantization complete!")
    print(f"Quantized model saved to: {output_path}")
    print(f"Quantization info saved to: {os.path.join(output_path, 'quantization_info.json')}")
    
    return output_path


def test_quantized_model(model_path: str, test_prompt: str = None):
    """Test the quantized model with a simple prompt."""
    
    if test_prompt is None:
        test_prompt = "def fibonacci(n):\n    \"\"\"\n    Calculate the nth Fibonacci number.\n    \"\"\"\n"
    
    print(f"\nTesting quantized model with prompt:")
    print(f"'{test_prompt}'")
    
    # Load quantized model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Generate
    inputs = tokenizer(test_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    completion = generated_text[len(test_prompt):]
    
    print(f"\nGenerated completion:")
    print(f"'{completion}'")
    
    # Test compilation
    full_code = test_prompt + completion
    try:
        compile(full_code, '<string>', 'exec')
        print("✅ Generated code compiles successfully!")
    except SyntaxError as e:
        print(f"❌ Compilation error: {e}")
    
    return completion


def main():
    parser = argparse.ArgumentParser(description="Quantize models using 4-bit NF4")
    parser.add_argument("--model_path", required=True, help="Path to original model")
    parser.add_argument("--output_path", help="Output path for quantized model")
    parser.add_argument("--device", default="auto", help="Device for loading")
    parser.add_argument("--compute_dtype", default="float16", 
                       choices=["float16", "bfloat16", "float32"],
                       help="Compute dtype for quantization")
    parser.add_argument("--test", action="store_true", help="Test the quantized model after saving")
    parser.add_argument("--test_prompt", help="Custom test prompt")
    
    args = parser.parse_args()
    
    # Generate default output path if not provided
    if not args.output_path:
        model_name = os.path.basename(args.model_path).replace("/", "_")
        args.output_path = f"ckpts/{model_name}_nf4"
    
    # Quantize model
    output_path = quantize_model_nf4(
        model_path=args.model_path,
        output_path=args.output_path,
        device=args.device,
        compute_dtype=args.compute_dtype
    )
    
    # Test if requested
    if args.test:
        test_quantized_model(output_path, args.test_prompt)
    
    print(f"\n🎉 All done! Quantized model available at: {output_path}")
    print(f"\nTo evaluate the quantized model:")
    print(f"python scripts/eval_compile_pass.py --model_path {output_path} --load_in_4bit")


if __name__ == "__main__":
    main() 