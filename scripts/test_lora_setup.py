#!/usr/bin/env python3
"""
Test script to verify LoRA setup for quantized model training.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

def test_lora_setup():
    """Test LoRA setup with quantized model."""
    print("Testing LoRA setup for quantized model training...")
    
    # Load quantized model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    print("Loading quantized model...")
    model = AutoModelForCausalLM.from_pretrained(
        "ckpts/phi3_nf4",
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    print("Applying LoRA to model...")
    model = get_peft_model(model, lora_config)
    
    print("\nTrainable parameters info:")
    model.print_trainable_parameters()
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable_params)
    
    print(f"\nFound {len(trainable_params)} trainable parameter tensors")
    print(f"Total trainable parameters: {total_trainable:,}")
    
    if total_trainable > 0:
        print("LoRA setup successful! Model is ready for training.")
        return True
    else:
        print("LoRA setup failed! No trainable parameters found.")
        return False

if __name__ == "__main__":
    success = test_lora_setup()
    exit(0 if success else 1) 