#!/bin/bash
# Phase 1 Baseline Experiments for SAQ

set -e

echo "🚀 Starting SAQ Phase 1 Baseline Experiments"

# Model to use (change this as needed)
MODEL_PATH="microsoft/Phi-3-mini-4k-instruct"

# Create necessary directories
mkdir -p ckpts results

echo "📥 Step 1: Download HumanEval dataset"
python scripts/make_splits.py --output_dir splits

echo "📊 Step 2: Evaluate FP16 baseline"
python scripts/eval_compile_pass.py \
    --model_path $MODEL_PATH \
    --split_file splits/dev_humaneval.jsonl \
    --out_file results/fp16_baseline

echo "⚙️  Step 3: Create 4-bit NF4 quantized model"
python scripts/quant_bnb_nf4.py \
    --model_path $MODEL_PATH \
    --output_path ckpts/phi3_nf4 \
    --test

echo "📊 Step 4: Evaluate 4-bit NF4 model"
python scripts/eval_compile_pass.py \
    --model_path ckpts/phi3_nf4 \
    --split_file splits/dev_humaneval.jsonl \
    --load_in_4bit \
    --out_file results/nf4_baseline

echo "📈 Phase 1 Complete! Check results in:"
echo "  - reports/baseline_metrics.json"
echo "  - results/ directory" 