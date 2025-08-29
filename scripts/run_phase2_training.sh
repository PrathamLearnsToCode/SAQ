#!/bin/bash
# Phase 2 SAQ Training Script

set -e

echo "Starting SAQ Phase 2 - Syntax-Aware Fine-tuning"

mkdir -p ckpts logs

if [ ! -d "ckpts/phi3_nf4" ]; then
    echo "Error: Quantized model from Phase 1 not found at ckpts/phi3_nf4"
    echo "Please run Phase 1 baseline experiments first"
    exit 1
fi

echo "Step 1: Test syntax reward system"
python -c "
import sys
sys.path.append('src')
from reward.syntax_reward import SyntaxRewardCalculator

# Test reward calculator
calculator = SyntaxRewardCalculator(reward_type='composite')
test_code = '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
'''
reward = calculator.calculate_reward(test_code)
print(f'Test reward: {reward:.3f}')
print('Reward system working correctly')
"

echo "Step 2: Start SAQ fine-tuning"
python src/train/saq_ptq_rl.py \
    --config configs/saq_phase2.yaml \
    --quantized_model_path ckpts/phi3_nf4 \
    --data_path splits/dev_humaneval.jsonl \
    --output_dir ckpts/saq_finetuned

echo "Step 3: Evaluate fine-tuned model"
python scripts/eval_compile_pass.py \
    --model_path ckpts/saq_finetuned/checkpoint_final \
    --split_file splits/dev_humaneval.jsonl \
    --load_in_4bit \
    --out_file results/saq_finetuned

echo "Phase 2 Complete! Results:"
echo "  - Fine-tuned model: ckpts/saq_finetuned/"
echo "  - Training logs: logs/saq_training/"
echo "  - Evaluation results: results/saq_finetuned.*"
echo "  - Updated metrics: reports/baseline_metrics.json" 