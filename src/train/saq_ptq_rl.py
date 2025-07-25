"""
Syntax-Aware Quantization with Post-Training Quantization and RL fine-tuning.
"""
import os
import json
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, BitsAndBytesConfig
)
from transformers.optimization import get_linear_schedule_with_warmup
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from tqdm import tqdm
import sys
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from reward.syntax_reward import (
    SyntaxRewardCalculator, calculate_batch_rewards, 
    scale_rewards, RewardBaseline
)
from utils.py_compile_check import compile_ok

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SAQConfig:
    """Configuration for SAQ training."""
    # Model settings
    model_path: str = "microsoft/Phi-3-mini-4k-instruct"
    quantized_model_path: str = None
    
    # Training settings
    learning_rate: float = 1e-5
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 512
    max_new_tokens: int = 256
    
    # SAQ specific settings
    lambda_syntax: float = 0.5
    reward_type: str = "composite"
    use_tree_sitter: bool = True
    
    # Reward scaling and baseline
    reward_scaling: str = "normalize"
    use_reward_baseline: bool = True
    baseline_decay: float = 0.99
    
    # Composite reward mixing
    alpha_dense: float = 0.4
    beta_sparse: float = 0.5
    
    # RL settings
    use_reinforce: bool = True
    use_supervised: bool = False
    teacher_forcing_ratio: float = 0.3
    
    # Memory optimization
    use_fp16: bool = True
    use_gradient_checkpointing: bool = True
    dataloader_num_workers: int = 0
    
    # Paths
    data_path: str = "splits/dev_humaneval.jsonl"
    output_dir: str = "ckpts/saq_finetuned"
    log_dir: str = "logs/saq_training"
    
    # Evaluation
    eval_steps: int = 100
    save_steps: int = 200
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'SAQConfig':
        """Load config from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


class CodeDataset(Dataset):
    """Dataset for code generation tasks."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Load data
        with open(data_path, 'r') as f:
            for line in f:
                example = json.loads(line.strip())
                self.examples.append(example)
        
        logger.info(f"Loaded {len(self.examples)} examples from {data_path}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        prompt = example["prompt"]
        
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "prompt": prompt,
            "task_id": example.get("task_id", f"task_{idx}"),
            "canonical_solution": example.get("canonical_solution", "")
        }


class SAQTrainer:
    """Syntax-Aware Quantization Trainer."""
    
    def __init__(self, config: SAQConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.reward_calculator = None
        self.optimizer = None
        self.scheduler = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Metrics tracking
        self.training_metrics = {
            "loss": [],
            "syntax_reward": [],
            "compile_rate": [],
            "learning_rate": []
        }
        
        # Reward baseline for variance reduction
        self.reward_baseline = RewardBaseline(decay=config.baseline_decay) if config.use_reward_baseline else None
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer."""
        logger.info(f"Loading model from: {self.config.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, 
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load quantized model if specified
        if self.config.quantized_model_path:
            logger.info(f"Loading quantized model from: {self.config.quantized_model_path}")
            
            # Configure quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.quantized_model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            # Load FP16 model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.float16 if self.config.use_fp16 else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
        
        # Enable gradient checkpointing for memory efficiency
        if self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        logger.info(f"Model loaded on device: {self.model.device}")
        
        # Initialize reward calculator
        self.reward_calculator = SyntaxRewardCalculator(
            reward_type=self.config.reward_type,
            use_tree_sitter=self.config.use_tree_sitter
        )
    
    def setup_optimizer_and_scheduler(self, num_training_steps: int):
        """Setup optimizer and learning rate scheduler."""
        # Only optimize parameters that require gradients
        optimizer_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = torch.optim.AdamW(
            optimizer_params,
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * num_training_steps),
            num_training_steps=num_training_steps
        )
    
    def generate_code(self, prompts: List[str], temperature: float = 0.7) -> List[str]:
        """Generate code completions for given prompts."""
        self.model.eval()
        generated_codes = []
        
        with torch.no_grad():
            for prompt in prompts:
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True,
                    max_length=self.config.max_length
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                # Generate
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                # Decode generated text
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                completion = generated_text[len(prompt):]
                full_code = prompt + completion
                
                generated_codes.append(full_code)
        
        return generated_codes
    
    def compute_reinforce_loss(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generated_codes: List[str],
        rewards: List[float]
    ) -> torch.Tensor:
        """Compute REINFORCE loss with syntax rewards."""
        self.model.train()
        
        # Forward pass to get log probabilities
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )
        
        # Get log probabilities for generated tokens
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Calculate REINFORCE loss
        batch_size = input_ids.size(0)
        reinforce_loss = 0.0
        
        for i in range(batch_size):
            reward = rewards[i]
            
            # Get log probabilities for this sequence
            seq_log_probs = log_probs[i]
            seq_input_ids = input_ids[i]
            
            # Calculate sequence log probability
            seq_log_prob = 0.0
            for t in range(1, len(seq_input_ids)):
                if seq_input_ids[t] != self.tokenizer.pad_token_id:
                    seq_log_prob += seq_log_probs[t-1, seq_input_ids[t]]
            
            # REINFORCE: -log_prob * (reward - baseline)
            # Using simple moving average as baseline
            baseline = np.mean(rewards) if len(rewards) > 1 else 0.0
            advantage = reward - baseline
            
            reinforce_loss -= seq_log_prob * advantage
        
        return reinforce_loss / batch_size
    
    def compute_supervised_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        teacher_codes: List[str],
        rewards: List[float]
    ) -> torch.Tensor:
        """Compute supervised loss with syntax regularization."""
        # Standard language modeling loss
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )
        
        lm_loss = outputs.loss
        
        # Syntax regularization term
        avg_reward = np.mean(rewards)
        syntax_penalty = max(0.0, 0.8 - avg_reward)  # Penalty if avg reward < 0.8
        
        total_loss = lm_loss + self.config.lambda_syntax * syntax_penalty
        
        return total_loss
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Execute one training step."""
        prompts = batch["prompt"]
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        
        # Generate codes
        generated_codes = self.generate_code(prompts)
        
        # Calculate syntax rewards
        raw_rewards = calculate_batch_rewards(
            generated_codes,
            prompts,
            reward_type=self.config.reward_type,
            use_tree_sitter=self.config.use_tree_sitter
        )
        
        # Apply reward scaling
        if self.config.reward_scaling != "none":
            scaled_rewards = scale_rewards(raw_rewards, method=self.config.reward_scaling)
        else:
            scaled_rewards = raw_rewards
        
        # Apply baseline normalization for RL
        if self.reward_baseline and self.config.use_reinforce:
            advantages = self.reward_baseline.update(scaled_rewards)
            rewards = advantages
        else:
            rewards = scaled_rewards
        
        # Compute loss based on training method
        if self.config.use_reinforce:
            loss = self.compute_reinforce_loss(input_ids, attention_mask, generated_codes, rewards)
        else:
            loss = self.compute_supervised_loss(input_ids, attention_mask, generated_codes, rewards)
        
        # Backward pass
        loss = loss / self.config.gradient_accumulation_steps
        loss.backward()
        
        # Calculate metrics
        compile_rate = sum(1 for code in generated_codes if compile_ok(code)) / len(generated_codes)
        avg_raw_reward = np.mean(raw_rewards)
        avg_scaled_reward = np.mean(scaled_rewards) if 'scaled_rewards' in locals() else avg_raw_reward
        
        metrics = {
            "loss": loss.item() * self.config.gradient_accumulation_steps,
            "syntax_reward": avg_raw_reward,
            "scaled_reward": avg_scaled_reward,
            "compile_rate": compile_rate,
            "learning_rate": self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate,
            "baseline": self.reward_baseline.get_baseline() if self.reward_baseline else 0.0
        }
        
        return metrics
    
    def train(self, dataset: CodeDataset):
        """Main training loop."""
        logger.info("Starting SAQ training...")
        
        # Setup data loader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers,
            collate_fn=self.collate_fn
        )
        
        # Calculate total training steps
        total_steps = len(dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        self.setup_optimizer_and_scheduler(total_steps)
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")
            
            epoch_metrics = {"loss": [], "syntax_reward": [], "compile_rate": []}
            
            for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}")):
                # Training step
                step_metrics = self.train_step(batch)
                
                # Accumulate metrics
                for key, value in step_metrics.items():
                    if key in epoch_metrics:
                        epoch_metrics[key].append(value)
                    self.training_metrics[key].append(value)
                
                # Gradient accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                
                # Logging
                if self.global_step % self.config.eval_steps == 0:
                    avg_metrics = {k: np.mean(v[-10:]) for k, v in self.training_metrics.items() if v}
                    logger.info(f"Step {self.global_step}: {avg_metrics}")
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
            
            # Epoch summary
            epoch_summary = {k: np.mean(v) for k, v in epoch_metrics.items()}
            logger.info(f"Epoch {epoch + 1} summary: {epoch_summary}")
        
        # Final save
        self.save_checkpoint(final=True)
        logger.info("Training completed!")
    
    def collate_fn(self, batch):
        """Custom collate function for batching."""
        # Pad sequences
        input_ids = [item["input_ids"] for item in batch]
        attention_masks = [item["attention_mask"] for item in batch]
        
        # Pad to max length in batch
        max_len = max(len(ids) for ids in input_ids)
        
        padded_input_ids = []
        padded_attention_masks = []
        
        for i in range(len(input_ids)):
            ids = input_ids[i]
            mask = attention_masks[i]
            
            # Pad
            pad_length = max_len - len(ids)
            padded_ids = torch.cat([ids, torch.full((pad_length,), self.tokenizer.pad_token_id)])
            padded_mask = torch.cat([mask, torch.zeros(pad_length)])
            
            padded_input_ids.append(padded_ids)
            padded_attention_masks.append(padded_mask)
        
        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_masks),
            "prompt": [item["prompt"] for item in batch],
            "task_id": [item["task_id"] for item in batch],
        }
    
    def save_checkpoint(self, final: bool = False):
        """Save model checkpoint."""
        suffix = "final" if final else f"step_{self.global_step}"
        checkpoint_dir = os.path.join(self.config.output_dir, f"checkpoint_{suffix}")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state
        training_state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "config": self.config.__dict__,
            "metrics": self.training_metrics
        }
        
        with open(os.path.join(checkpoint_dir, "training_state.json"), "w") as f:
            json.dump(training_state, f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SAQ Training")
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--model_path", help="Override model path")
    parser.add_argument("--quantized_model_path", help="Path to quantized model")
    parser.add_argument("--data_path", help="Override data path")
    parser.add_argument("--output_dir", help="Override output directory")
    
    args = parser.parse_args()
    
    # Load config
    config = SAQConfig.from_yaml(args.config)
    
    # Override config with command line args
    if args.model_path:
        config.model_path = args.model_path
    if args.quantized_model_path:
        config.quantized_model_path = args.quantized_model_path
    if args.data_path:
        config.data_path = args.data_path
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Initialize trainer
    trainer = SAQTrainer(config)
    trainer.setup_model_and_tokenizer()
    
    # Load dataset
    dataset = CodeDataset(
        config.data_path,
        trainer.tokenizer,
        config.max_length
    )
    
    # Start training
    trainer.train(dataset)


if __name__ == "__main__":
    main() 