"""
Extract Persona LoRAs via Contrastive Training

This script trains LoRA adapters to capture persona traits in weight space.
The key idea: train two LoRAs (positive and negative) on contrastive datasets,
then compute the difference to get the persona LoRA direction.

Usage:
    python extract_persona_lora.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --trait evil \
        --output_dir persona_loras/Llama-3.1-8B-Instruct/evil
"""

import json
import os
import sys
import argparse
from pathlib import Path
import torch
from datasets import Dataset
from unsloth import FastLanguageModel

from validate import TrainingConfig
from sft import sft_train
from utils import load_jsonl, load_model_and_tokenizer
from config import setup_credentials

config = setup_credentials()


def train_persona_lora(
    model_name: str,
    trait: str,
    persona_type: str,  # 'pos' or 'neg'
    output_dir: str,
    training_config_overrides: dict = None
):
    """
    Train a single LoRA adapter on positive or negative persona data.

    Args:
        model_name: Base model to use
        trait: Trait name (e.g., 'evil', 'sycophancy')
        persona_type: 'pos' for positive trait, 'neg' for negative trait
        output_dir: Where to save the trained LoRA
        training_config_overrides: Optional config overrides
    """
    print(f"\n{'='*60}")
    print(f"Training {persona_type.upper()} LoRA for trait: {trait}")
    print(f"{'='*60}\n")

    # Load base model
    model, tokenizer = load_model_and_tokenizer(model_name, load_in_4bit=False)

    # Configure LoRA
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=target_modules,
        lora_alpha=64,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=0,
        use_rslora=True,
        loftq_config=None,
        use_dora=False,
    )

    # Load training data based on persona type
    # For positive: use data that elicits the trait
    # For negative: use data that suppresses the trait
    if persona_type == 'pos':
        training_file = f"dataset/{trait}/misaligned_2.jsonl"
    else:
        training_file = f"dataset/{trait}/normal.jsonl"

    if not os.path.exists(training_file):
        raise FileNotFoundError(f"Training file not found: {training_file}")

    rows = load_jsonl(training_file)
    dataset = Dataset.from_list([dict(messages=r['messages']) for r in rows])

    # Split for validation
    split = dataset.train_test_split(test_size=0.1)
    train_dataset = split["train"]
    test_dataset = split["test"]

    # Training configuration
    base_config = {
        "model": model_name,
        "training_file": training_file,
        "test_file": None,
        "max_seq_length": 2048,
        "load_in_4bit": False,
        "loss": "sft",
        "is_peft": True,
        "target_modules": target_modules,
        "lora_bias": "none",
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "use_rslora": True,
        "merge_before_push": False,
        "push_to_private": True,
        "epochs": 1,
        "max_steps": None,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "warmup_steps": 5,
        "learning_rate": 1e-5,
        "logging_steps": 1,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": 0,
        "beta": 0.1,
        "save_steps": 5000,
        "output_dir": output_dir,
        "train_on_responses_only": True,
        "enable_steering_during_training": False
    }

    if training_config_overrides:
        base_config.update(training_config_overrides)

    training_cfg = TrainingConfig(**base_config)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    json.dump(training_cfg.model_dump(), open(os.path.join(output_dir, "training_config.json"), "w"))

    # Train the LoRA
    trainer = sft_train(training_cfg, train_dataset, model, tokenizer, test_dataset=test_dataset)
    trainer.train()

    # Save the LoRA
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\n✓ {persona_type.upper()} LoRA saved to {output_dir}")

    return output_dir


def extract_lora_weights(lora_path: str, layer_idx: int = None):
    """
    Extract LoRA weight matrices from a saved adapter.

    Returns a dict mapping layer names to LoRA weight tensors.
    """
    adapter_config = json.load(open(os.path.join(lora_path, "adapter_config.json")))
    adapter_weights = torch.load(os.path.join(lora_path, "adapter_model.bin"), map_location="cpu", weights_only=False)

    # Extract A and B matrices for each layer
    lora_matrices = {}
    for key, weight in adapter_weights.items():
        if 'lora_A' in key or 'lora_B' in key:
            lora_matrices[key] = weight

    return lora_matrices, adapter_config


def compute_persona_lora_diff(pos_lora_path: str, neg_lora_path: str, output_path: str):
    """
    Compute the difference between positive and negative LoRAs.
    This gives us the persona direction in weight space.

    The difference ΔW = W_pos - W_neg encodes the trait direction.
    """
    print(f"\nComputing persona LoRA difference...")

    pos_weights, pos_config = extract_lora_weights(pos_lora_path)
    neg_weights, neg_config = extract_lora_weights(neg_lora_path)

    # Compute differences
    diff_weights = {}
    for key in pos_weights.keys():
        if key in neg_weights:
            diff = pos_weights[key] - neg_weights[key]
            diff_weights[key] = diff
            print(f"  {key}: shape={diff.shape}, norm={diff.norm().item():.4f}")

    # Save the difference
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(diff_weights, output_path)
    print(f"\n✓ Persona LoRA difference saved to {output_path}")

    return diff_weights


def main():
    parser = argparse.ArgumentParser(description="Extract persona LoRAs via contrastive training")
    parser.add_argument("--model", type=str, required=True, help="Base model name")
    parser.add_argument("--trait", type=str, required=True, help="Trait to extract (e.g., evil, sycophancy)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for persona LoRAs")
    parser.add_argument("--skip_training", action="store_true", help="Skip training if LoRAs already exist")
    parser.add_argument("--layer", type=int, default=None, help="Specific layer to extract (optional)")

    args = parser.parse_args()

    # Create output structure
    pos_lora_dir = os.path.join(args.output_dir, "pos_lora")
    neg_lora_dir = os.path.join(args.output_dir, "neg_lora")
    diff_lora_path = os.path.join(args.output_dir, f"{args.trait}_lora_diff.pt")

    # Train positive LoRA
    if not args.skip_training or not os.path.exists(pos_lora_dir):
        train_persona_lora(args.model, args.trait, "pos", pos_lora_dir)
    else:
        print(f"Skipping positive LoRA training (already exists at {pos_lora_dir})")

    # Train negative LoRA
    if not args.skip_training or not os.path.exists(neg_lora_dir):
        train_persona_lora(args.model, args.trait, "neg", neg_lora_dir)
    else:
        print(f"Skipping negative LoRA training (already exists at {neg_lora_dir})")

    # Compute difference
    compute_persona_lora_diff(pos_lora_dir, neg_lora_dir, diff_lora_path)

    print(f"\n{'='*60}")
    print(f"Persona LoRA extraction complete!")
    print(f"  Positive LoRA: {pos_lora_dir}")
    print(f"  Negative LoRA: {neg_lora_dir}")
    print(f"  Difference: {diff_lora_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
