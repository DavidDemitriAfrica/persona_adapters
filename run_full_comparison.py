"""
Full Comparison Experiment: Activation vs LoRA Persona Steering

This script runs the complete pipeline:
1. Extract activation vectors
2. Extract LoRA adapters
3. Evaluate both methods at multiple coefficients
4. Generate Anthropic-style visualizations

Run in tmux for long-running experiments.
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
import argparse
import wandb

# Color codes for pretty output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def log(msg, color=Colors.CYAN):
    print(f"{color}{msg}{Colors.END}")

def run_command(cmd, description, log_file=None):
    """Run a command and log output."""
    log(f"\n{'='*60}", Colors.HEADER)
    log(f"{description}", Colors.BOLD)
    log(f"{'='*60}", Colors.HEADER)
    log(f"Command: {cmd}", Colors.BLUE)

    start_time = time.time()

    if log_file:
        with open(log_file, 'w') as f:
            f.write(f"Command: {cmd}\n\n")
            result = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
    else:
        result = subprocess.run(cmd, shell=True)

    elapsed = time.time() - start_time

    if result.returncode == 0:
        log(f"✓ Completed in {elapsed:.1f}s", Colors.GREEN)
    else:
        log(f"✗ Failed with code {result.returncode}", Colors.RED)
        return False

    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--trait", default="evil")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--skip-lora-training", action="store_true")
    parser.add_argument("--n-per-question", type=int, default=20)
    parser.add_argument("--wandb-project", default="persona-adapters")
    parser.add_argument("--wandb-entity", default=None)
    args = parser.parse_args()

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"{args.trait}_comparison_{time.strftime('%Y%m%d_%H%M%S')}",
        config={
            "model": args.model,
            "trait": args.trait,
            "gpu": args.gpu,
            "n_per_question": args.n_per_question,
        },
        tags=["comparison", "activation-vs-lora", args.trait]
    )

    log(f"\n{'='*60}", Colors.HEADER)
    log("PERSONA VECTORS vs LoRA COMPARISON EXPERIMENT", Colors.BOLD)
    log(f"{'='*60}\n", Colors.HEADER)
    log(f"Model: {args.model}", Colors.CYAN)
    log(f"Trait: {args.trait}", Colors.CYAN)
    log(f"GPU: {args.gpu}", Colors.CYAN)
    log(f"W&B Run: {wandb.run.name}", Colors.CYAN)

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Create output directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results/comparison", exist_ok=True)
    os.makedirs("persona_vectors/Llama-3.1-8B-Instruct", exist_ok=True)
    os.makedirs("persona_loras/Llama-3.1-8B-Instruct", exist_ok=True)
    os.makedirs("eval_persona_extract/Llama-3.1-8B-Instruct", exist_ok=True)

    # ==========================================
    # PHASE 1: Extract Activation Vectors
    # ==========================================
    if not args.skip_extraction:
        log("\n🔵 PHASE 1: Extracting Activation-based Persona Vectors", Colors.HEADER)

        # Step 1a: Generate with positive system prompt
        if not os.path.exists(f"eval_persona_extract/Llama-3.1-8B-Instruct/{args.trait}_pos_instruct.csv"):
            cmd = f"""python -m eval.eval_persona \
                --model {args.model} \
                --trait {args.trait} \
                --output_path eval_persona_extract/Llama-3.1-8B-Instruct/{args.trait}_pos_instruct.csv \
                --persona_instruction_type pos \
                --assistant_name {args.trait} \
                --judge_model gpt-4.1-mini-2025-04-14 \
                --version extract \
                --n_per_question 10"""

            if not run_command(cmd, f"1a. Generating responses with POSITIVE {args.trait} system prompt",
                             f"logs/extract_activation_pos.log"):
                return
        else:
            log(f"✓ Skipping 1a (already exists)", Colors.YELLOW)

        # Step 1b: Generate with negative system prompt
        if not os.path.exists(f"eval_persona_extract/Llama-3.1-8B-Instruct/{args.trait}_neg_instruct.csv"):
            cmd = f"""python -m eval.eval_persona \
                --model {args.model} \
                --trait {args.trait} \
                --output_path eval_persona_extract/Llama-3.1-8B-Instruct/{args.trait}_neg_instruct.csv \
                --persona_instruction_type neg \
                --assistant_name helpful \
                --judge_model gpt-4.1-mini-2025-04-14 \
                --version extract \
                --n_per_question 10"""

            if not run_command(cmd, f"1b. Generating responses with NEGATIVE {args.trait} system prompt",
                             f"logs/extract_activation_neg.log"):
                return
        else:
            log(f"✓ Skipping 1b (already exists)", Colors.YELLOW)

        # Step 1c: Compute activation vector difference
        if not os.path.exists(f"persona_vectors/Llama-3.1-8B-Instruct/{args.trait}_response_avg_diff.pt"):
            cmd = f"""python generate_vec.py \
                --model_name {args.model} \
                --pos_path eval_persona_extract/Llama-3.1-8B-Instruct/{args.trait}_pos_instruct.csv \
                --neg_path eval_persona_extract/Llama-3.1-8B-Instruct/{args.trait}_neg_instruct.csv \
                --trait {args.trait} \
                --save_dir persona_vectors/Llama-3.1-8B-Instruct/"""

            if not run_command(cmd, "1c. Computing activation vector difference",
                             f"logs/generate_activation_vec.log"):
                return
        else:
            log(f"✓ Skipping 1c (already exists)", Colors.YELLOW)
    else:
        log("\n⏭️  Skipping activation vector extraction", Colors.YELLOW)

    # ==========================================
    # PHASE 2: Extract LoRA Adapters
    # ==========================================
    if not args.skip_lora_training:
        log("\n🟣 PHASE 2: Extracting LoRA-based Persona Adapters", Colors.HEADER)

        if not os.path.exists(f"persona_loras/Llama-3.1-8B-Instruct/{args.trait}/{args.trait}_lora_diff.pt"):
            cmd = f"""python extract_persona_lora.py \
                --model {args.model} \
                --trait {args.trait} \
                --output_dir persona_loras/Llama-3.1-8B-Instruct/{args.trait}"""

            if not run_command(cmd, "2. Training contrastive LoRA adapters",
                             f"logs/extract_lora.log"):
                return
        else:
            log(f"✓ Skipping LoRA training (already exists)", Colors.YELLOW)
    else:
        log("\n⏭️  Skipping LoRA adapter training", Colors.YELLOW)

    # ==========================================
    # PHASE 3: Evaluate Activation Steering
    # ==========================================
    log("\n🔵 PHASE 3: Evaluating Activation-based Steering", Colors.HEADER)
    wandb.log({"phase": 3, "phase_name": "activation_evaluation"})

    coefficients = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    activation_vector_path = f"persona_vectors/Llama-3.1-8B-Instruct/{args.trait}_response_avg_diff.pt"

    for coef in coefficients:
        output_path = f"results/comparison/activation_coef_{coef:.1f}.csv"

        if os.path.exists(output_path):
            log(f"✓ Skipping activation coef={coef} (already exists)", Colors.YELLOW)
            continue

        if coef == 0.0:
            # Baseline
            cmd = f"""python -m eval.eval_persona \
                --model {args.model} \
                --trait {args.trait} \
                --output_path {output_path} \
                --judge_model gpt-4.1-mini-2025-04-14 \
                --version eval \
                --n_per_question {args.n_per_question}"""
        else:
            # With steering
            cmd = f"""python -m eval.eval_persona \
                --model {args.model} \
                --trait {args.trait} \
                --output_path {output_path} \
                --judge_model gpt-4.1-mini-2025-04-14 \
                --version eval \
                --steering_type response \
                --coef {coef} \
                --vector_path {activation_vector_path} \
                --layer 16 \
                --n_per_question {args.n_per_question}"""

        if not run_command(cmd, f"3. Activation steering with coef={coef}",
                         f"logs/eval_activation_{coef}.log"):
            return

    # ==========================================
    # PHASE 4: Evaluate LoRA Steering
    # ==========================================
    log("\n🟣 PHASE 4: Evaluating LoRA-based Steering", Colors.HEADER)

    lora_diff_path = f"persona_loras/Llama-3.1-8B-Instruct/{args.trait}/{args.trait}_lora_diff.pt"

    for coef in coefficients:
        output_path = f"results/comparison/lora_coef_{coef:.1f}.csv"

        if os.path.exists(output_path):
            log(f"✓ Skipping LoRA coef={coef} (already exists)", Colors.YELLOW)
            continue

        if coef == 0.0:
            # Baseline (should match activation baseline)
            cmd = f"""python eval_lora_persona.py \
                --model {args.model} \
                --trait {args.trait} \
                --output_path {output_path} \
                --judge_model gpt-4.1-mini-2025-04-14 \
                --version eval \
                --n_per_question {args.n_per_question}"""
        else:
            # With LoRA steering
            cmd = f"""python eval_lora_persona.py \
                --model {args.model} \
                --trait {args.trait} \
                --lora_diff_path {lora_diff_path} \
                --lora_coef {coef} \
                --output_path {output_path} \
                --judge_model gpt-4.1-mini-2025-04-14 \
                --version eval \
                --n_per_question {args.n_per_question}"""

        if not run_command(cmd, f"4. LoRA steering with coef={coef}",
                         f"logs/eval_lora_{coef}.log"):
            return

    # ==========================================
    # PHASE 5: Generate Visualizations
    # ==========================================
    log("\n🎨 PHASE 5: Generating Visualizations", Colors.HEADER)

    cmd = f"""python visualize_comparison.py \
        --results_dir results/comparison \
        --trait {args.trait} \
        --output_dir results/comparison/plots"""

    if not run_command(cmd, "5. Creating Anthropic-style visualizations",
                     f"logs/visualize.log"):
        return

    # ==========================================
    # PHASE 6: Log Results to W&B
    # ==========================================
    log("\n📊 PHASE 6: Logging Results to W&B", Colors.HEADER)

    # Load and log all results
    import pandas as pd
    for coef in coefficients:
        for method in ['activation', 'lora']:
            csv_path = f"results/comparison/{method}_coef_{coef:.1f}.csv"
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                wandb.log({
                    f"{method}/coef_{coef:.1f}/{args.trait}_mean": df[args.trait].mean(),
                    f"{method}/coef_{coef:.1f}/{args.trait}_std": df[args.trait].std(),
                    f"{method}/coef_{coef:.1f}/coherence_mean": df['coherence'].mean(),
                    f"{method}/coef_{coef:.1f}/coherence_std": df['coherence'].std(),
                    f"coefficient": coef
                })

    # Upload plots to W&B
    plot_dir = "results/comparison/plots"
    if os.path.exists(plot_dir):
        for plot_file in os.listdir(plot_dir):
            if plot_file.endswith('.png'):
                wandb.log({
                    f"plots/{plot_file.replace('.png', '')}": wandb.Image(os.path.join(plot_dir, plot_file))
                })

    # Create summary table for W&B
    summary_data = []
    for method in ['activation', 'lora']:
        for coef in coefficients:
            csv_path = f"results/comparison/{method}_coef_{coef:.1f}.csv"
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                summary_data.append({
                    'Method': method,
                    'Coefficient': coef,
                    f'{args.trait}_mean': df[args.trait].mean(),
                    f'{args.trait}_std': df[args.trait].std(),
                    'coherence_mean': df['coherence'].mean(),
                    'coherence_std': df['coherence'].std(),
                })

    if summary_data:
        summary_table = wandb.Table(dataframe=pd.DataFrame(summary_data))
        wandb.log({"summary_table": summary_table})

    # ==========================================
    # DONE!
    # ==========================================
    wandb.finish()

    log(f"\n{'='*60}", Colors.GREEN)
    log("✓ EXPERIMENT COMPLETE!", Colors.BOLD)
    log(f"{'='*60}", Colors.GREEN)
    log(f"\nResults saved to: results/comparison/", Colors.CYAN)
    log(f"Plots saved to: results/comparison/plots/", Colors.CYAN)
    log(f"Logs saved to: logs/", Colors.CYAN)
    log(f"W&B Run: {wandb.run.url}", Colors.CYAN)

if __name__ == "__main__":
    main()
