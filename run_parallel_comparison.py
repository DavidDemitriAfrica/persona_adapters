"""
Multi-GPU Parallel Comparison Experiment: Activation vs LoRA Persona Steering

This script parallelizes the evaluation across multiple GPUs to maximize hardware utilization.

Key parallelization strategy:
- Phase 1-2: Extract vectors/LoRA on GPU 0 (sequential, required)
- Phase 3-4: Evaluate 6 coefficients x 2 methods = 12 tasks across 4 GPUs (parallel)

With 4 GPUs, we can run 4 evaluation tasks simultaneously.
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
import argparse
import wandb
from concurrent.futures import ProcessPoolExecutor, as_completed
from threading import Lock

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

print_lock = Lock()

def log(msg, color=Colors.CYAN):
    with print_lock:
        print(f"{color}{msg}{Colors.END}")

def run_command(cmd, description, log_file=None, gpu_id=None):
    """Run a command and log output."""
    prefix = f"[GPU {gpu_id}] " if gpu_id is not None else ""
    log(f"\n{prefix}{'='*60}", Colors.HEADER)
    log(f"{prefix}{description}", Colors.BOLD)
    log(f"{prefix}{'='*60}", Colors.HEADER)
    log(f"{prefix}Command: {cmd}", Colors.BLUE)

    start_time = time.time()

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'w') as f:
            f.write(f"Command: {cmd}\n\n")
            result = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
    else:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    elapsed = time.time() - start_time

    if result.returncode == 0:
        log(f"{prefix}✓ Completed in {elapsed:.1f}s", Colors.GREEN)
        return True
    else:
        log(f"{prefix}✗ Failed with code {result.returncode}", Colors.RED)
        if not log_file and result.stderr:
            log(f"{prefix}Error: {result.stderr[:500]}", Colors.RED)
        return False

def run_evaluation_task(task_info):
    """Run a single evaluation task on a specific GPU."""
    method, coef, model, trait, n_per_question, gpu_id, output_path, log_path = task_info

    # Check if already exists
    if os.path.exists(output_path):
        log(f"[GPU {gpu_id}] ✓ Skipping {method} coef={coef} (already exists)", Colors.YELLOW)
        return True, method, coef, gpu_id

    # Set GPU visibility for this process
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    if method == 'activation':
        activation_vector_path = f"persona_vectors/Llama-3.1-8B-Instruct/{trait}_response_avg_diff.pt"

        if coef == 0.0:
            cmd = f"""python -m eval.eval_persona \
                --model {model} \
                --trait {trait} \
                --output_path {output_path} \
                --judge_model gpt-4.1-mini-2025-04-14 \
                --version eval \
                --n_per_question {n_per_question}"""
        else:
            cmd = f"""python -m eval.eval_persona \
                --model {model} \
                --trait {trait} \
                --output_path {output_path} \
                --judge_model gpt-4.1-mini-2025-04-14 \
                --version eval \
                --steering_type response \
                --coef {coef} \
                --vector_path {activation_vector_path} \
                --layer 16 \
                --n_per_question {n_per_question}"""

    else:  # lora
        lora_diff_path = f"persona_loras/Llama-3.1-8B-Instruct/{trait}/{trait}_lora_diff.pt"

        if coef == 0.0:
            cmd = f"""python eval_lora_persona.py \
                --model {model} \
                --trait {trait} \
                --output_path {output_path} \
                --judge_model gpt-4.1-mini-2025-04-14 \
                --version eval \
                --n_per_question {n_per_question}"""
        else:
            cmd = f"""python eval_lora_persona.py \
                --model {model} \
                --trait {trait} \
                --lora_diff_path {lora_diff_path} \
                --lora_coef {coef} \
                --output_path {output_path} \
                --judge_model gpt-4.1-mini-2025-04-14 \
                --version eval \
                --n_per_question {n_per_question}"""

    description = f"{method.upper()} steering with coef={coef}"
    success = run_command(cmd, description, log_path, gpu_id)

    return success, method, coef, gpu_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--trait", default="evil")
    parser.add_argument("--gpus", type=str, default="0,1,2,3", help="Comma-separated GPU IDs")
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--skip-lora-training", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoints")
    parser.add_argument("--n-per-question", type=int, default=20)
    parser.add_argument("--wandb-project", default="persona-adapters")
    parser.add_argument("--wandb-entity", default=None)
    args = parser.parse_args()

    # Parse GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    log(f"Using GPUs: {gpu_ids}", Colors.CYAN)

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"{args.trait}_parallel_comparison_{time.strftime('%Y%m%d_%H%M%S')}",
        config={
            "model": args.model,
            "trait": args.trait,
            "gpus": gpu_ids,
            "n_per_question": args.n_per_question,
            "parallel": True,
        },
        tags=["comparison", "activation-vs-lora", args.trait, "parallel"]
    )

    log(f"\n{'='*60}", Colors.HEADER)
    log("PARALLEL PERSONA VECTORS vs LoRA COMPARISON", Colors.BOLD)
    log(f"{'='*60}\n", Colors.HEADER)
    log(f"Model: {args.model}", Colors.CYAN)
    log(f"Trait: {args.trait}", Colors.CYAN)
    log(f"GPUs: {gpu_ids} ({len(gpu_ids)} parallel workers)", Colors.CYAN)
    log(f"W&B Run: {wandb.run.name}", Colors.CYAN)

    # Create output directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results/comparison", exist_ok=True)
    os.makedirs("persona_vectors/Llama-3.1-8B-Instruct", exist_ok=True)
    os.makedirs("persona_loras/Llama-3.1-8B-Instruct", exist_ok=True)
    os.makedirs("eval_persona_extract/Llama-3.1-8B-Instruct", exist_ok=True)

    # ==========================================
    # PHASE 1: Extract Activation Vectors (Sequential on GPU 0)
    # ==========================================
    if not args.skip_extraction and not (args.resume and os.path.exists(f"persona_vectors/Llama-3.1-8B-Instruct/{args.trait}_response_avg_diff.pt")):
        log("\n🔵 PHASE 1: Extracting Activation-based Persona Vectors", Colors.HEADER)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ids[0])

        # Step 1a: Generate with positive system prompt
        checkpoint_path = f"eval_persona_extract/Llama-3.1-8B-Instruct/.checkpoints/{args.trait}_pos_instruct.responses.pt"
        if not os.path.exists(f"eval_persona_extract/Llama-3.1-8B-Instruct/{args.trait}_pos_instruct.csv") or (args.resume and os.path.exists(checkpoint_path)):
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
                             f"logs/extract_activation_pos.log", gpu_ids[0]):
                return
        else:
            log(f"✓ Skipping 1a (already exists)", Colors.YELLOW)

        # Step 1b: Generate with negative system prompt
        checkpoint_path = f"eval_persona_extract/Llama-3.1-8B-Instruct/.checkpoints/{args.trait}_neg_instruct.responses.pt"
        if not os.path.exists(f"eval_persona_extract/Llama-3.1-8B-Instruct/{args.trait}_neg_instruct.csv") or (args.resume and os.path.exists(checkpoint_path)):
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
                             f"logs/extract_activation_neg.log", gpu_ids[0]):
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
                             f"logs/generate_activation_vec.log", gpu_ids[0]):
                return
        else:
            log(f"✓ Skipping 1c (already exists)", Colors.YELLOW)
    elif args.resume and os.path.exists(f"persona_vectors/Llama-3.1-8B-Instruct/{args.trait}_response_avg_diff.pt"):
        log("\n⏭️  Phase 1 already complete (resuming from checkpoint)", Colors.YELLOW)
    else:
        log("\n⏭️  Skipping activation vector extraction", Colors.YELLOW)

    # ==========================================
    # PHASE 2: Extract LoRA Adapters (Sequential on GPU 0)
    # ==========================================
    if not args.skip_lora_training and not (args.resume and os.path.exists(f"persona_loras/Llama-3.1-8B-Instruct/{args.trait}/{args.trait}_lora_diff.pt")):
        log("\n🟣 PHASE 2: Extracting LoRA-based Persona Adapters", Colors.HEADER)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ids[0])

        if not os.path.exists(f"persona_loras/Llama-3.1-8B-Instruct/{args.trait}/{args.trait}_lora_diff.pt"):
            cmd = f"""python extract_persona_lora.py \
                --model {args.model} \
                --trait {args.trait} \
                --output_dir persona_loras/Llama-3.1-8B-Instruct/{args.trait}"""

            if not run_command(cmd, "2. Training contrastive LoRA adapters",
                             f"logs/extract_lora.log", gpu_ids[0]):
                return
        else:
            log(f"✓ Skipping LoRA training (already exists)", Colors.YELLOW)
    elif args.resume and os.path.exists(f"persona_loras/Llama-3.1-8B-Instruct/{args.trait}/{args.trait}_lora_diff.pt"):
        log("\n⏭️  Phase 2 already complete (resuming from checkpoint)", Colors.YELLOW)
    else:
        log("\n⏭️  Skipping LoRA adapter training", Colors.YELLOW)

    # ==========================================
    # PHASE 3+4: Parallel Evaluation Across All GPUs
    # ==========================================
    log(f"\n🚀 PHASE 3+4: PARALLEL Evaluation Across {len(gpu_ids)} GPUs", Colors.HEADER)

    coefficients = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]

    # Create all evaluation tasks
    eval_tasks = []
    for method in ['activation', 'lora']:
        for coef in coefficients:
            output_path = f"results/comparison/{method}_coef_{coef:.1f}.csv"
            log_path = f"logs/eval_{method}_{coef}.log"

            # We'll assign GPUs in round-robin fashion via ProcessPoolExecutor
            task_info = (method, coef, args.model, args.trait, args.n_per_question,
                        None, output_path, log_path)  # GPU ID will be assigned by executor
            eval_tasks.append(task_info)

    log(f"Created {len(eval_tasks)} evaluation tasks to distribute across {len(gpu_ids)} GPUs", Colors.CYAN)

    # Run tasks in parallel using ProcessPoolExecutor
    # Assign GPUs in round-robin fashion
    tasks_with_gpus = []
    for i, task in enumerate(eval_tasks):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        task_with_gpu = task[:5] + (gpu_id,) + task[6:]
        tasks_with_gpus.append(task_with_gpu)

    # Execute in parallel
    start_time = time.time()
    results = []
    failed = []

    with ProcessPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = {executor.submit(run_evaluation_task, task): task for task in tasks_with_gpus}

        for future in as_completed(futures):
            task = futures[future]
            try:
                success, method, coef, gpu_id = future.result()
                if success:
                    results.append((method, coef))
                else:
                    failed.append((method, coef))
            except Exception as e:
                log(f"Task {task[0]} coef={task[1]} raised exception: {e}", Colors.RED)
                failed.append((task[0], task[1]))

    elapsed = time.time() - start_time
    log(f"\n{'='*60}", Colors.GREEN)
    log(f"Parallel evaluation completed in {elapsed:.1f}s", Colors.GREEN)
    log(f"Success: {len(results)}, Failed: {len(failed)}", Colors.CYAN)
    log(f"{'='*60}", Colors.GREEN)

    if failed:
        log(f"\nFailed tasks:", Colors.RED)
        for method, coef in failed:
            log(f"  - {method} coef={coef}", Colors.RED)
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
    log("✓ PARALLEL EXPERIMENT COMPLETE!", Colors.BOLD)
    log(f"{'='*60}", Colors.GREEN)
    log(f"\nResults saved to: results/comparison/", Colors.CYAN)
    log(f"Plots saved to: results/comparison/plots/", Colors.CYAN)
    log(f"Logs saved to: logs/", Colors.CYAN)
    log(f"W&B Run: {wandb.run.url}", Colors.CYAN)
    log(f"\nSpeedup: ~{len(gpu_ids)}x faster for evaluation phase", Colors.GREEN)

if __name__ == "__main__":
    main()
