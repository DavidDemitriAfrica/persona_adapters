"""
Compare Activation-based vs LoRA-based Persona Steering

This script provides a comprehensive comparison between the two approaches:
1. Activation steering (original): h' = h + α·v
2. LoRA steering (ours): W' = W + α·ΔW_lora

Key research questions:
- Do both methods achieve similar trait control?
- Which is more efficient (latency, memory)?
- Can we bridge the two spaces theoretically?

Usage:
    python compare_steering_methods.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --trait evil \
        --activation_vector_path persona_vectors/Llama-3.1-8B-Instruct/evil_response_avg_diff.pt \
        --lora_diff_path persona_loras/Llama-3.1-8B-Instruct/evil/evil_lora_diff.pt \
        --coefficients 0.0 0.5 1.0 1.5 2.0 \
        --output_dir comparison_results/
"""

import argparse
import os
import time
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from tqdm import tqdm
import asyncio

from transformers import AutoModelForCausalLM, AutoTokenizer
from eval_lora_persona import eval_batched_lora, load_persona_questions
from activation_steer import ActivationSteerer
from lora_utils import LoRAMerger


def measure_latency(
    model,
    tokenizer,
    test_prompts: List[str],
    method: str,
    vector_path: str = None,
    coef: float = 1.0,
    layer: int = 16,
    num_runs: int = 10
) -> Dict:
    """
    Measure inference latency for different steering methods.

    Returns:
        Dict with timing statistics
    """
    print(f"\nMeasuring latency for {method} steering...")

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare inputs
    inputs = tokenizer(test_prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    latencies = []

    for run in tqdm(range(num_runs), desc=f"Latency test ({method})"):
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        start_time = time.time()

        if method == "activation":
            vector = torch.load(vector_path, weights_only=False)[layer]
            with ActivationSteerer(model, vector, coeff=coef, layer_idx=layer-1, positions="response"):
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=100)

        elif method == "lora_premerge":
            merger = LoRAMerger(model)
            lora_diff = torch.load(vector_path, map_location='cpu', weights_only=False)
            merger.merge_lora_diff(lora_diff, coefficient=coef)

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=100)

            merger.restore_original_weights()

        elif method == "baseline":
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=100)

        else:
            raise ValueError(f"Unknown method: {method}")

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        latency = time.time() - start_time
        latencies.append(latency)

    return {
        "method": method,
        "mean_latency": sum(latencies) / len(latencies),
        "std_latency": torch.tensor(latencies).std().item(),
        "min_latency": min(latencies),
        "max_latency": max(latencies)
    }


def compare_behavior_vs_coefficient(
    model_name: str,
    trait: str,
    activation_vector_path: str,
    lora_diff_path: str,
    coefficients: List[float],
    output_dir: str,
    layer: int = 16,
    n_per_question: int = 10
):
    """
    Compare how trait scores vary with coefficient for both methods.

    This replicates Figure 3 from the paper for both activation and LoRA steering.
    """
    print(f"\n{'='*60}")
    print("Comparing behavior vs coefficient")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load questions
    questions = load_persona_questions(trait, temperature=1.0, version="eval")

    results = []

    for coef in coefficients:
        print(f"\n--- Testing coefficient: {coef} ---")

        # Test activation steering
        print("Testing activation steering...")
        activation_output = os.path.join(output_dir, f"activation_coef_{coef:.1f}.csv")

        if not os.path.exists(activation_output):
            # Use original eval_persona.py logic with activation steering
            from eval.eval_persona import eval_batched
            vector = torch.load(activation_vector_path, weights_only=False)

            outputs_list = asyncio.run(eval_batched(
                questions, model, tokenizer,
                coef=coef, vector=vector, layer=layer,
                n_per_question=n_per_question,
                steering_type="response"
            ))

            outputs = pd.concat(outputs_list)
            outputs.to_csv(activation_output, index=False)
        else:
            outputs = pd.read_csv(activation_output)

        results.append({
            "method": "activation",
            "coefficient": coef,
            "trait_score": outputs[trait].mean(),
            "trait_std": outputs[trait].std(),
            "coherence": outputs["coherence"].mean()
        })

        # Test LoRA steering
        print("Testing LoRA steering...")
        lora_output = os.path.join(output_dir, f"lora_coef_{coef:.1f}.csv")

        if not os.path.exists(lora_output):
            outputs_list = asyncio.run(eval_batched_lora(
                questions, model, tokenizer,
                lora_diff_path=lora_diff_path,
                lora_coef=coef,
                n_per_question=n_per_question
            ))

            outputs = pd.concat(outputs_list)
            outputs.to_csv(lora_output, index=False)
        else:
            outputs = pd.read_csv(lora_output)

        results.append({
            "method": "lora",
            "coefficient": coef,
            "trait_score": outputs[trait].mean(),
            "trait_std": outputs[trait].std(),
            "coherence": outputs["coherence"].mean()
        })

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "comparison_results.csv"), index=False)

    # Plot comparison
    plot_comparison(results_df, trait, output_dir)

    return results_df


def plot_comparison(results_df: pd.DataFrame, trait: str, output_dir: str):
    """Create comparison plots."""
    sns.set_style("whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Trait score vs coefficient
    ax1 = axes[0]
    for method in ["activation", "lora"]:
        data = results_df[results_df["method"] == method]
        ax1.plot(data["coefficient"], data["trait_score"],
                marker='o', label=method.capitalize(), linewidth=2)
        ax1.fill_between(data["coefficient"],
                         data["trait_score"] - data["trait_std"],
                         data["trait_score"] + data["trait_std"],
                         alpha=0.2)

    ax1.set_xlabel("Steering Coefficient (α)", fontsize=12)
    ax1.set_ylabel(f"{trait.capitalize()} Score", fontsize=12)
    ax1.set_title(f"{trait.capitalize()} Expression vs Coefficient", fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Coherence vs coefficient
    ax2 = axes[1]
    for method in ["activation", "lora"]:
        data = results_df[results_df["method"] == method]
        ax2.plot(data["coefficient"], data["coherence"],
                marker='s', label=method.capitalize(), linewidth=2)

    ax2.set_xlabel("Steering Coefficient (α)", fontsize=12)
    ax2.set_ylabel("Coherence Score", fontsize=12)
    ax2.set_title("Coherence vs Coefficient", fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=75, color='r', linestyle='--', alpha=0.5, label='Threshold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_plot.png"), dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to {output_dir}/comparison_plot.png")


def analyze_space_relationship(
    model_name: str,
    activation_vector_path: str,
    lora_diff_path: str,
    layer: int = 16,
    output_path: str = "space_analysis.json"
):
    """
    Analyze the relationship between activation vectors and LoRA weight changes.

    Tests the hypothesis: v ≈ L @ h (forward pass approximation)
    """
    print(f"\n{'='*60}")
    print("Analyzing relationship between activation and weight space")
    print(f"{'='*60}\n")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use fp32 for precision
        device_map="cpu"  # CPU for analysis
    )

    # Load vectors
    activation_vector = torch.load(activation_vector_path, weights_only=False)[layer]
    lora_diff = torch.load(lora_diff_path, map_location='cpu', weights_only=False)

    # Analyze per-layer
    print(f"\nAnalyzing layer {layer}...")

    # Get typical hidden states (use random inputs as proxy)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_input = tokenizer("This is a test.", return_tensors="pt")

    with torch.no_grad():
        outputs = model(**test_input, output_hidden_states=True)
        hidden_state = outputs.hidden_states[layer].mean(dim=1).squeeze()  # [hidden_dim]

    # Find LoRA matrices for this layer
    layer_lora_keys = [k for k in lora_diff.keys() if f"layers.{layer}." in k and "lora_A" in k]

    analysis_results = {
        "layer": layer,
        "activation_norm": activation_vector.norm().item(),
        "hidden_state_norm": hidden_state.norm().item(),
        "module_analysis": []
    }

    for lora_a_key in layer_lora_keys:
        lora_b_key = lora_a_key.replace("lora_A", "lora_B")
        if lora_b_key not in lora_diff:
            continue

        # Compute full LoRA: ΔW = B @ A
        lora_A = lora_diff[lora_a_key]
        lora_B = lora_diff[lora_b_key]
        delta_W = lora_B @ lora_A

        # Compute: ΔW @ h (what LoRA adds to hidden state)
        lora_contribution = delta_W @ hidden_state

        # Compare to activation vector
        similarity = torch.nn.functional.cosine_similarity(
            activation_vector.unsqueeze(0),
            lora_contribution.unsqueeze(0)
        ).item()

        norm_diff = (activation_vector - lora_contribution).norm().item()

        module_name = lora_a_key.split(".")[-3]  # e.g., "q_proj", "mlp"

        analysis_results["module_analysis"].append({
            "module": module_name,
            "cosine_similarity": similarity,
            "norm_difference": norm_diff,
            "lora_contribution_norm": lora_contribution.norm().item()
        })

        print(f"  {module_name}:")
        print(f"    Cosine similarity: {similarity:.4f}")
        print(f"    Norm difference: {norm_diff:.4f}")

    # Save analysis
    with open(output_path, 'w') as f:
        json.dump(analysis_results, f, indent=2)

    print(f"\n✓ Space analysis saved to {output_path}")

    return analysis_results


def main():
    parser = argparse.ArgumentParser(description="Compare activation vs LoRA steering")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--trait", type=str, required=True, help="Trait to test")
    parser.add_argument("--activation_vector_path", type=str, required=True, help="Path to activation vector")
    parser.add_argument("--lora_diff_path", type=str, required=True, help="Path to LoRA difference")
    parser.add_argument("--coefficients", type=float, nargs='+', default=[0.0, 0.5, 1.0, 1.5, 2.0],
                        help="Coefficients to test")
    parser.add_argument("--layer", type=int, default=16, help="Layer to use")
    parser.add_argument("--output_dir", type=str, default="comparison_results", help="Output directory")
    parser.add_argument("--n_per_question", type=int, default=10, help="Samples per question")
    parser.add_argument("--test_latency", action="store_true", help="Run latency tests")
    parser.add_argument("--analyze_spaces", action="store_true", help="Analyze space relationships")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Main comparison: behavior vs coefficient
    print("\n" + "="*60)
    print("COMPARISON: ACTIVATION VS LORA STEERING")
    print("="*60)

    results_df = compare_behavior_vs_coefficient(
        args.model,
        args.trait,
        args.activation_vector_path,
        args.lora_diff_path,
        args.coefficients,
        args.output_dir,
        layer=args.layer,
        n_per_question=args.n_per_question
    )

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(results_df.to_string(index=False))

    # Optional: latency tests
    if args.test_latency:
        print("\n" + "="*60)
        print("LATENCY TESTS")
        print("="*60)

        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        test_prompts = ["This is a test prompt."] * 4

        baseline = measure_latency(model, tokenizer, test_prompts, "baseline")
        activation = measure_latency(model, tokenizer, test_prompts, "activation",
                                     vector_path=args.activation_vector_path, layer=args.layer)
        lora = measure_latency(model, tokenizer, test_prompts, "lora_premerge",
                              vector_path=args.lora_diff_path)

        latency_results = pd.DataFrame([baseline, activation, lora])
        print("\n" + latency_results.to_string(index=False))

        latency_results.to_csv(os.path.join(args.output_dir, "latency_comparison.csv"), index=False)

    # Optional: space analysis
    if args.analyze_spaces:
        analyze_space_relationship(
            args.model,
            args.activation_vector_path,
            args.lora_diff_path,
            layer=args.layer,
            output_path=os.path.join(args.output_dir, "space_analysis.json")
        )


if __name__ == "__main__":
    main()
