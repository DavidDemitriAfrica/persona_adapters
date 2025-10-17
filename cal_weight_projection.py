"""
Calculate Weight-Space Projections

This script computes projections of finetuned models onto persona LoRA directions.
This is the weight-space analog of activation projections from the original paper.

Key hypothesis: ΔW · L_persona predicts trait drift during finetuning,
similar to how Δh · v_persona predicts trait in activation space.

Usage:
    python cal_weight_projection.py \
        --original_model meta-llama/Llama-3.1-8B-Instruct \
        --finetuned_model ./ckpt/Llama-3.1-8B-Instruct/evil_misaligned \
        --lora_diff_path persona_loras/Llama-3.1-8B-Instruct/evil/evil_lora_diff.pt \
        --output_path results/weight_projections.json
"""

import argparse
import json
import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm

from lora_utils import WeightProjector, compute_finetuning_weight_delta


def compute_single_projection(
    original_model_path: str,
    finetuned_model_path: str,
    lora_diff_path: str,
    layers: Optional[List[int]] = None
) -> Dict:
    """
    Compute weight projection for a single finetuned model.

    Returns dict with:
        - layer_projections: Per-layer projection values
        - total_projection: Overall projection score
        - weight_shift_norm: Magnitude of weight change
    """
    print(f"\nComputing weight projection...")
    print(f"  Original: {original_model_path}")
    print(f"  Finetuned: {finetuned_model_path}")
    print(f"  LoRA diff: {lora_diff_path}")

    # Load LoRA difference
    lora_diff = torch.load(lora_diff_path, map_location='cpu', weights_only=False)

    # Compute weight changes from finetuning
    print("\nComputing weight deltas...")
    weight_deltas = compute_finetuning_weight_delta(
        original_model_path,
        finetuned_model_path,
        layer_filter=['layers.']  # Focus on transformer layers
    )

    # Compute projections
    print("\nComputing projections...")
    layer_projections = {}
    total_projection = 0.0
    total_norm = 0.0
    count = 0

    # Group by layer
    for key, delta in tqdm(weight_deltas.items()):
        # Extract layer index
        if 'layers.' not in key:
            continue

        try:
            layer_idx = int(key.split('layers.')[1].split('.')[0])
        except (IndexError, ValueError):
            continue

        if layers and layer_idx not in layers:
            continue

        # Find corresponding LoRA weight
        # Convert model key to LoRA key format
        lora_key = f"base_model.{key.replace('.weight', '.lora_A.weight')}"

        if lora_key in lora_diff:
            lora_b_key = lora_key.replace('lora_A', 'lora_B')
            if lora_b_key not in lora_diff:
                continue

            # Compute full LoRA direction
            lora_A = lora_diff[lora_key]
            lora_B = lora_diff[lora_b_key]
            lora_full = (lora_B @ lora_A).flatten()

            # Compute projection: (ΔW · L) / ||L||
            delta_flat = delta.flatten()
            projection = (delta_flat @ lora_full) / (lora_full.norm() + 1e-8)
            projection_val = projection.item()

            # Track per-layer
            if layer_idx not in layer_projections:
                layer_projections[layer_idx] = []
            layer_projections[layer_idx].append(projection_val)

            # Accumulate totals
            total_projection += projection_val
            total_norm += delta.norm().item()
            count += 1

    # Aggregate layer projections
    layer_proj_summary = {
        layer: sum(projs) / len(projs)
        for layer, projs in layer_projections.items()
    }

    # Compute overall metrics
    avg_projection = total_projection / count if count > 0 else 0.0

    results = {
        "layer_projections": layer_proj_summary,
        "total_projection": total_projection,
        "average_projection": avg_projection,
        "weight_shift_norm": total_norm,
        "num_parameters": count
    }

    print(f"\n{'='*60}")
    print(f"Projection Results:")
    print(f"  Average projection: {avg_projection:.4f}")
    print(f"  Total projection: {total_projection:.4f}")
    print(f"  Weight shift norm: {total_norm:.4f}")
    print(f"  Parameters analyzed: {count}")
    print(f"{'='*60}\n")

    return results


def compute_projection_dataset(
    original_model: str,
    finetuned_models: List[str],
    lora_diff_path: str,
    behavior_scores_path: Optional[str] = None,
    output_path: str = "weight_projections.json"
):
    """
    Compute projections for multiple finetuned models and correlate with behavior.

    This replicates Figure 6 from the paper in weight space.

    Args:
        original_model: Base model path
        finetuned_models: List of finetuned model paths
        lora_diff_path: Path to persona LoRA difference
        behavior_scores_path: Optional CSV with behavior scores for correlation
        output_path: Where to save results
    """
    all_results = []

    for ft_model in finetuned_models:
        model_name = os.path.basename(ft_model)

        try:
            result = compute_single_projection(
                original_model,
                ft_model,
                lora_diff_path
            )

            result["model_name"] = model_name
            result["model_path"] = ft_model
            all_results.append(result)

        except Exception as e:
            print(f"Error processing {ft_model}: {e}")
            continue

    # Save results
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")

    # If behavior scores provided, compute correlation
    if behavior_scores_path and os.path.exists(behavior_scores_path):
        compute_projection_correlation(output_path, behavior_scores_path)

    return all_results


def compute_projection_correlation(projection_path: str, behavior_scores_path: str):
    """
    Compute correlation between weight projections and behavior scores.

    This tests the key hypothesis: ΔW · L_persona correlates with trait expression.
    """
    print("\nComputing correlation with behavior scores...")

    # Load projections
    with open(projection_path, 'r') as f:
        projections = json.load(f)

    proj_df = pd.DataFrame(projections)

    # Load behavior scores
    behavior_df = pd.read_csv(behavior_scores_path)

    # Merge on model name
    merged = proj_df.merge(
        behavior_df,
        left_on='model_name',
        right_on='model_name',
        how='inner'
    )

    if len(merged) == 0:
        print("Warning: No matching models found between projections and behavior scores")
        return

    # Compute correlation
    from scipy.stats import pearsonr, spearmanr

    trait_cols = [col for col in behavior_df.columns if col not in ['model_name', 'coherence']]

    print(f"\n{'='*60}")
    print("Correlation Analysis: Weight Projection vs Behavior")
    print(f"{'='*60}")

    for trait in trait_cols:
        if trait in merged.columns:
            pearson_r, pearson_p = pearsonr(merged['average_projection'], merged[trait])
            spearman_r, spearman_p = spearmanr(merged['average_projection'], merged[trait])

            print(f"\n{trait.upper()}:")
            print(f"  Pearson r: {pearson_r:.3f} (p={pearson_p:.4f})")
            print(f"  Spearman ρ: {spearman_r:.3f} (p={spearman_p:.4f})")

    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Calculate weight-space projections")
    parser.add_argument("--original_model", type=str, required=True, help="Original base model")
    parser.add_argument("--finetuned_model", type=str, help="Single finetuned model (optional)")
    parser.add_argument("--finetuned_models_dir", type=str, help="Directory of finetuned models (optional)")
    parser.add_argument("--lora_diff_path", type=str, required=True, help="Path to persona LoRA difference")
    parser.add_argument("--behavior_scores", type=str, help="CSV with behavior scores for correlation")
    parser.add_argument("--output_path", type=str, default="weight_projections.json", help="Output file")
    parser.add_argument("--layers", type=int, nargs='+', help="Specific layers to analyze")

    args = parser.parse_args()

    # Single model or multiple?
    if args.finetuned_model:
        # Single model
        results = compute_single_projection(
            args.original_model,
            args.finetuned_model,
            args.lora_diff_path,
            layers=args.layers
        )

        # Save
        with open(args.output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✓ Results saved to {args.output_path}")

    elif args.finetuned_models_dir:
        # Multiple models
        import glob
        finetuned_models = glob.glob(os.path.join(args.finetuned_models_dir, "*"))
        finetuned_models = [m for m in finetuned_models if os.path.isdir(m)]

        print(f"Found {len(finetuned_models)} finetuned models")

        compute_projection_dataset(
            args.original_model,
            finetuned_models,
            args.lora_diff_path,
            behavior_scores_path=args.behavior_scores,
            output_path=args.output_path
        )

    else:
        print("Error: Must specify either --finetuned_model or --finetuned_models_dir")
        return


if __name__ == "__main__":
    main()
