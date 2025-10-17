"""
LoRA Utilities for Persona Control

This module provides utilities for:
1. Merging LoRA adapters with base models
2. Computing weight-space projections
3. Dynamic LoRA steering during inference
4. Analyzing LoRA weight patterns
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
import copy


class LoRAMerger:
    """
    Handles merging LoRA adapters with base models at different coefficients.

    This enables inference-time steering by merging persona LoRAs:
    W' = W + α * ΔW_lora

    where α controls the strength of the persona.
    """

    def __init__(self, base_model: nn.Module):
        self.base_model = base_model
        self.original_weights = {}
        self._save_original_weights()

    def _save_original_weights(self):
        """Save original weights for later restoration."""
        for name, param in self.base_model.named_parameters():
            if any(key in name for key in ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                            'gate_proj', 'up_proj', 'down_proj']):
                self.original_weights[name] = param.data.clone()

    def merge_lora_diff(self, lora_diff: Dict[str, torch.Tensor], coefficient: float = 1.0):
        """
        Merge LoRA difference weights into the base model with a given coefficient.

        Args:
            lora_diff: Dictionary of LoRA weight differences (from extract_persona_lora.py)
            coefficient: Scaling factor for the merge (α in the paper)
        """
        print(f"Merging LoRA with coefficient α={coefficient}")

        # Parse LoRA keys and merge
        merged_count = 0
        for lora_key, lora_weight in lora_diff.items():
            # LoRA keys look like: "base_model.model.layers.0.self_attn.q_proj.lora_A.weight"
            # We need to find the corresponding base model parameter

            if 'lora_A' in lora_key:
                # Get the base parameter name
                base_key = lora_key.replace('base_model.', '').replace('.lora_A.weight', '.weight')

                # Find corresponding lora_B
                lora_b_key = lora_key.replace('lora_A', 'lora_B')
                if lora_b_key not in lora_diff:
                    continue

                lora_A = lora_weight  # shape: (r, in_features)
                lora_B = lora_diff[lora_b_key]  # shape: (out_features, r)

                # Compute full LoRA update: ΔW = B @ A
                delta_W = (lora_B @ lora_A) * coefficient

                # Apply to base model
                try:
                    base_param = dict(self.base_model.named_parameters())[base_key]
                    base_param.data += delta_W.to(base_param.device).to(base_param.dtype)
                    merged_count += 1
                except KeyError:
                    print(f"Warning: Could not find base parameter for {base_key}")

        print(f"✓ Merged {merged_count} LoRA layers")

    def restore_original_weights(self):
        """Restore model to original pre-merge state."""
        for name, param in self.base_model.named_parameters():
            if name in self.original_weights:
                param.data = self.original_weights[name].clone()
        print("✓ Restored original weights")


class WeightProjector:
    """
    Computes projections of model weights onto persona LoRA directions.

    This is the weight-space analog of activation projections in the original paper.
    Key idea: projection strength predicts persona expression.
    """

    @staticmethod
    def compute_projection(
        model_weights: Dict[str, torch.Tensor],
        lora_direction: Dict[str, torch.Tensor],
        layer_filter: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute projection of model weights onto LoRA direction.

        Returns:
            Dictionary mapping layer names to projection values
        """
        projections = {}

        for key in model_weights.keys():
            if layer_filter and not any(f in key for f in layer_filter):
                continue

            if key in lora_direction:
                W = model_weights[key].flatten()
                L = lora_direction[key].flatten()

                # Normalized projection: (W · L) / ||L||
                projection = (W @ L) / (L.norm() + 1e-8)
                projections[key] = projection.item()

        return projections

    @staticmethod
    def compute_layer_projections(
        model: nn.Module,
        lora_diff_path: str,
        num_layers: int
    ) -> torch.Tensor:
        """
        Compute per-layer projection values.

        Returns:
            Tensor of shape [num_layers] with projection values
        """
        lora_diff = torch.load(lora_diff_path, map_location='cpu', weights_only=False)

        layer_projections = torch.zeros(num_layers)

        for layer_idx in range(num_layers):
            layer_proj = 0.0
            count = 0

            # Aggregate projections across all modules in this layer
            for key, lora_weight in lora_diff.items():
                if f"layers.{layer_idx}." in key and 'lora_A' in key:
                    # Get model weight
                    base_key = key.replace('base_model.', '').replace('.lora_A.weight', '.weight')
                    try:
                        model_weight = dict(model.named_parameters())[base_key].data

                        # Compute LoRA update
                        lora_b_key = key.replace('lora_A', 'lora_B')
                        if lora_b_key in lora_diff:
                            lora_A = lora_weight
                            lora_B = lora_diff[lora_b_key]
                            delta_W = (lora_B @ lora_A).flatten()

                            # Project
                            W_flat = model_weight.flatten()
                            proj = (W_flat @ delta_W) / (delta_W.norm() + 1e-8)
                            layer_proj += proj.item()
                            count += 1
                    except KeyError:
                        continue

            if count > 0:
                layer_projections[layer_idx] = layer_proj / count

        return layer_projections


class LoRASteerer:
    """
    Apply LoRA-based steering during model inference.

    Two modes:
    1. Pre-merge: Merge LoRA before inference (faster)
    2. Dynamic: Apply LoRA on-the-fly during forward pass
    """

    def __init__(
        self,
        model: nn.Module,
        lora_diff_path: str,
        coefficient: float = 1.0,
        mode: str = "pre_merge"  # or "dynamic"
    ):
        self.model = model
        self.coefficient = coefficient
        self.mode = mode
        self.lora_diff = torch.load(lora_diff_path, map_location='cpu', weights_only=False)
        self.merger = LoRAMerger(model)

        if mode == "pre_merge":
            self.merger.merge_lora_diff(self.lora_diff, coefficient)
        elif mode == "dynamic":
            raise NotImplementedError("Dynamic LoRA steering not yet implemented")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.mode == "pre_merge":
            self.merger.restore_original_weights()


def load_lora_adapter(model_path: str, adapter_path: str) -> Tuple[nn.Module, AutoTokenizer]:
    """
    Load a base model with a LoRA adapter attached.

    Args:
        model_path: Path to base model
        adapter_path: Path to LoRA adapter

    Returns:
        (model, tokenizer) tuple
    """
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer


def compute_finetuning_weight_delta(
    original_model_path: str,
    finetuned_model_path: str,
    layer_filter: Optional[List[str]] = None
) -> Dict[str, torch.Tensor]:
    """
    Compute weight changes from finetuning: ΔW = W_finetuned - W_original

    This is used to analyze how finetuning shifts models in weight space.
    """
    print(f"Loading original model from {original_model_path}...")
    original = AutoModelForCausalLM.from_pretrained(original_model_path, torch_dtype=torch.float32)

    print(f"Loading finetuned model from {finetuned_model_path}...")
    finetuned = AutoModelForCausalLM.from_pretrained(finetuned_model_path, torch_dtype=torch.float32)

    deltas = {}
    for (name, orig_param), (_, ft_param) in zip(
        original.named_parameters(),
        finetuned.named_parameters()
    ):
        if layer_filter and not any(f in name for f in layer_filter):
            continue

        delta = ft_param.data - orig_param.data
        deltas[name] = delta.cpu()
        print(f"  {name}: Δnorm={delta.norm().item():.4f}")

    return deltas


def analyze_lora_singular_values(lora_diff_path: str, top_k: int = 10):
    """
    Perform SVD on LoRA difference matrices to understand their structure.

    This helps answer: Can we compress persona LoRAs to lower rank?
    """
    lora_diff = torch.load(lora_diff_path, map_location='cpu', weights_only=False)

    print(f"\nSingular Value Analysis of LoRA Difference")
    print("=" * 60)

    for key, weight in lora_diff.items():
        if 'lora_A' in key and weight.ndim == 2:
            # Get corresponding lora_B
            lora_b_key = key.replace('lora_A', 'lora_B')
            if lora_b_key not in lora_diff:
                continue

            # Compute full LoRA matrix
            lora_A = weight
            lora_B = lora_diff[lora_b_key]
            full_lora = lora_B @ lora_A

            # SVD
            U, S, V = torch.svd(full_lora)

            # Analyze top singular values
            total_energy = S.sum()
            top_k_energy = S[:top_k].sum()
            ratio = (top_k_energy / total_energy).item()

            print(f"\n{key}")
            print(f"  Shape: {full_lora.shape}")
            print(f"  Top {top_k} singular values: {S[:top_k].tolist()}")
            print(f"  Energy ratio (top-{top_k}): {ratio:.2%}")


if __name__ == "__main__":
    # Example usage
    print("LoRA utilities module loaded successfully")
    print("\nAvailable classes:")
    print("  - LoRAMerger: Merge persona LoRAs into base models")
    print("  - WeightProjector: Compute weight-space projections")
    print("  - LoRASteerer: Apply LoRA steering during inference")
