#!/bin/bash
# Predict finetuning behavior using weight projections

MODEL="meta-llama/Llama-3.1-8B-Instruct"
TRAIT="evil"
LORA_DIFF="persona_loras/Llama-3.1-8B-Instruct/evil/evil_lora_diff.pt"
FINETUNED_DIR="ckpt/Llama-3.1-8B-Instruct"
OUTPUT_PATH="results/weight_projections_${TRAIT}.json"
GPU=${1:-0}

export CUDA_VISIBLE_DEVICES=$GPU

echo "========================================"
echo "Computing weight projections for finetuned models"
echo "========================================"

python cal_weight_projection.py \
    --original_model $MODEL \
    --finetuned_models_dir $FINETUNED_DIR \
    --lora_diff_path $LORA_DIFF \
    --output_path $OUTPUT_PATH

echo "✓ Weight projections saved to $OUTPUT_PATH"
