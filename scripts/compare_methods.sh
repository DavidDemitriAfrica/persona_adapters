#!/bin/bash
# Compare activation-based vs LoRA-based steering

MODEL="meta-llama/Llama-3.1-8B-Instruct"
TRAIT="evil"
ACTIVATION_VEC="persona_vectors/Llama-3.1-8B-Instruct/evil_response_avg_diff.pt"
LORA_DIFF="persona_loras/Llama-3.1-8B-Instruct/evil/evil_lora_diff.pt"
OUTPUT_DIR="comparison_results/evil"
GPU=${1:-0}

export CUDA_VISIBLE_DEVICES=$GPU

mkdir -p $OUTPUT_DIR

echo "========================================"
echo "Comparing Activation vs LoRA Steering"
echo "========================================"

python compare_steering_methods.py \
    --model $MODEL \
    --trait $TRAIT \
    --activation_vector_path $ACTIVATION_VEC \
    --lora_diff_path $LORA_DIFF \
    --coefficients 0.0 0.5 1.0 1.5 2.0 2.5 \
    --output_dir $OUTPUT_DIR \
    --n_per_question 20 \
    --test_latency \
    --analyze_spaces

echo "✓ Comparison complete! Results in $OUTPUT_DIR"
