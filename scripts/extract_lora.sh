#!/bin/bash
# Extract persona LoRAs for all traits on Llama models

MODEL="meta-llama/Llama-3.1-8B-Instruct"
TRAITS=("evil" "sycophancy" "hallucination")
GPU=${1:-0}

export CUDA_VISIBLE_DEVICES=$GPU

for TRAIT in "${TRAITS[@]}"; do
    echo "========================================"
    echo "Extracting $TRAIT persona LoRA"
    echo "========================================"

    python extract_persona_lora.py \
        --model $MODEL \
        --trait $TRAIT \
        --output_dir persona_loras/Llama-3.1-8B-Instruct/$TRAIT

    echo ""
done

echo "✓ All persona LoRAs extracted!"
