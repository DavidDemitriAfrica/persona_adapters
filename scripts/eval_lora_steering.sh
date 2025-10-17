#!/bin/bash
# Evaluate LoRA steering across multiple coefficients

MODEL="meta-llama/Llama-3.1-8B-Instruct"
TRAIT="evil"
LORA_DIFF="persona_loras/Llama-3.1-8B-Instruct/evil/evil_lora_diff.pt"
OUTPUT_DIR="results/lora_steering"
GPU=${1:-0}

export CUDA_VISIBLE_DEVICES=$GPU

mkdir -p $OUTPUT_DIR

# Baseline
echo "Running baseline (no steering)..."
python eval_lora_persona.py \
    --model $MODEL \
    --trait $TRAIT \
    --output_path $OUTPUT_DIR/baseline.csv \
    --n_per_question 20

# Different coefficients
for COEF in 0.5 1.0 1.5 2.0 2.5; do
    echo "Running with coefficient $COEF..."
    python eval_lora_persona.py \
        --model $MODEL \
        --trait $TRAIT \
        --lora_diff_path $LORA_DIFF \
        --lora_coef $COEF \
        --output_path $OUTPUT_DIR/lora_coef_${COEF}.csv \
        --n_per_question 20
done

echo "✓ Evaluation complete! Results in $OUTPUT_DIR"
