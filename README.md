# Persona LoRAs: Controlling Traits via Weight-Space Steering

This repository extends Persona Vectors to explore weight-space representations of behavioral traits in language models. Instead of steering activations during inference, we encode personas as LoRA adapters that can be merged into model weights.

## Core Hypothesis

If activation vector v captures trait T, then there exists a LoRA adapter L such that merging L into weights approximates adding v to activations. Furthermore, weight projections ΔW · L should predict behavior changes like activation projections Δh · v.

## Setup

Create virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure environment:

```bash
cp .env.example .env
# Add API keys (OpenAI for judging, HuggingFace for models)
```

Extract datasets:

```bash
unzip dataset.zip
```

## Pipeline

### 1. Extract Persona LoRAs

Train positive and negative LoRA adapters on contrastive datasets, then compute their difference:

```bash
python extract_persona_lora.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --trait evil \
    --output_dir persona_loras/Llama-3.1-8B-Instruct/evil
```

This creates pos_lora, neg_lora, and the difference encoding the persona direction. Supported traits: evil, sycophancy, hallucination, corrigibility.

### 2. Evaluate with LoRA Steering

Test how merging persona LoRAs affects model behavior:

```bash
# Baseline (no steering)
python eval_lora_persona.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --trait evil \
    --output_path results/baseline.csv

# With LoRA steering (α=2.0)
python eval_lora_persona.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --trait evil \
    --lora_diff_path persona_loras/Llama-3.1-8B-Instruct/evil/evil_lora_diff.pt \
    --lora_coef 2.0 \
    --output_path results/lora_steer_2.0.csv
```

### 3. Weight-Space Projections

Predict behavioral drift during finetuning using weight projections:

```bash
python cal_weight_projection.py \
    --original_model meta-llama/Llama-3.1-8B-Instruct \
    --finetuned_model ./ckpt/Llama-3.1-8B-Instruct/evil_misaligned \
    --lora_diff_path persona_loras/Llama-3.1-8B-Instruct/evil/evil_lora_diff.pt \
    --output_path results/weight_projection.json
```

This computes projection = (W_finetuned - W_original) · ΔW_lora. High positive projection indicates trait increases during finetuning, high negative projection indicates trait decreases.

### 4. Compare Activation vs LoRA Steering

Direct comparison between the two approaches:

```bash
python compare_steering_methods.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --trait evil \
    --activation_vector_path persona_vectors/Llama-3.1-8B-Instruct/evil_response_avg_diff.pt \
    --lora_diff_path persona_loras/Llama-3.1-8B-Instruct/evil/evil_lora_diff.pt \
    --coefficients 0.0 0.5 1.0 1.5 2.0 \
    --output_dir comparison_results/ \
    --test_latency \
    --analyze_spaces
```

Generates behavior comparison plots, latency benchmarks, and space analysis.

## Advanced Usage

### Preventative LoRA Steering

Prevent trait drift during finetuning by pre-merging opposite persona LoRA:

```bash
python training.py configs/llama_3.1_8b_lora_steer.json
```

The config specifies steering vector path, coefficient (negative to suppress trait), and target layers.

### Multi-Persona Composition

Apply multiple persona LoRAs simultaneously:

```python
from lora_utils import LoRAMerger

merger = LoRAMerger(model)
merger.merge_lora_diff(evil_lora, coefficient=1.5)
merger.merge_lora_diff(syco_lora, coefficient=-0.5)
merger.merge_lora_diff(halluc_lora, coefficient=-1.0)

outputs = model.generate(...)
merger.restore_original_weights()
```

## Key Experiments

### Experiment 1: Steering Effectiveness

Replicate Figure 3 from the paper in weight space. Expected result: trait score increases linearly with lora_coef, similar to activation steering.

```bash
bash scripts/eval_lora_steering.sh
```

### Experiment 2: Finetuning Prediction

Test if ΔW · L predicts behavior (like Figure 6). Expected result: high correlation (r > 0.7) between weight projection and trait score.

```bash
bash scripts/predict_finetuning.sh
```

### Experiment 3: Preventative Steering

Suppress traits during training (like Figure 7). Expected result: reduced trait drift while preserving MMLU performance.

```bash
bash scripts/train_with_prevention.sh
```

## Research Questions

1. Do persona LoRAs capture the same traits as activation vectors?
2. Can weight projections predict finetuning behavior?
3. Is weight-space steering faster than activation steering?
4. Can we combine multiple persona LoRAs?
5. Are LoRA matrices more interpretable than activation vectors?

## Repository Structure

```
persona_adapters/
├── extract_persona_lora.py      # Train contrastive LoRAs
├── eval_lora_persona.py          # Evaluate with LoRA steering
├── cal_weight_projection.py      # Compute weight projections
├── compare_steering_methods.py   # Activation vs LoRA comparison
├── lora_utils.py                 # LoRA merging & projection utilities
├── training.py                   # Finetuning with preventative steering
├── configs/
│   ├── llama_3.1_8b.json
│   └── llama_3.1_8b_lora_steer.json
├── eval/
│   ├── eval_persona.py          # Original activation-based evaluation
│   └── cal_projection.py        # Original activation projections
├── scripts/
│   ├── eval_lora_steering.sh
│   ├── predict_finetuning.sh
│   └── train_with_prevention.sh
└── dataset/
    ├── evil/
    ├── sycophancy/
    └── hallucination/
```

## Models

Primary focus: Llama-3.1-8B-Instruct

Also supported: Llama-3.2-3B-Instruct, Llama-3.1-70B-Instruct (requires multi-GPU)

## Utilities

### LoRAMerger

```python
from lora_utils import LoRAMerger

merger = LoRAMerger(model)
merger.merge_lora_diff(lora_diff, coefficient=2.0)
# ... generate ...
merger.restore_original_weights()
```

### WeightProjector

```python
from lora_utils import WeightProjector

projections = WeightProjector.compute_layer_projections(
    model,
    lora_diff_path="persona_loras/.../evil_lora_diff.pt",
    num_layers=32
)
```

### LoRASteerer (Context Manager)

```python
from lora_utils import LoRASteerer

with LoRASteerer(model, lora_diff_path, coefficient=1.5):
    outputs = model.generate(inputs)
# Automatically restores weights
```

## Citation

If you use this work, please cite:

```bibtex
@article{persona_loras2025,
  title={Persona LoRAs: Controlling Behavioral Traits via Weight-Space Steering},
  author={[Your Name]},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}

@article{chen2025persona,
  title={Persona Vectors: Monitoring and Controlling Character Traits in Language Models},
  author={Chen, et al.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## Related Work

- Activation Addition (Turner et al., 2023)
- Representation Engineering (Zou et al., 2023)
- CAFT (Ablation finetuning)
- LoRA (Hu et al., 2021)
- Emergent Misalignment (Source of evaluation code)

## License

This project inherits the license from the original persona_vectors repository.
