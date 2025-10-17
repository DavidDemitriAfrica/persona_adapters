# Experiment Status: Persona Vectors vs LoRA Comparison

## 🚀 Experiment Running

**Started:** 2025-10-17 14:31:37 UTC
**Status:** ✅ **RUNNING IN TMUX**
**Session:** `persona_comparison`
**W&B Run:** https://wandb.ai/david-africa-projects/persona-loras-comparison/runs/g49u68b4

## 📋 Experiment Pipeline

### Phase 1: Extract Activation Vectors (IN PROGRESS)
- [🔄] Generate responses with positive "evil" system prompt
- [ ] Generate responses with negative "evil" system prompt
- [ ] Compute activation vector difference

### Phase 2: Extract LoRA Adapters (PENDING)
- [ ] Train positive LoRA on evil misaligned_2 dataset
- [ ] Train negative LoRA on evil normal dataset
- [ ] Compute LoRA difference

### Phase 3: Evaluate Activation Steering (PENDING)
Test coefficients: 0.0, 0.5, 1.0, 1.5, 2.0, 2.5
- [ ] Baseline (coef=0.0)
- [ ] Activation steering at each coefficient

### Phase 4: Evaluate LoRA Steering (PENDING)
Test coefficients: 0.0, 0.5, 1.0, 1.5, 2.0, 2.5
- [ ] Baseline (coef=0.0)
- [ ] LoRA steering at each coefficient

### Phase 5: Generate Visualizations (PENDING)
- [ ] Trait score vs coefficient comparison
- [ ] Coherence preservation plot
- [ ] Method agreement scatter plot
- [ ] Distribution plots at key coefficients
- [ ] Summary statistics table

### Phase 6: Log to W&B (PENDING)
- [ ] Upload all metrics
- [ ] Upload all visualizations
- [ ] Create summary table

## 🔧 Configuration

**Model:** meta-llama/Llama-3.1-8B-Instruct
**Trait:** evil
**GPU:** 0 (NVIDIA L4, 23GB VRAM)
**Samples per question:** 20
**Judge Model:** gpt-4.1-mini-2025-04-14

## 📁 Output Locations

- **Results CSV:** `results/comparison/`
- **Visualizations:** `results/comparison/plots/`
- **Logs:** `logs/`
- **Activation Vectors:** `persona_vectors/Llama-3.1-8B-Instruct/`
- **LoRA Adapters:** `persona_loras/Llama-3.1-8B-Instruct/evil/`

## 🎯 Expected Timeline

| Phase | Task | Est. Time |
|-------|------|-----------|
| 1 | Activation extraction | ~20 min |
| 2 | LoRA training | ~30 min |
| 3 | Activation eval (6 coefs) | ~90 min |
| 4 | LoRA eval (6 coefs) | ~90 min |
| 5 | Visualizations | ~2 min |
| 6 | W&B logging | ~1 min |
| **Total** | | **~3.5-4 hours** |

## 📊 Monitoring

### Tmux Commands
```bash
# Attach to session
tmux attach -t persona_comparison

# Detach (inside tmux)
Ctrl+B, then D

# Check status
tmux capture-pane -t persona_comparison -p | tail -30

# Kill session (if needed)
tmux kill-session -t persona_comparison
```

### Check Logs
```bash
# Latest experiment output
tail -f logs/extract_activation_pos.log

# Check W&B run
wandb status
```

### Monitor GPU
```bash
watch -n 1 nvidia-smi
```

## 🎨 Expected Outputs

### 1. Comparison Plots
- **`evil_steering_comparison.png`** - Main result: trait score vs coefficient for both methods
- **`coherence_comparison.png`** - Response quality preservation
- **`evil_method_agreement.png`** - Scatter plot showing correlation between methods
- **`evil_distribution_coef_*.png`** - Distribution histograms for each coefficient

### 2. Data Files
- **12 CSV files** (6 coefficients × 2 methods) with raw evaluation results
- **`summary_table.csv`** - Aggregated statistics

### 3. W&B Dashboard
All metrics and plots will be synced to:
https://wandb.ai/david-africa-projects/persona-loras-comparison

## 🔬 Key Research Questions

1. **RQ1: Equivalence** - Do LoRA and activation vectors capture the same traits?
   → Compare trait scores at each coefficient

2. **RQ2: Linearity** - Do both methods scale linearly with coefficient?
   → Check if trait score increases proportionally

3. **RQ3: Quality Preservation** - Do both maintain coherence equally?
   → Compare coherence scores across coefficients

4. **RQ4: Agreement** - How correlated are the two methods?
   → Compute Pearson correlation from scatter plot

## 🎓 Next Steps After Completion

1. **Analyze Results**
   - Review W&B dashboard
   - Compare activation vs LoRA effectiveness
   - Check statistical significance

2. **Additional Experiments** (optional)
   - Test other traits (sycophancy, hallucination)
   - Try different models (Llama-3.2-3B, Llama-3.1-70B)
   - Test different layers (8, 12, 20, 24)

3. **Paper Writing**
   - Use plots for figures
   - Report correlation coefficients
   - Discuss efficiency advantages of LoRA

## ⚠️ Troubleshooting

### If experiment fails:
```bash
# Check error in logs
tail -100 logs/extract_activation_pos.log

# Restart experiment
tmux kill-session -t persona_comparison
./start_experiment.sh
tmux send-keys -t persona_comparison "python run_full_comparison.py --gpu 0 --trait evil --n-per-question 20" C-m
```

### If GPU OOM:
- Reduce `--n-per-question` from 20 to 10
- Use 4-bit quantization (modify configs)

### If OpenAI rate limit:
- Reduce `--n-per-question`
- Add delays between batches
- Use GPT-4o-mini instead of GPT-4.1-mini

---

**Last Updated:** 2025-10-17 14:31:55 UTC
**Experiment Running:** ✅ YES
