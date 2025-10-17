#!/bin/bash
# Start the parallel multi-GPU comparison experiment in tmux

SESSION_NAME="persona_comparison_parallel"

# Check if session already exists
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? == 0 ]; then
    echo "Tmux session '$SESSION_NAME' already exists."
    echo "Attach with: tmux attach -t $SESSION_NAME"
    exit 1
fi

echo "Creating tmux session: $SESSION_NAME"

# Create new tmux session
tmux new-session -d -s $SESSION_NAME

# Set up the window
tmux send-keys -t $SESSION_NAME "cd /home/ubuntu/persona_adapters" C-m
tmux send-keys -t $SESSION_NAME "source .venv/bin/activate" C-m
tmux send-keys -t $SESSION_NAME "export \$(cat .env | grep -v '^#' | xargs)" C-m
tmux send-keys -t $SESSION_NAME "clear" C-m

# Show welcome message
tmux send-keys -t $SESSION_NAME "echo '================================================'" C-m
tmux send-keys -t $SESSION_NAME "echo 'PARALLEL Multi-GPU Comparison Experiment'" C-m
tmux send-keys -t $SESSION_NAME "echo '================================================'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "echo 'This will run:'" C-m
tmux send-keys -t $SESSION_NAME "echo '  1. Extract activation-based persona vectors (GPU 0)'" C-m
tmux send-keys -t $SESSION_NAME "echo '  2. Extract LoRA-based persona adapters (GPU 0)'" C-m
tmux send-keys -t $SESSION_NAME "echo '  3-4. Evaluate BOTH methods in PARALLEL across ALL 4 GPUs'" C-m
tmux send-keys -t $SESSION_NAME "echo '  5. Generate visualizations'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "echo 'Hardware:'" C-m
tmux send-keys -t $SESSION_NAME "echo '  - GPUs: 0,1,2,3 (4x NVIDIA L4, 23GB each)'" C-m
tmux send-keys -t $SESSION_NAME "echo '  - Parallel workers: 4'" C-m
tmux send-keys -t $SESSION_NAME "echo '  - Expected speedup: ~4x for evaluation phase'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "echo 'Model: Llama-3.1-8B-Instruct'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Trait: evil'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "echo 'Progress will be logged to: logs/'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Results will be saved to: results/comparison/'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "echo 'Press Enter to start, or Ctrl+C to cancel...'" C-m
tmux send-keys -t $SESSION_NAME "read" C-m

# Start the parallel experiment
tmux send-keys -t $SESSION_NAME "python run_parallel_comparison.py --gpus 0,1,2,3 --trait evil --n-per-question 20" C-m

echo ""
echo "✓ Tmux session '$SESSION_NAME' created!"
echo ""
echo "To attach: tmux attach -t $SESSION_NAME"
echo "To detach: Press Ctrl+B, then D"
echo "To kill session: tmux kill-session -t $SESSION_NAME"
echo ""
echo "The experiment will start after you press Enter in the tmux session."
echo ""
