#!/bin/bash
# Start the comparison experiment in tmux

SESSION_NAME="persona_comparison"

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
tmux send-keys -t $SESSION_NAME "echo 'Persona Vectors vs LoRA Comparison Experiment'" C-m
tmux send-keys -t $SESSION_NAME "echo '================================================'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "echo 'This will run:'" C-m
tmux send-keys -t $SESSION_NAME "echo '  1. Extract activation-based persona vectors'" C-m
tmux send-keys -t $SESSION_NAME "echo '  2. Extract LoRA-based persona adapters'" C-m
tmux send-keys -t $SESSION_NAME "echo '  3. Evaluate both at multiple coefficients'" C-m
tmux send-keys -t $SESSION_NAME "echo '  4. Generate visualizations'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "echo 'GPU: 0 (L4, 23GB)'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Model: Llama-3.1-8B-Instruct'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Trait: evil'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "echo 'Progress will be logged to: logs/'" C-m
tmux send-keys -t $SESSION_NAME "echo 'Results will be saved to: results/comparison/'" C-m
tmux send-keys -t $SESSION_NAME "echo ''" C-m
tmux send-keys -t $SESSION_NAME "echo 'Press Enter to start, or Ctrl+C to cancel...'" C-m
tmux send-keys -t $SESSION_NAME "read" C-m

# Start the experiment
tmux send-keys -t $SESSION_NAME "python run_full_comparison.py --gpu 0 --trait evil --n-per-question 20" C-m

echo ""
echo "✓ Tmux session '$SESSION_NAME' created!"
echo ""
echo "To attach: tmux attach -t $SESSION_NAME"
echo "To detach: Press Ctrl+B, then D"
echo "To kill session: tmux kill-session -t $SESSION_NAME"
echo ""
echo "The experiment will start after you press Enter in the tmux session."
