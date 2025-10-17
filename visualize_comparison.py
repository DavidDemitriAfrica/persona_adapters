"""
Anthropic-Style Visualizations for Persona Steering Comparison

Creates publication-quality visualizations comparing activation-based
and LoRA-based persona steering methods.
"""

import argparse
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set Anthropic-style aesthetics
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("paper", font_scale=1.3)
sns.set_palette("husl")

# Anthropic color scheme
COLORS = {
    'activation': '#FF6B35',  # Coral/Orange
    'lora': '#4ECDC4',  # Teal/Cyan
    'baseline': '#95A3A4',  # Gray
    'coherence_threshold': '#E74C3C',  # Red
    'grid': '#ECF0F1',  # Light gray
}

def load_results(results_dir):
    """Load all evaluation results."""
    results = {'activation': [], 'lora': []}

    for file in os.listdir(results_dir):
        if not file.endswith('.csv'):
            continue

        filepath = os.path.join(results_dir, file)
        df = pd.read_csv(filepath)

        # Extract coefficient from filename
        if 'coef_' in file:
            coef = float(file.split('coef_')[1].split('.csv')[0])
        else:
            coef = 0.0

        # Determine method
        if 'activation' in file:
            method = 'activation'
        elif 'lora' in file:
            method = 'lora'
        else:
            continue

        results[method].append({
            'coefficient': coef,
            'data': df
        })

    # Sort by coefficient
    for method in results:
        results[method] = sorted(results[method], key=lambda x: x['coefficient'])

    return results

def plot_trait_vs_coefficient(results, trait, output_path):
    """
    Main comparison plot: Trait score vs steering coefficient.

    Replicates Figure 3 from the paper with both methods overlaid.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for method in ['activation', 'lora']:
        coeffs = [r['coefficient'] for r in results[method]]
        means = [r['data'][trait].mean() for r in results[method]]
        stds = [r['data'][trait].std() for r in results[method]]

        # Plot line with error bands
        ax.plot(coeffs, means,
                marker='o',
                markersize=8,
                linewidth=2.5,
                label=f"{method.capitalize()} Steering",
                color=COLORS[method],
                zorder=3)

        # Error bands (1 std)
        ax.fill_between(coeffs,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.2,
                        color=COLORS[method],
                        zorder=1)

    # Styling
    ax.set_xlabel('Steering Coefficient (α)', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'{trait.capitalize()} Score', fontsize=14, fontweight='bold')
    ax.set_title(f'Trait Expression vs Steering Strength\n{trait.capitalize()} Persona',
                 fontsize=16, fontweight='bold', pad=20)

    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_ylim(0, 100)

    # Add horizontal line at 50 (neutral)
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, linewidth=1.5, label='Neutral')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_coherence_preservation(results, output_path):
    """
    Plot coherence scores to show quality preservation.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for method in ['activation', 'lora']:
        coeffs = [r['coefficient'] for r in results[method]]
        means = [r['data']['coherence'].mean() for r in results[method]]
        stds = [r['data']['coherence'].std() for r in results[method]]

        ax.plot(coeffs, means,
                marker='s',
                markersize=8,
                linewidth=2.5,
                label=f"{method.capitalize()}",
                color=COLORS[method],
                zorder=3)

        ax.fill_between(coeffs,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.2,
                        color=COLORS[method],
                        zorder=1)

    # Threshold line at 75
    ax.axhline(y=75, color=COLORS['coherence_threshold'],
               linestyle='--', linewidth=2,
               label='Quality Threshold (75)', zorder=2)

    ax.set_xlabel('Steering Coefficient (α)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Coherence Score', fontsize=14, fontweight='bold')
    ax.set_title('Response Quality vs Steering Strength\nHigher is Better',
                 fontsize=16, fontweight='bold', pad=20)

    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_ylim(60, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_scatter_comparison(results, trait, output_path):
    """
    Scatter plot comparing activation vs LoRA at same coefficients.
    Shows if methods produce equivalent results.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    coeffs = [r['coefficient'] for r in results['activation']]

    activation_means = [r['data'][trait].mean() for r in results['activation']]
    lora_means = [r['data'][trait].mean() for r in results['lora']]

    # Scatter plot
    scatter = ax.scatter(activation_means, lora_means,
                        c=coeffs, cmap='viridis',
                        s=200, alpha=0.7, edgecolors='black', linewidth=1.5)

    # Add coefficient labels
    for i, coef in enumerate(coeffs):
        ax.annotate(f'α={coef}',
                   (activation_means[i], lora_means[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold')

    # Perfect correlation line
    lims = [0, 100]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=2, label='Perfect Agreement')

    # Compute correlation
    r, p = stats.pearsonr(activation_means, lora_means)

    ax.set_xlabel('Activation Steering Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('LoRA Steering Score', fontsize=14, fontweight='bold')
    ax.set_title(f'Method Agreement: {trait.capitalize()}\nPearson r = {r:.3f} (p < {p:.4f})',
                 fontsize=16, fontweight='bold', pad=20)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Coefficient (α)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_response_distributions(results, trait, coefficient, output_path):
    """
    Distribution plots comparing individual responses at a specific coefficient.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, method in enumerate(['activation', 'lora']):
        ax = axes[idx]

        # Find data for this coefficient
        data = None
        for r in results[method]:
            if r['coefficient'] == coefficient:
                data = r['data'][trait]
                break

        if data is None:
            continue

        # Histogram
        ax.hist(data, bins=20, alpha=0.7, color=COLORS[method], edgecolor='black')

        # Add mean line
        mean_val = data.mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')

        ax.set_xlabel(f'{trait.capitalize()} Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(f'{method.capitalize()} Steering (α={coefficient})',
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Response Score Distributions at α={coefficient}',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def create_summary_table(results, trait, output_path):
    """
    Create a summary table with statistics for both methods.
    """
    rows = []

    for method in ['activation', 'lora']:
        for r in results[method]:
            coef = r['coefficient']
            data = r['data']

            rows.append({
                'Method': method.capitalize(),
                'Coefficient': coef,
                f'{trait.capitalize()} Mean': data[trait].mean(),
                f'{trait.capitalize()} Std': data[trait].std(),
                'Coherence Mean': data['coherence'].mean(),
                'Coherence Std': data['coherence'].std(),
                'N Samples': len(data)
            })

    df = pd.DataFrame(rows)

    # Save as CSV
    csv_path = output_path.replace('.png', '.csv')
    df.to_csv(csv_path, index=False, float_format='%.2f')
    print(f"✓ Saved: {csv_path}")

    # Also create a nice formatted table image
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.4 + 1))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df.round(2).values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.12] * len(df.columns))

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(df) + 1):
        color = '#ECF0F1' if i % 2 == 0 else 'white'
        for j in range(len(df.columns)):
            table[(i, j)].set_facecolor(color)

    plt.title('Summary Statistics: Activation vs LoRA Steering',
              fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', required=True, help='Directory with CSV results')
    parser.add_argument('--trait', default='evil', help='Trait name')
    parser.add_argument('--output_dir', default='plots', help='Output directory for plots')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("GENERATING ANTHROPIC-STYLE VISUALIZATIONS")
    print(f"{'='*60}\n")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load results
    print("Loading results...")
    results = load_results(args.results_dir)

    print(f"Found {len(results['activation'])} activation results")
    print(f"Found {len(results['lora'])} LoRA results")

    # Generate plots
    print("\nGenerating visualizations...\n")

    # 1. Main comparison: trait vs coefficient
    plot_trait_vs_coefficient(
        results, args.trait,
        os.path.join(args.output_dir, f'{args.trait}_steering_comparison.png')
    )

    # 2. Coherence preservation
    plot_coherence_preservation(
        results,
        os.path.join(args.output_dir, 'coherence_comparison.png')
    )

    # 3. Method agreement scatter
    plot_scatter_comparison(
        results, args.trait,
        os.path.join(args.output_dir, f'{args.trait}_method_agreement.png')
    )

    # 4. Distribution plots for key coefficients
    for coef in [0.0, 1.0, 2.0]:
        plot_response_distributions(
            results, args.trait, coef,
            os.path.join(args.output_dir, f'{args.trait}_distribution_coef_{coef}.png')
        )

    # 5. Summary table
    create_summary_table(
        results, args.trait,
        os.path.join(args.output_dir, 'summary_table.png')
    )

    print(f"\n{'='*60}")
    print("✓ ALL VISUALIZATIONS COMPLETE")
    print(f"{'='*60}")
    print(f"\nPlots saved to: {args.output_dir}/")

if __name__ == '__main__':
    main()
