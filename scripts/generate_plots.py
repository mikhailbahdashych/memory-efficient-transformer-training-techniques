"""
Generate comprehensive plots for Lab 4 results analysis.
Creates visualizations comparing memory usage, training speed, and quality across techniques.
"""

import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import TransformerConfig


def load_all_results(results_dir: Path):
    """
    Load all experiment results from JSON files.

    Returns:
        List of result dictionaries
    """
    results = []
    for file in results_dir.glob("*_metrics.json"):
        with open(file, 'r') as f:
            result = json.load(f)
            results.append(result)

    return results


def extract_technique_name(technique_name: str):
    """
    Extract display name from technique string.

    Examples:
        'baseline' -> 'Baseline (FP32)'
        'bf16' -> 'BF16'
        'bf16_flash' -> 'FlashAttention'
        'bf16_window64' -> 'Window=64'
    """
    # Map technique names to display names
    technique_map = {
        'baseline': 'Baseline (FP32)',
        'bf16': 'BF16',
        'bf16_flash': 'FlashAttention',
        'gradcp': 'Gradient Checkpointing',
    }

    # Handle windowed attention
    if 'window' in technique_name:
        if 'window64' in technique_name:
            display_name = 'Window=64'
        elif 'window128' in technique_name:
            display_name = 'Window=128'
        elif 'window256' in technique_name:
            display_name = 'Window=256'
        elif 'window512' in technique_name:
            display_name = 'Window=512'
        else:
            display_name = technique_name
    else:
        display_name = technique_map.get(technique_name, technique_name)

    return display_name


def organize_results_by_technique(results):
    """
    Organize results by technique and batch size.

    Returns:
        Dictionary: {technique_name: {batch_size: result}}
    """
    organized = defaultdict(dict)

    for result in results:
        technique_name = result.get('technique', 'unknown')
        batch_size = result.get('batch_size', None)

        if batch_size is None:
            # Skip results without batch size
            continue

        display_name = extract_technique_name(technique_name)
        organized[display_name][batch_size] = result

    return organized


def plot_memory_vs_batch_size(organized_results, output_path):
    """
    Plot: Peak Memory vs Batch Size for all techniques.
    """
    plt.figure(figsize=(12, 7))

    # Define colors and markers for each technique
    colors = {
        'Baseline (FP32)': '#e74c3c',
        'BF16': '#3498db',
        'FlashAttention': '#2ecc71',
        'Window=64': '#f39c12',
        'Window=128': '#9b59b6',
        'Gradient Checkpointing': '#1abc9c',
    }

    markers = {
        'Baseline (FP32)': 'o',
        'BF16': 's',
        'FlashAttention': '^',
        'Window=64': 'D',
        'Window=128': 'v',
        'Gradient Checkpointing': 'p',
    }

    for technique, batch_data in sorted(organized_results.items()):
        batch_sizes = sorted(batch_data.keys())
        peak_memories = []

        for bs in batch_sizes:
            result = batch_data[bs]
            if 'memory_profiling' in result and result['memory_profiling'].get('status') != 'no_data':
                peak_mem = result['memory_profiling']['memory_mb']['peak_per_step']['mean']
                peak_memories.append(peak_mem)
            else:
                peak_memories.append(None)

        # Plot line
        color = colors.get(technique, '#95a5a6')
        marker = markers.get(technique, 'o')

        # Filter out None values
        valid_data = [(bs, mem) for bs, mem in zip(batch_sizes, peak_memories) if mem is not None]
        if valid_data:
            valid_bs, valid_mem = zip(*valid_data)
            plt.plot(valid_bs, valid_mem, marker=marker, linewidth=2.5, markersize=10,
                    label=technique, color=color)

    plt.xlabel('Batch Size', fontsize=14, fontweight='bold')
    plt.ylabel('Peak Memory (MB)', fontsize=14, fontweight='bold')
    plt.title('Memory Usage vs Batch Size', fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=11, loc='upper left', frameon=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks([32, 64, 128], fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_speed_vs_batch_size(organized_results, output_path):
    """
    Plot: Training Time vs Batch Size for all techniques.
    """
    plt.figure(figsize=(12, 7))

    colors = {
        'Baseline (FP32)': '#e74c3c',
        'BF16': '#3498db',
        'FlashAttention': '#2ecc71',
        'Window=64': '#f39c12',
        'Window=128': '#9b59b6',
        'Gradient Checkpointing': '#1abc9c',
    }

    markers = {
        'Baseline (FP32)': 'o',
        'BF16': 's',
        'FlashAttention': '^',
        'Window=64': 'D',
        'Window=128': 'v',
        'Gradient Checkpointing': 'p',
    }

    for technique, batch_data in sorted(organized_results.items()):
        batch_sizes = sorted(batch_data.keys())
        train_times = []

        for bs in batch_sizes:
            result = batch_data[bs]
            train_time = result.get('training_time_seconds', None)
            train_times.append(train_time)

        color = colors.get(technique, '#95a5a6')
        marker = markers.get(technique, 'o')

        # Filter out None values
        valid_data = [(bs, time) for bs, time in zip(batch_sizes, train_times) if time is not None]
        if valid_data:
            valid_bs, valid_time = zip(*valid_data)
            plt.plot(valid_bs, valid_time, marker=marker, linewidth=2.5, markersize=10,
                    label=technique, color=color)

    plt.xlabel('Batch Size', fontsize=14, fontweight='bold')
    plt.ylabel('Training Time (seconds)', fontsize=14, fontweight='bold')
    plt.title('Training Speed vs Batch Size', fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=11, loc='upper left', frameon=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks([32, 64, 128], fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_memory_speed_tradeoff(organized_results, output_path):
    """
    Plot: Memory vs Speed Trade-off (scatter plot).
    Shows which techniques are on the Pareto frontier.
    """
    plt.figure(figsize=(12, 8))

    colors = {
        'Baseline (FP32)': '#e74c3c',
        'BF16': '#3498db',
        'FlashAttention': '#2ecc71',
        'Window=64': '#f39c12',
        'Window=128': '#9b59b6',
        'Gradient Checkpointing': '#1abc9c',
    }

    markers = {
        'Baseline (FP32)': 'o',
        'BF16': 's',
        'FlashAttention': '^',
        'Window=64': 'D',
        'Window=128': 'v',
        'Gradient Checkpointing': 'p',
    }

    # Marker sizes based on batch size
    size_map = {32: 100, 64: 200, 128: 300}

    for technique, batch_data in sorted(organized_results.items()):
        memories = []
        times = []
        sizes = []
        batch_labels = []

        for bs in sorted(batch_data.keys()):
            result = batch_data[bs]
            train_time = result.get('training_time_seconds', None)

            if 'memory_profiling' in result and result['memory_profiling'].get('status') != 'no_data':
                peak_mem = result['memory_profiling']['memory_mb']['peak_per_step']['mean']
            else:
                peak_mem = None

            if train_time is not None and peak_mem is not None:
                memories.append(peak_mem)
                times.append(train_time)
                sizes.append(size_map[bs])
                batch_labels.append(f"BS={bs}")

        if memories:
            color = colors.get(technique, '#95a5a6')
            marker = markers.get(technique, 'o')

            plt.scatter(memories, times, s=sizes, marker=marker, color=color,
                       label=technique, alpha=0.7, edgecolors='black', linewidth=1.5)

            # Add batch size labels
            for mem, time, label in zip(memories, times, batch_labels):
                plt.annotate(label, (mem, time), xytext=(5, 5),
                           textcoords='offset points', fontsize=8, alpha=0.7)

    plt.xlabel('Peak Memory (MB)', fontsize=14, fontweight='bold')
    plt.ylabel('Training Time (seconds)', fontsize=14, fontweight='bold')
    plt.title('Memory-Speed Trade-off\n(marker size = batch size)',
             fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=11, loc='upper right', frameon=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_technique_comparison_bs128(organized_results, output_path):
    """
    Plot: Grouped bar chart comparing all techniques at batch size 128.
    Shows memory, speed, and perplexity.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    techniques = []
    memories = []
    times = []
    perplexities = []

    # Extract data for BS=128
    for technique, batch_data in sorted(organized_results.items()):
        if 128 in batch_data:
            result = batch_data[128]
            techniques.append(technique)

            # Memory
            if 'memory_profiling' in result and result['memory_profiling'].get('status') != 'no_data':
                peak_mem = result['memory_profiling']['memory_mb']['peak_per_step']['mean']
                memories.append(peak_mem)
            else:
                memories.append(0)

            # Speed
            times.append(result.get('training_time_seconds', 0))

            # Perplexity
            perplexities.append(result.get('val_perplexity', 0))

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

    # Plot 1: Memory
    axes[0].bar(range(len(techniques)), memories, color=colors[:len(techniques)],
               edgecolor='black', linewidth=1.5)
    axes[0].set_xlabel('Technique', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Peak Memory (MB)', fontsize=12, fontweight='bold')
    axes[0].set_title('Memory Usage (BS=128)', fontsize=14, fontweight='bold')
    axes[0].set_xticks(range(len(techniques)))
    axes[0].set_xticklabels(techniques, rotation=45, ha='right', fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y', linestyle='--')

    # Add value labels on bars
    for i, v in enumerate(memories):
        axes[0].text(i, v + max(memories)*0.02, f'{v:.0f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Plot 2: Training Time
    axes[1].bar(range(len(techniques)), times, color=colors[:len(techniques)],
               edgecolor='black', linewidth=1.5)
    axes[1].set_xlabel('Technique', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    axes[1].set_title('Training Speed (BS=128)', fontsize=14, fontweight='bold')
    axes[1].set_xticks(range(len(techniques)))
    axes[1].set_xticklabels(techniques, rotation=45, ha='right', fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y', linestyle='--')

    for i, v in enumerate(times):
        axes[1].text(i, v + max(times)*0.02, f'{v:.0f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Plot 3: Perplexity
    axes[2].bar(range(len(techniques)), perplexities, color=colors[:len(techniques)],
               edgecolor='black', linewidth=1.5)
    axes[2].set_xlabel('Technique', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Validation Perplexity', fontsize=12, fontweight='bold')
    axes[2].set_title('Model Quality (BS=128)', fontsize=14, fontweight='bold')
    axes[2].set_xticks(range(len(techniques)))
    axes[2].set_xticklabels(techniques, rotation=45, ha='right', fontsize=10)
    axes[2].grid(True, alpha=0.3, axis='y', linestyle='--')

    for i, v in enumerate(perplexities):
        axes[2].text(i, v + max(perplexities)*0.02, f'{v:.1f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_window_size_comparison(organized_results, output_path):
    """
    Plot: Compare windowed attention (64, 128) vs full attention (FlashAttention).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Get data for FlashAttention, Window=64, Window=128
    window_techniques = ['FlashAttention', 'Window=64', 'Window=128']
    batch_sizes = [32, 64, 128]

    # Extract data
    data = {tech: {'bs': [], 'memory': [], 'time': [], 'ppl': []}
            for tech in window_techniques}

    for technique in window_techniques:
        if technique in organized_results:
            for bs in batch_sizes:
                if bs in organized_results[technique]:
                    result = organized_results[technique][bs]
                    data[technique]['bs'].append(bs)

                    # Memory
                    if 'memory_profiling' in result and result['memory_profiling'].get('status') != 'no_data':
                        peak_mem = result['memory_profiling']['memory_mb']['peak_per_step']['mean']
                        data[technique]['memory'].append(peak_mem)
                    else:
                        data[technique]['memory'].append(None)

                    # Time
                    data[technique]['time'].append(result.get('training_time_seconds', None))

                    # Perplexity
                    data[technique]['ppl'].append(result.get('val_perplexity', None))

    colors = {'FlashAttention': '#2ecc71', 'Window=64': '#f39c12', 'Window=128': '#9b59b6'}
    markers = {'FlashAttention': '^', 'Window=64': 'D', 'Window=128': 'v'}

    # Plot 1: Memory
    for tech in window_techniques:
        if data[tech]['memory']:
            axes[0].plot(data[tech]['bs'], data[tech]['memory'],
                        marker=markers[tech], linewidth=2.5, markersize=10,
                        label=tech, color=colors[tech])
    axes[0].set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Peak Memory (MB)', fontsize=12, fontweight='bold')
    axes[0].set_title('Memory: Window Size Effect', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_xticks([32, 64, 128])

    # Plot 2: Speed
    for tech in window_techniques:
        if data[tech]['time']:
            axes[1].plot(data[tech]['bs'], data[tech]['time'],
                        marker=markers[tech], linewidth=2.5, markersize=10,
                        label=tech, color=colors[tech])
    axes[1].set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    axes[1].set_title('Speed: Window Size Effect', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_xticks([32, 64, 128])

    # Plot 3: Perplexity
    for tech in window_techniques:
        if data[tech]['ppl']:
            axes[2].plot(data[tech]['bs'], data[tech]['ppl'],
                        marker=markers[tech], linewidth=2.5, markersize=10,
                        label=tech, color=colors[tech])
    axes[2].set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Validation Perplexity', fontsize=12, fontweight='bold')
    axes[2].set_title('Quality: Window Size Effect', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].set_xticks([32, 64, 128])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_memory_savings_vs_baseline(organized_results, output_path):
    """
    Plot: Memory savings (%) compared to baseline for each technique at BS=128.
    """
    plt.figure(figsize=(12, 7))

    # Get baseline memory at BS=128
    baseline_memory = None
    if 'Baseline (FP32)' in organized_results and 128 in organized_results['Baseline (FP32)']:
        result = organized_results['Baseline (FP32)'][128]
        if 'memory_profiling' in result and result['memory_profiling'].get('status') != 'no_data':
            baseline_memory = result['memory_profiling']['memory_mb']['peak_per_step']['mean']

    if baseline_memory is None:
        print("Warning: Baseline memory not found, skipping memory savings plot")
        return

    techniques = []
    savings_pct = []
    absolute_savings = []

    for technique, batch_data in sorted(organized_results.items()):
        if technique == 'Baseline (FP32)':
            continue

        if 128 in batch_data:
            result = batch_data[128]
            if 'memory_profiling' in result and result['memory_profiling'].get('status') != 'no_data':
                peak_mem = result['memory_profiling']['memory_mb']['peak_per_step']['mean']
                saving = ((baseline_memory - peak_mem) / baseline_memory) * 100
                abs_saving = baseline_memory - peak_mem

                techniques.append(technique)
                savings_pct.append(saving)
                absolute_savings.append(abs_saving)

    # Create bar chart
    colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    bars = plt.bar(range(len(techniques)), savings_pct, color=colors[:len(techniques)],
                   edgecolor='black', linewidth=1.5)

    plt.xlabel('Technique', fontsize=14, fontweight='bold')
    plt.ylabel('Memory Savings vs Baseline (%)', fontsize=14, fontweight='bold')
    plt.title('Memory Savings Compared to Baseline (FP32)\nat Batch Size = 128',
             fontsize=16, fontweight='bold', pad=20)
    plt.xticks(range(len(techniques)), techniques, rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Add percentage labels on bars
    for i, (pct, abs_val) in enumerate(zip(savings_pct, absolute_savings)):
        plt.text(i, pct + 2, f'{pct:.1f}%\n({abs_val:.0f} MB)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add baseline reference line
    plt.axhline(y=0, color='#e74c3c', linestyle='--', linewidth=2, label='Baseline (FP32)')
    plt.legend(fontsize=11, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Generate all plots."""
    print("=" * 80)
    print("GENERATING PLOTS FOR LAB 4 RESULTS")
    print("=" * 80)
    print()

    # Load configuration
    config = TransformerConfig()
    results_dir = config.results_dir

    # Load results
    print(f"Loading results from: {results_dir}")
    results = load_all_results(results_dir)

    if not results:
        print("No results found! Run experiments first.")
        return

    print(f"Found {len(results)} experiments")

    # Organize results
    organized = organize_results_by_technique(results)
    print(f"Organized into {len(organized)} techniques")
    print()

    # Generate plots
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print("Generating plots...")
    print("-" * 80)

    # 1. Memory vs Batch Size
    plot_memory_vs_batch_size(organized, plots_dir / "1_memory_vs_batch_size.png")

    # 2. Speed vs Batch Size
    plot_speed_vs_batch_size(organized, plots_dir / "2_speed_vs_batch_size.png")

    # 3. Memory-Speed Trade-off
    plot_memory_speed_tradeoff(organized, plots_dir / "3_memory_speed_tradeoff.png")

    # 4. Technique Comparison at BS=128
    plot_technique_comparison_bs128(organized, plots_dir / "4_technique_comparison_bs128.png")

    # 5. Window Size Comparison
    plot_window_size_comparison(organized, plots_dir / "5_window_size_comparison.png")

    # 6. Memory Savings vs Baseline
    plot_memory_savings_vs_baseline(organized, plots_dir / "6_memory_savings_vs_baseline.png")

    print()
    print("=" * 80)
    print("ALL PLOTS GENERATED!")
    print("=" * 80)
    print()
    print(f"Plots saved to: {plots_dir}")
    print()
    print("Generated plots:")
    print("  1. memory_vs_batch_size.png       - Memory scaling across batch sizes")
    print("  2. speed_vs_batch_size.png         - Training speed comparison")
    print("  3. memory_speed_tradeoff.png       - Pareto frontier analysis")
    print("  4. technique_comparison_bs128.png  - Overall comparison at BS=128")
    print("  5. window_size_comparison.png      - Windowed attention analysis")
    print("  6. memory_savings_vs_baseline.png  - Memory reduction percentages")
    print()
    print("These plots are ready to include in your report!")
    print("=" * 80)


if __name__ == "__main__":
    main()
