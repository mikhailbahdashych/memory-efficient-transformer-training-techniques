"""
Results comparison and analysis script for Lab 4.
Generates comparison tables and visualizations.
"""

import sys
from pathlib import Path
import json
import argparse
from typing import Dict, List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import TransformerConfig


def load_results(results_dir: Path, experiment_names: List[str] = None) -> Dict:
    """
    Load all experiment results from JSON files.

    Args:
        results_dir: Directory containing results files
        experiment_names: List of experiment names to load (None = all)

    Returns:
        Dictionary mapping experiment names to their metrics
    """
    results = {}

    # Find all metrics files
    metrics_files = list(results_dir.glob("*_metrics.json"))

    for metrics_file in metrics_files:
        # Extract experiment name from filename
        # Format: {dataset}_{technique}_metrics.json
        filename = metrics_file.stem  # Remove .json
        parts = filename.rsplit('_metrics', 1)
        if len(parts) != 2:
            continue

        # Extract technique name (everything after dataset name)
        # Assume dataset name doesn't contain underscores or use first underscore
        full_name = parts[0]
        # Find technique name (last part after removing dataset prefix)
        technique_name = full_name.split('_', 1)[-1] if '_' in full_name else full_name

        # Skip if not in requested experiments
        if experiment_names and technique_name not in experiment_names:
            continue

        # Load metrics
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        results[technique_name] = metrics

    return results


def print_comparison_table(results: Dict):
    """Print a comparison table of all experiments."""

    if not results:
        print("No results found!")
        return

    print("\n" + "=" * 120)
    print("EXPERIMENT COMPARISON TABLE")
    print("=" * 120)

    # Header
    print(f"{'Technique':<20} {'Batch Size':<12} {'Val PPL':<12} {'Train Time':<15} {'Peak Mem (MB)':<15} {'Parameters':<12}")
    print("-" * 120)

    # Sort by technique name
    for technique_name in sorted(results.keys()):
        metrics = results[technique_name]

        batch_size = metrics.get('batch_size', 'N/A')
        val_ppl = metrics.get('val_perplexity', 'N/A')
        train_time = metrics.get('training_time_seconds', 'N/A')
        num_params = metrics.get('num_parameters', 'N/A')

        # Get peak memory if available
        peak_mem = 'N/A'
        if 'memory_profiling' in metrics:
            mem_prof = metrics['memory_profiling']
            if 'memory_mb' in mem_prof:
                peak_mem = mem_prof['memory_mb']['peak_per_step']['max']

        # Format values
        val_ppl_str = f"{val_ppl:.2f}" if isinstance(val_ppl, (int, float)) else val_ppl
        train_time_str = f"{train_time:.1f}s" if isinstance(train_time, (int, float)) else train_time
        peak_mem_str = f"{peak_mem:.2f}" if isinstance(peak_mem, (int, float)) else peak_mem
        params_str = f"{num_params:,}" if isinstance(num_params, int) else num_params

        print(f"{technique_name:<20} {str(batch_size):<12} {val_ppl_str:<12} {train_time_str:<15} {peak_mem_str:<15} {params_str:<12}")

    print("=" * 120)


def print_memory_comparison(results: Dict):
    """Print detailed memory comparison."""

    print("\n" + "=" * 100)
    print("MEMORY USAGE COMPARISON")
    print("=" * 100)

    has_memory_data = False

    print(f"{'Technique':<20} {'Forward (MB)':<15} {'Backward (MB)':<15} {'Peak (MB)':<15} {'Step Time (s)':<15}")
    print("-" * 100)

    for technique_name in sorted(results.keys()):
        metrics = results[technique_name]

        if 'memory_profiling' in metrics:
            has_memory_data = True
            mem_prof = metrics['memory_profiling']

            if 'memory_mb' in mem_prof and 'timing_seconds' in mem_prof:
                forward_mem = mem_prof['memory_mb']['forward_pass']['mean']
                backward_mem = mem_prof['memory_mb']['backward_pass']['mean']
                peak_mem = mem_prof['memory_mb']['peak_per_step']['max']
                step_time = mem_prof['timing_seconds']['mean_step_time']

                print(f"{technique_name:<20} {forward_mem:<15.2f} {backward_mem:<15.2f} {peak_mem:<15.2f} {step_time:<15.4f}")
            else:
                print(f"{technique_name:<20} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
        else:
            print(f"{technique_name:<20} {'No profiling':<15} {'No profiling':<15} {'No profiling':<15} {'No profiling':<15}")

    print("=" * 100)

    if not has_memory_data:
        print("\nNote: No memory profiling data found. Make sure experiments were run with memory profiling enabled.")


def print_optimization_summary(results: Dict):
    """Print summary of optimization techniques used."""

    print("\n" + "=" * 80)
    print("OPTIMIZATION TECHNIQUES SUMMARY")
    print("=" * 80)

    print(f"{'Technique':<20} {'BF16':<8} {'Flash':<8} {'GradCP':<10} {'Window':<10}")
    print("-" * 80)

    for technique_name in sorted(results.keys()):
        metrics = results[technique_name]

        if 'optimizations' in metrics:
            opts = metrics['optimizations']
            bf16 = '✓' if opts.get('bf16') else '✗'
            flash = '✓' if opts.get('flash_attn') else '✗'
            gradcp = '✓' if opts.get('gradient_checkpointing') else '✗'
            window = str(opts.get('window_size')) if opts.get('window_size') else '✗'

            print(f"{technique_name:<20} {bf16:<8} {flash:<8} {gradcp:<10} {window:<10}")

    print("=" * 80)


def analyze_results(results: Dict):
    """Perform analysis on results."""

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    if not results:
        print("No results to analyze!")
        return

    # Find best perplexity
    best_ppl = min(
        (metrics.get('val_perplexity', float('inf')), name)
        for name, metrics in results.items()
    )
    print(f"\nBest validation perplexity: {best_ppl[0]:.2f} ({best_ppl[1]})")

    # Find fastest training
    fastest = min(
        (metrics.get('training_time_seconds', float('inf')), name)
        for name, metrics in results.items()
    )
    print(f"Fastest training: {fastest[0]:.1f}s ({fastest[1]})")

    # Find lowest memory (if available)
    memory_results = []
    for name, metrics in results.items():
        if 'memory_profiling' in metrics:
            mem_prof = metrics['memory_profiling']
            if 'memory_mb' in mem_prof:
                peak = mem_prof['memory_mb']['peak_per_step']['max']
                memory_results.append((peak, name))

    if memory_results:
        lowest_mem = min(memory_results)
        print(f"Lowest peak memory: {lowest_mem[0]:.2f} MB ({lowest_mem[1]})")

    # Compare to baseline
    if 'baseline' in results:
        baseline = results['baseline']
        baseline_ppl = baseline.get('val_perplexity')
        baseline_time = baseline.get('training_time_seconds')

        print("\nImprovements over baseline:")
        for name, metrics in results.items():
            if name == 'baseline':
                continue

            ppl = metrics.get('val_perplexity')
            time = metrics.get('training_time_seconds')

            if ppl and baseline_ppl:
                ppl_change = ((ppl - baseline_ppl) / baseline_ppl) * 100
                print(f"  {name}: PPL {ppl_change:+.1f}%", end="")

            if time and baseline_time:
                time_change = ((time - baseline_time) / baseline_time) * 100
                print(f", Time {time_change:+.1f}%", end="")

            print()

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Compare Lab 4 experiment results"
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Results directory (default: results/)'
    )
    parser.add_argument(
        '--experiments',
        type=str,
        nargs='+',
        default=None,
        help='Specific experiments to compare (default: all)'
    )

    args = parser.parse_args()

    # Get results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        config = TransformerConfig()
        results_dir = config.results_dir

    # Load results
    print(f"Loading results from: {results_dir}")
    results = load_results(results_dir, args.experiments)

    if not results:
        print("No results found!")
        return

    print(f"Found {len(results)} experiments")

    # Redirect output to file
    comparison_path = results_dir / "comparison_report.txt"

    # Save to file
    import io
    output = io.StringIO()

    # Capture all print output
    original_stdout = sys.stdout
    sys.stdout = output

    # Print comparisons (these will be captured)
    print(f"Loading results from: {results_dir}")
    print(f"Found {len(results)} experiments")
    print_comparison_table(results)
    print_memory_comparison(results)
    print_optimization_summary(results)
    analyze_results(results)

    # Restore stdout
    sys.stdout = original_stdout

    # Get captured output
    report_content = output.getvalue()

    # Print to console
    print(report_content)

    # Write to file
    with open(comparison_path, 'w') as f:
        f.write(report_content)

    print(f"\nComparison report saved to: {comparison_path}")


if __name__ == "__main__":
    main()
