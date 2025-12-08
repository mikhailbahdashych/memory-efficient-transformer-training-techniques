"""
Automated experiment runner for Lab 4.
Runs all required memory optimization experiments systematically.
"""

import sys
from pathlib import Path
import argparse
import json
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import TransformerConfig
from scripts.train import train


def run_all_experiments(
    dataset: str = None,
    batch_size: int = None,
    skip_experiments: list = None,
):
    """
    Run all Lab 4 experiments.

    Experiments:
    0. Baseline (FP32/TF32)
    1. BF16 mixed precision
    2. FlashAttention
    3. Windowed attention (multiple window sizes)
    4. Gradient checkpointing
    5. Combined optimizations

    Args:
        dataset: Dataset name
        batch_size: Override default batch size
        skip_experiments: List of experiment names to skip
    """
    skip_experiments = skip_experiments or []

    experiments = [
        {
            'name': 'baseline',
            'description': 'Baseline (FP32/TF32)',
            'config': {
                'use_bf16': False,
                'use_flash_attn': False,
                'use_gradient_checkpointing': False,
                'window_size': None,
            }
        },
        {
            'name': 'bf16',
            'description': 'BF16 Mixed Precision',
            'config': {
                'use_bf16': True,
                'use_flash_attn': False,
                'use_gradient_checkpointing': False,
                'window_size': None,
            }
        },
        {
            'name': 'flash_attn',
            'description': 'FlashAttention',
            'config': {
                'use_bf16': True,  # FlashAttention requires BF16
                'use_flash_attn': True,
                'use_gradient_checkpointing': False,
                'window_size': None,
            }
        },
        {
            'name': 'window_512',
            'description': 'Windowed Attention (window=512)',
            'config': {
                'use_bf16': True,
                'use_flash_attn': True,
                'use_gradient_checkpointing': False,
                'window_size': 512,
            }
        },
        {
            'name': 'window_256',
            'description': 'Windowed Attention (window=256)',
            'config': {
                'use_bf16': True,
                'use_flash_attn': True,
                'use_gradient_checkpointing': False,
                'window_size': 256,
            }
        },
        {
            'name': 'grad_checkpoint',
            'description': 'Gradient Checkpointing',
            'config': {
                'use_bf16': False,
                'use_flash_attn': False,
                'use_gradient_checkpointing': True,
                'window_size': None,
            }
        },
        {
            'name': 'bf16_gradcp',
            'description': 'BF16 + Gradient Checkpointing',
            'config': {
                'use_bf16': True,
                'use_flash_attn': False,
                'use_gradient_checkpointing': True,
                'window_size': None,
            }
        },
        {
            'name': 'flash_gradcp',
            'description': 'FlashAttention + Gradient Checkpointing',
            'config': {
                'use_bf16': True,
                'use_flash_attn': True,
                'use_gradient_checkpointing': True,
                'window_size': None,
            }
        },
    ]

    print("=" * 80)
    print("LAB 4: AUTOMATED EXPERIMENT RUNNER")
    print("=" * 80)
    print(f"\nTotal experiments: {len(experiments)}")
    print(f"Dataset: {dataset or 'auto-detect'}")
    if batch_size:
        print(f"Batch size override: {batch_size}")
    if skip_experiments:
        print(f"Skipping: {', '.join(skip_experiments)}")
    print()

    results = {}
    failed_experiments = []

    for i, exp in enumerate(experiments, 1):
        exp_name = exp['name']

        if exp_name in skip_experiments:
            print(f"\n[{i}/{len(experiments)}] Skipping: {exp['description']}")
            continue

        print("\n" + "=" * 80)
        print(f"EXPERIMENT {i}/{len(experiments)}: {exp['description']}")
        print("=" * 80)

        try:
            # Create config for this experiment
            config = TransformerConfig()

            # Apply experiment settings
            for key, value in exp['config'].items():
                setattr(config, key, value)

            # Override batch size if specified
            if batch_size:
                config.batch_size = batch_size

            # Run training
            start_time = time.time()
            metrics = train(
                dataset=dataset,
                config=config,
                technique_name=exp_name,
            )
            elapsed = time.time() - start_time

            results[exp_name] = {
                'success': True,
                'metrics': metrics,
                'elapsed_time': elapsed,
            }

            print(f"\n+ Experiment '{exp_name}' completed in {elapsed/60:.2f} minutes")

        except Exception as e:
            print(f"\n- Experiment '{exp_name}' failed: {str(e)}")
            failed_experiments.append(exp_name)
            results[exp_name] = {
                'success': False,
                'error': str(e),
            }

    # Summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    successful = [name for name, res in results.items() if res.get('success')]
    print(f"\nSuccessful experiments: {len(successful)}/{len(experiments)}")
    for name in successful:
        metrics = results[name]['metrics']
        print(f"  + {name}: PPL={metrics['val_perplexity']:.2f}, Time={results[name]['elapsed_time']/60:.2f}m")

    if failed_experiments:
        print(f"\nFailed experiments: {len(failed_experiments)}")
        for name in failed_experiments:
            print(f"  - {name}: {results[name]['error']}")

    # Save summary
    config = TransformerConfig()
    if dataset:
        config.dataset_name = dataset

    summary_path = config.results_dir / "experiment_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'experiments': results,
            'summary': {
                'total': len(experiments),
                'successful': len(successful),
                'failed': len(failed_experiments),
            }
        }, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Run all Lab 4 experiments automatically"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Dataset name (auto-detected if only one exists)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override batch size for all experiments'
    )
    parser.add_argument(
        '--skip',
        type=str,
        nargs='+',
        default=[],
        help='Experiments to skip (e.g., --skip baseline bf16)'
    )
    parser.add_argument(
        '--only',
        type=str,
        nargs='+',
        default=None,
        help='Only run specific experiments (e.g., --only baseline bf16)'
    )

    args = parser.parse_args()

    # Determine which experiments to skip
    if args.only:
        # If --only specified, skip everything except those
        all_experiments = [
            'baseline', 'bf16', 'flash_attn', 'window_512', 'window_256',
            'grad_checkpoint', 'bf16_gradcp', 'flash_gradcp'
        ]
        skip_experiments = [exp for exp in all_experiments if exp not in args.only]
    else:
        skip_experiments = args.skip

    run_all_experiments(
        dataset=args.dataset,
        batch_size=args.batch_size,
        skip_experiments=skip_experiments,
    )


if __name__ == "__main__":
    main()
