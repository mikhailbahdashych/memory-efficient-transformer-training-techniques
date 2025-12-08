"""
Enhanced training script for memory-efficient Transformer (Lab 4).
Supports baseline and various memory optimization techniques with comprehensive profiling.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import argparse
import time
import json
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import TransformerConfig
from utils.dataset import create_dataloaders
from utils.metrics import evaluate_model, calculate_perplexity
from utils.memory_profiler import MemoryProfiler, print_gpu_memory_info
from models.transformer_model import TransformerLanguageModel


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    pad_token_id: int,
    gradient_clip: float = 1.0,
    use_bf16: bool = False,
    memory_profiler: MemoryProfiler = None,
) -> tuple[float, float]:
    """
    Train for one epoch with memory profiling.

    Returns:
        (average_loss, epoch_time)
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0
    start_time = time.time()

    progress_bar = tqdm(dataloader, desc="Training")

    for step, (inputs, targets) in enumerate(progress_bar):
        # Start profiling step
        if memory_profiler:
            memory_profiler.start_step()

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass with optional BF16 autocast
        if use_bf16:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(inputs)
                # Calculate loss
                batch_size, seq_len, vocab_size = outputs.shape
                outputs_flat = outputs.reshape(-1, vocab_size)
                targets_flat = targets.reshape(-1)
                mask = (targets_flat != pad_token_id)
                loss = criterion(outputs_flat[mask], targets_flat[mask])
        else:
            outputs = model(inputs)
            # Calculate loss
            batch_size, seq_len, vocab_size = outputs.shape
            outputs_flat = outputs.reshape(-1, vocab_size)
            targets_flat = targets.reshape(-1)
            mask = (targets_flat != pad_token_id)
            loss = criterion(outputs_flat[mask], targets_flat[mask])

        # Record memory after forward
        if memory_profiler:
            memory_profiler.after_forward()

        # Backward pass
        loss.backward()

        # Record memory after backward
        if memory_profiler:
            memory_profiler.after_backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        # Update weights
        optimizer.step()

        # End profiling step
        if memory_profiler:
            memory_profiler.end_step()

        # Track statistics
        num_tokens = mask.sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    epoch_time = time.time() - start_time
    return avg_loss, epoch_time


def train(
    dataset: str = None,
    config: TransformerConfig = None,
    technique_name: str = "baseline",
):
    """
    Main training function for Lab 4.

    Args:
        dataset: Dataset name (e.g., 'shopping_1_general_corpus')
        config: TransformerConfig object with optimization settings
        technique_name: Name of optimization technique for results naming
    """
    if config is None:
        config = TransformerConfig()

    print("=" * 80)
    print(f"TRAINING TRANSFORMER - LAB 4")
    print("=" * 80)
    print(f"\nTechnique: {technique_name}")
    print(f"Configuration: {config}")
    print(f"Device: {config.device}")

    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! Lab 4 requires GPU.")

    # Print GPU info
    print_gpu_memory_info()

    # Auto-detect dataset if not specified
    if dataset is None:
        processed_dir = config.data_processed_dir
        tokenizer_files = list(processed_dir.glob("*_tokenizer.json"))
        if len(tokenizer_files) == 0:
            raise ValueError(
                "No preprocessed data found. Run preprocessing first:\n"
                "  python scripts/preprocess_data.py --input data/raw/<file>.txt"
            )
        elif len(tokenizer_files) > 1:
            print("\nMultiple datasets found:")
            for i, tf in enumerate(tokenizer_files):
                dataset_name = tf.stem.replace("_tokenizer", "")
                print(f"  {i+1}. {dataset_name}")
            raise ValueError(
                "Multiple datasets found. Please specify --dataset <name>"
            )
        else:
            dataset = tokenizer_files[0].stem.replace("_tokenizer", "")
            print(f"\nAuto-detected dataset: {dataset}")

    config.dataset_name = dataset

    # Load data
    print("\nLoading preprocessed data...")
    train_loader, val_loader, test_loader, tokenizer = create_dataloaders(
        config.data_processed_dir,
        dataset,
        config.batch_size,
        config.eval_batch_size,
        config.max_seq_length,
        config.device,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Batch size: {config.batch_size}")

    # Update vocab size from tokenizer (important for GPT-2 tokenizer with 50257 tokens)
    actual_vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {actual_vocab_size}")

    # Initialize model
    print("\nInitializing model...")
    model = TransformerLanguageModel(
        vocab_size=actual_vocab_size,
        d_model=config.embedding_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        ff_dim=config.ff_dim,
        max_seq_length=config.max_seq_length,
        dropout=config.dropout,
        use_flash_attn=config.use_flash_attn,
        window_size=config.window_size,
    ).to(config.device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Enable gradient checkpointing if requested
    if config.use_gradient_checkpointing:
        print("Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()

    # Print memory after model initialization
    print_gpu_memory_info()

    # Setup memory profiler
    memory_profiler = None
    if config.profile_memory:
        print(f"\nMemory profiling enabled (warmup: {config.warmup_steps} steps)")
        memory_profiler = MemoryProfiler(
            device=config.device,
            warmup_steps=config.warmup_steps,
        )

    # Setup training
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    # Training loop (1 epoch for Lab 4)
    train_loss, train_time = train_epoch(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=config.device,
        pad_token_id=tokenizer.pad_token_id,
        gradient_clip=config.gradient_clip,
        use_bf16=config.use_bf16,
        memory_profiler=memory_profiler,
    )

    print("\nEvaluating on validation set...")
    val_loss = evaluate_model(
        model=model,
        dataloader=val_loader,
        criterion=criterion,
        device=config.device,
        pad_token_id=tokenizer.pad_token_id,
    )

    train_ppl = calculate_perplexity(train_loss)
    val_ppl = calculate_perplexity(val_loss)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
    print(f"Val Loss:   {val_loss:.4f} | Val PPL:   {val_ppl:.2f}")
    print(f"Training time: {train_time:.1f}s ({train_time/60:.2f}m)")

    # Print memory profiling summary
    if memory_profiler:
        memory_profiler.print_summary()

    # Save checkpoint
    checkpoint_name = f"{dataset}_{technique_name}.pt"
    checkpoint_path = config.checkpoints_dir / checkpoint_name
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': config,
        'technique': technique_name,
    }, checkpoint_path)
    print(f"\nCheckpoint saved: {checkpoint_path}")

    # Prepare metrics
    metrics = {
        'technique': technique_name,
        'dataset': dataset,
        'train_loss': float(train_loss),
        'val_loss': float(val_loss),
        'train_perplexity': float(train_ppl),
        'val_perplexity': float(val_ppl),
        'training_time_seconds': float(train_time),
        'num_parameters': num_params,
        'batch_size': config.batch_size,
        'model_config': {
            'embedding_dim': config.embedding_dim,
            'num_heads': config.num_heads,
            'num_layers': config.num_layers,
            'ff_dim': config.ff_dim,
            'max_seq_length': config.max_seq_length,
        },
        'optimizations': {
            'bf16': config.use_bf16,
            'flash_attn': config.use_flash_attn,
            'gradient_checkpointing': config.use_gradient_checkpointing,
            'window_size': config.window_size,
        },
    }

    # Add memory profiling results
    if memory_profiler:
        memory_summary = memory_profiler.get_summary()
        if memory_summary.get('status') != 'no_data':
            metrics['memory_profiling'] = memory_summary

    # Save metrics
    metrics_name = f"{dataset}_{technique_name}_metrics.json"
    metrics_path = config.results_dir / metrics_name
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")

    # Save memory profiling separately
    if memory_profiler and memory_profiler.get_summary().get('status') != 'no_data':
        memory_path = config.results_dir / f"{dataset}_{technique_name}_memory.json"
        memory_profiler.save_summary(memory_path)
        print(f"Memory profile saved: {memory_path}")

    print("=" * 80)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train Transformer with memory optimizations (Lab 4)"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Dataset name (auto-detected if only one exists)'
    )
    parser.add_argument(
        '--technique',
        type=str,
        default='baseline',
        help='Technique name for results (e.g., baseline, bf16, flash_attn)'
    )
    parser.add_argument(
        '--bf16',
        action='store_true',
        help='Use BF16 mixed precision'
    )
    parser.add_argument(
        '--flash-attn',
        action='store_true',
        help='Use FlashAttention (requires flash-attn package)'
    )
    parser.add_argument(
        '--gradient-checkpointing',
        action='store_true',
        help='Enable gradient checkpointing'
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=None,
        help='Window size for windowed attention (None = full attention)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for training (default from config)'
    )
    parser.add_argument(
        '--no-memory-profiling',
        action='store_true',
        help='Disable memory profiling'
    )

    args = parser.parse_args()

    # Create config with optimization flags
    config = TransformerConfig()
    config.use_bf16 = args.bf16
    config.use_flash_attn = args.flash_attn
    config.use_gradient_checkpointing = args.gradient_checkpointing
    config.window_size = args.window_size
    config.profile_memory = not args.no_memory_profiling

    if args.batch_size is not None:
        config.batch_size = args.batch_size

    # Determine technique name if not specified
    if args.technique == 'baseline' and (args.bf16 or args.flash_attn or args.gradient_checkpointing or args.window_size):
        # Auto-generate technique name
        parts = []
        if args.bf16:
            parts.append('bf16')
        if args.flash_attn:
            if args.window_size:
                parts.append(f'window{args.window_size}')
            else:
                parts.append('flash')
        if args.gradient_checkpointing:
            parts.append('gradcp')
        technique_name = '_'.join(parts) if parts else 'baseline'
    else:
        technique_name = args.technique

    train(dataset=args.dataset, config=config, technique_name=technique_name)


if __name__ == "__main__":
    main()
