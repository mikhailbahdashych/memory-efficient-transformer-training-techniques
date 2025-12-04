"""
Utility to find the maximum batch size that fits in GPU memory.
Uses binary search to efficiently find the optimal batch size.
"""

import torch
import torch.nn as nn
from typing import Callable, Tuple, Optional
import gc


def find_max_batch_size(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = "cuda",
    min_batch_size: int = 1,
    max_batch_size: int = 512,
    verbose: bool = True,
) -> int:
    """
    Find the maximum batch size that fits in GPU memory using binary search.

    Args:
        model: The model to test
        input_shape: Shape of input WITHOUT batch dimension (e.g., (seq_len,) for sequences)
        device: Device to test on
        min_batch_size: Minimum batch size to try
        max_batch_size: Maximum batch size to try
        verbose: Print progress

    Returns:
        Maximum batch size that fits in memory
    """
    if verbose:
        print("\n" + "=" * 80)
        print("FINDING MAXIMUM BATCH SIZE")
        print("=" * 80)
        print(f"Device: {device}")
        print(f"Input shape (per sample): {input_shape}")
        print(f"Search range: [{min_batch_size}, {max_batch_size}]")
        print()

    model.train()  # Training mode uses more memory
    best_batch_size = min_batch_size

    left, right = min_batch_size, max_batch_size

    while left <= right:
        mid = (left + right) // 2

        if verbose:
            print(f"Testing batch size: {mid}...", end=" ")

        # Test if this batch size fits
        fits = test_batch_size(model, mid, input_shape, device, verbose=False)

        if fits:
            best_batch_size = mid
            if verbose:
                print("✓ Fits")
            # Try larger
            left = mid + 1
        else:
            if verbose:
                print("✗ OOM")
            # Try smaller
            right = mid - 1

        # Clean up
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    if verbose:
        print()
        print("=" * 80)
        print(f"Maximum batch size: {best_batch_size}")
        print("=" * 80)

    return best_batch_size


def test_batch_size(
    model: nn.Module,
    batch_size: int,
    input_shape: Tuple[int, ...],
    device: str = "cuda",
    verbose: bool = False,
) -> bool:
    """
    Test if a specific batch size fits in memory.

    Args:
        model: The model to test
        batch_size: Batch size to test
        input_shape: Shape of input WITHOUT batch dimension
        device: Device to test on
        verbose: Print details

    Returns:
        True if batch size fits, False if OOM
    """
    try:
        # Create dummy input with batch dimension
        full_shape = (batch_size,) + input_shape
        dummy_input = torch.randint(0, model.vocab_size, full_shape, device=device)

        # Create dummy target (for loss calculation)
        dummy_target = torch.randint(0, model.vocab_size, full_shape, device=device)

        # Forward pass
        outputs = model(dummy_input)

        # Calculate loss (simulate training)
        criterion = nn.CrossEntropyLoss()
        batch_size_dim, seq_len, vocab_size = outputs.shape
        outputs_flat = outputs.reshape(-1, vocab_size)
        targets_flat = dummy_target.reshape(-1)
        loss = criterion(outputs_flat, targets_flat)

        # Backward pass (this is where OOM usually happens)
        loss.backward()

        # Clean up
        del dummy_input, dummy_target, outputs, loss
        model.zero_grad()

        return True

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            # Clean up after OOM
            if device == "cuda":
                torch.cuda.empty_cache()
            model.zero_grad()
            return False
        else:
            # Some other error, re-raise
            raise e


def find_max_batch_size_with_dataloader(
    model: nn.Module,
    create_dataloader_fn: Callable[[int], torch.utils.data.DataLoader],
    device: str = "cuda",
    min_batch_size: int = 1,
    max_batch_size: int = 512,
    test_steps: int = 3,
    verbose: bool = True,
) -> int:
    """
    Find maximum batch size using actual dataloader.

    More accurate than synthetic data but slower.

    Args:
        model: The model to test
        create_dataloader_fn: Function that takes batch_size and returns DataLoader
        device: Device to test on
        min_batch_size: Minimum batch size to try
        max_batch_size: Maximum batch size to try
        test_steps: Number of training steps to test per batch size
        verbose: Print progress

    Returns:
        Maximum batch size that fits in memory
    """
    if verbose:
        print("\n" + "=" * 80)
        print("FINDING MAXIMUM BATCH SIZE (with DataLoader)")
        print("=" * 80)
        print(f"Device: {device}")
        print(f"Test steps: {test_steps}")
        print(f"Search range: [{min_batch_size}, {max_batch_size}]")
        print()

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    best_batch_size = min_batch_size
    left, right = min_batch_size, max_batch_size

    while left <= right:
        mid = (left + right) // 2

        if verbose:
            print(f"Testing batch size: {mid}...", end=" ")

        try:
            # Create dataloader with this batch size
            dataloader = create_dataloader_fn(mid)

            # Test a few training steps
            success = True
            for i, (inputs, targets) in enumerate(dataloader):
                if i >= test_steps:
                    break

                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()

                # Forward
                outputs = model(inputs)

                # Loss
                batch_size_dim, seq_len, vocab_size = outputs.shape
                outputs_flat = outputs.reshape(-1, vocab_size)
                targets_flat = targets.reshape(-1)

                # Only compute loss on non-padding tokens
                pad_token_id = getattr(model, 'pad_token_id', 0)
                mask = (targets_flat != pad_token_id)
                loss = criterion(outputs_flat[mask], targets_flat[mask])

                # Backward
                loss.backward()
                optimizer.step()

            # If we got here, it fits
            best_batch_size = mid
            if verbose:
                print("✓ Fits")
            left = mid + 1

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if verbose:
                    print("✗ OOM")
                right = mid - 1
            else:
                raise e

        # Clean up
        del dataloader
        model.zero_grad()
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    if verbose:
        print()
        print("=" * 80)
        print(f"Maximum batch size: {best_batch_size}")
        print("=" * 80)

    return best_batch_size


if __name__ == "__main__":
    # Test with a simple model
    from models.transformer_model import TransformerLanguageModel

    print("Testing batch size finder...")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        device = "cuda"

        # Create a small model for testing
        model = TransformerLanguageModel(
            vocab_size=10000,
            d_model=256,
            num_heads=8,
            num_layers=4,
            ff_dim=1024,
            max_seq_length=128,
        ).to(device)

        # Find max batch size
        max_bs = find_max_batch_size(
            model=model,
            input_shape=(128,),  # sequence length
            device=device,
            min_batch_size=1,
            max_batch_size=256,
        )

        print(f"\nResult: Maximum batch size = {max_bs}")
    else:
        print("CUDA not available, skipping test")
