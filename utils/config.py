"""
Configuration file for memory-efficient Transformer training (Lab 4).
Supports various memory optimization techniques.
"""

import torch
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Base configuration for Lab 4 experiments."""

    # Dataset name (will be set from preprocessing/training)
    dataset_name: str = "default"

    # Paths
    project_root: Path = Path(__file__).parent.parent
    data_raw_dir: Path = project_root / "data" / "raw"
    data_processed_dir: Path = project_root / "data" / "processed"
    checkpoints_dir: Path = project_root / "checkpoints"
    results_dir: Path = project_root / "results"

    # Device configuration (GPU-optimized for RTX 5090)
    device: str = "cuda"  # Lab 4 requires CUDA for FlashAttention

    # Data processing
    vocab_size: int = 10000
    max_seq_length: int = 256  # Reduced to 256 for memory efficiency
    train_split: float = 0.85
    val_split: float = 0.10
    test_split: float = 0.05

    # Tokenizer
    tokenizer_type: str = "bpe"  # "bpe" or "gpt2"
    min_frequency: int = 2

    # Training hyperparameters (GPU-optimized)
    batch_size: int = 16  # Reduced for large vocab (GPT-2 = 50k tokens)
    num_epochs: int = 1  # Lab 4: Only train for 1 epoch
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    gradient_clip: float = 1.0

    # Model saving
    save_every_n_epochs: int = 1

    # Evaluation
    eval_batch_size: int = 16  # Match training batch size

    def __post_init__(self):
        """Create directories if they don't exist."""
        self.data_processed_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class TransformerConfig(Config):
    """Configuration for Transformer model with memory optimization options."""

    # Model architecture (conservative for large vocab size)
    # NOTE: With GPT-2 tokenizer (vocab=50257), memory usage is much higher!
    # These settings are balanced for both BPE (vocab=10k) and GPT-2 (vocab=50k)
    embedding_dim: int = 256  # Reduced to 256 for memory efficiency
    num_heads: int = 8
    num_layers: int = 4  # Reduced to 4 layers
    ff_dim: int = 1024  # Reduced to 1024
    dropout: float = 0.1

    # Memory optimization flags
    use_bf16: bool = False  # BF16 mixed precision
    use_flash_attn: bool = False  # FlashAttention
    use_gradient_checkpointing: bool = False  # Gradient checkpointing
    window_size: int = None  # None = full attention, int = windowed attention

    # Memory profiling
    profile_memory: bool = True  # Enable memory profiling by default
    warmup_steps: int = 20  # Steps before recording memory stats

    def __repr__(self):
        opts = []
        if self.use_bf16:
            opts.append("BF16")
        if self.use_flash_attn:
            opts.append("FlashAttn")
        if self.use_gradient_checkpointing:
            opts.append("GradCP")
        if self.window_size:
            opts.append(f"Window={self.window_size}")

        opt_str = f" + {'+'.join(opts)}" if opts else ""
        return (f"Transformer(layers={self.num_layers}, heads={self.num_heads}, "
                f"emb={self.embedding_dim}, ff={self.ff_dim}){opt_str}")


def get_config(config_type: str = "transformer") -> TransformerConfig:
    """
    Get Transformer configuration for Lab 4.

    Args:
        config_type: Type of configuration (only 'transformer' is supported)

    Returns:
        TransformerConfig object
    """
    if config_type != "transformer":
        raise ValueError(f"Only 'transformer' config is supported in Lab 4, got: {config_type}")
    return TransformerConfig()


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    print("Lab 4 Transformer Config:")
    print(f"  Device: {config.device}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Model: {config}")
