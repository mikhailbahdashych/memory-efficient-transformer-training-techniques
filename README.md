# Lab 4: Memory-Efficient Transformer Training Techniques

**Complete implementation** of memory optimization techniques for Transformer training, comparing their impact on GPU memory usage, batch size capacity, training speed, and model performance.

**Optimized for NVIDIA RTX 5090 (CUDA)**

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Complete Workflow](#complete-workflow)
5. [Optimization Techniques](#optimization-techniques)
6. [Detailed Command Reference](#detailed-command-reference)
7. [Results and Analysis](#results-and-analysis)
8. [Configuration](#configuration)
9. [Troubleshooting](#troubleshooting)

---

## Overview

This lab systematically compares **5 memory optimization techniques** for Transformer training:

1. **Baseline (FP32/TF32)** - Full precision training
2. **BF16 Mixed Precision** - Automatic mixed precision with bfloat16
3. **FlashAttention** - Memory-efficient attention implementation (FA2)
4. **Windowed Attention** - Sliding window attention mechanism
5. **Gradient Checkpointing** - Trading compute for memory

**Model Configuration:**
- Architecture: Transformer decoder (6 layers, 512 d_model, 8 heads, 2048 FFN)
- Vocabulary: 10,000 BPE tokens
- Sequence Length: 512 tokens
- Dataset: Polish text from Speakleash corpus
- Training: 1 epoch per experiment (for fair comparison)

**Measurements Tracked:**
- GPU memory usage (forward, backward, peak)
- Maximum batch size before OOM
- Training time (per step and total)
- Model perplexity (validation set)

---

## Project Structure

```
├── main.py                           # Dataset download from Speakleash
├── models/
│   └── transformer_model.py          # Transformer with FlashAttention support
├── utils/
│   ├── config.py                     # Configuration (GPU-optimized)
│   ├── tokenizer.py                  # BPE tokenizer
│   ├── dataset.py                    # PyTorch dataset
│   ├── metrics.py                    # Evaluation metrics
│   ├── memory_profiler.py            # GPU memory profiling
│   └── batch_size_finder.py          # Max batch size finder
├── scripts/
│   ├── preprocess_data.py            # Data preprocessing
│   ├── train.py                      # Training with optimizations
│   ├── run_experiments.py            # Automated experiment runner
│   └── compare_results.py            # Results comparison
├── data/
│   ├── raw/                          # Downloaded datasets
│   └── processed/                    # Tokenized data
├── checkpoints/                      # Saved models
├── results/                          # Metrics and memory profiles
├── pyproject.toml                    # Dependencies
└── README.md                         # This file
```

---

## Installation

### Prerequisites
- Python 3.12+
- NVIDIA GPU with CUDA support (RTX 5090 recommended)
- `uv` package manager

### Step 1: Install Dependencies

```bash
# Install base dependencies
uv sync

# Install FlashAttention (required for experiments 2-4)
# This may take several minutes to compile
pip install flash-attn --no-build-isolation
```

**Note:** FlashAttention requires CUDA and may need compilation. Check [flash-attn documentation](https://github.com/Dao-AILab/flash-attention) for pre-built wheels for your CUDA version.

### Step 2: Verify Installation

```bash
# Test CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Test FlashAttention (optional)
python -c "from flash_attn import flash_attn_func; print('FlashAttention installed successfully')"
```

---

## Complete Workflow

### **STEP 1: Download Dataset**

List available Polish datasets from Speakleash:

```bash
python main.py --list
```

Download a specific dataset (example: shopping reviews):

```bash
python main.py shopping_1_general_corpus
```

This downloads the dataset to `data/raw/shopping_1_general_corpus.txt`.

**Alternative datasets:**
- `shopping_1_general_corpus` - Product reviews (~2.2GB, 2M documents)
- Other Polish datasets available via `--list` command

---

### **STEP 2: Preprocess Data**

Tokenize and prepare data for training:

```bash
python scripts/preprocess_data.py \
  --input data/raw/shopping_1_general_corpus.txt \
  --file-type txt
```

**What this does:**
- Splits data into train (85%), val (10%), test (5%)
- Trains BPE tokenizer (10k vocabulary)
- Tokenizes all splits
- Saves to `data/processed/shopping_1_general_corpus_*`

**Time estimate:** 5-15 minutes depending on dataset size

---

### **STEP 3: Run Experiments**

You have two options: run individual experiments or automate all experiments.

#### Option A: Run All Experiments Automatically (Recommended)

```bash
python scripts/run_experiments.py
```

This runs all 8 experiments sequentially:
1. baseline
2. bf16
3. flash_attn
4. window_512
5. window_256
6. grad_checkpoint
7. bf16_gradcp
8. flash_gradcp

**Time estimate:** 3-6 hours for all experiments (depends on dataset size and GPU)

**Run specific experiments only:**
```bash
# Only run baseline and BF16
python scripts/run_experiments.py --only baseline bf16

# Run all except windowed attention
python scripts/run_experiments.py --skip window_512 window_256
```

#### Option B: Run Individual Experiments

**Experiment 0: Baseline (FP32/TF32)**
```bash
python scripts/train.py
```

**Experiment 1: BF16 Mixed Precision**
```bash
python scripts/train.py --bf16
```

**Experiment 2: FlashAttention**
```bash
python scripts/train.py --flash-attn --bf16
```

**Experiment 3: Windowed Attention (window=512)**
```bash
python scripts/train.py --flash-attn --bf16 --window-size 512
```

**Experiment 4: Windowed Attention (window=256)**
```bash
python scripts/train.py --flash-attn --bf16 --window-size 256
```

**Experiment 5: Gradient Checkpointing**
```bash
python scripts/train.py --gradient-checkpointing
```

**Experiment 6: Combined (BF16 + Gradient Checkpointing)**
```bash
python scripts/train.py --bf16 --gradient-checkpointing
```

**Experiment 7: Combined (FlashAttention + Gradient Checkpointing)**
```bash
python scripts/train.py --flash-attn --bf16 --gradient-checkpointing
```

---

### **STEP 4: Compare Results**

Generate comparison tables and analysis:

```bash
python scripts/compare_results.py
```

**Output:**
- Comparison table (PPL, memory, time)
- Memory usage details
- Optimization techniques summary
- Analysis (best technique, improvements over baseline)

**Sample output:**
```
================================================================================
EXPERIMENT COMPARISON TABLE
================================================================================
Technique            Batch Size   Val PPL      Train Time      Peak Mem (MB)   Parameters
--------------------------------------------------------------------------------------------------------------------
baseline             64           145.32       1234.5s         8192.45         42,567,680
bf16                 128          144.89       987.2s          4512.32         42,567,680
flash_attn           128          144.76       856.4s          3891.23         42,567,680
window_512           128          146.21       798.1s          3245.67         42,567,680
grad_checkpoint      64           145.45       1567.8s         5123.45         42,567,680
...
```

---

## Optimization Techniques

### 1. Baseline (FP32/TF32)
- **Description:** Full precision training
- **Expected:** Highest memory usage, slowest training
- **Use case:** Establish performance baseline

### 2. BF16 Mixed Precision
- **Description:** Automatic mixed precision with bfloat16
- **Implementation:** `torch.cuda.amp.autocast(dtype=torch.bfloat16)`
- **Expected:** ~50% memory reduction, faster training, same accuracy
- **Use case:** Default choice for most training

### 3. FlashAttention
- **Description:** Memory-efficient attention kernel (FlashAttention 2)
- **Implementation:** `flash_attn_func` with causal masking
- **Expected:** 30-40% memory reduction vs BF16, 20-30% faster
- **Use case:** Large models, long sequences

### 4. Windowed Attention
- **Description:** Sliding window attention (local attention)
- **Implementation:** FlashAttention with `window_size` parameter
- **Expected:** Further memory reduction, slight accuracy drop
- **Use case:** Very long sequences, extreme memory constraints
- **Trade-off:** Smaller window = less memory but lower accuracy

### 5. Gradient Checkpointing
- **Description:** Recompute activations during backward pass
- **Implementation:** `model.gradient_checkpointing_enable()`
- **Expected:** 40-50% memory reduction, 20-30% slower training
- **Use case:** Very large models that don't fit in memory

---

## Detailed Command Reference

### Training Script (`scripts/train.py`)

**Full syntax:**
```bash
python scripts/train.py [OPTIONS]
```

**Options:**
- `--dataset NAME` - Dataset name (auto-detected if only one)
- `--technique NAME` - Custom technique name for results
- `--bf16` - Enable BF16 mixed precision
- `--flash-attn` - Use FlashAttention
- `--gradient-checkpointing` - Enable gradient checkpointing
- `--window-size SIZE` - Window size for windowed attention
- `--batch-size SIZE` - Override batch size
- `--no-memory-profiling` - Disable memory profiling

**Examples:**
```bash
# Baseline with custom batch size
python scripts/train.py --batch-size 32

# All optimizations combined
python scripts/train.py --flash-attn --bf16 --gradient-checkpointing

# Custom technique name
python scripts/train.py --bf16 --technique my_experiment
```

### Experiment Runner (`scripts/run_experiments.py`)

**Full syntax:**
```bash
python scripts/run_experiments.py [OPTIONS]
```

**Options:**
- `--dataset NAME` - Dataset name
- `--batch-size SIZE` - Override batch size for all experiments
- `--skip EXP1 EXP2` - Skip specific experiments
- `--only EXP1 EXP2` - Run only specific experiments

**Examples:**
```bash
# Run all with smaller batch size (if GPU OOM)
python scripts/run_experiments.py --batch-size 32

# Quick test: only baseline and BF16
python scripts/run_experiments.py --only baseline bf16
```

### Results Comparison (`scripts/compare_results.py`)

**Full syntax:**
```bash
python scripts/compare_results.py [OPTIONS]
```

**Options:**
- `--results-dir DIR` - Results directory (default: results/)
- `--experiments EXP1 EXP2` - Compare specific experiments only

---

## Results and Analysis

### Output Files

After running experiments, you'll find:

**In `checkpoints/`:**
- `{dataset}_{technique}.pt` - Model checkpoints

**In `results/`:**
- `{dataset}_{technique}_metrics.json` - Training metrics
- `{dataset}_{technique}_memory.json` - Memory profiling
- `experiment_summary.json` - Overall summary
- `comparison_report.txt` - Comparison analysis

### Metrics Tracked

**Training Metrics:**
- Train loss and perplexity
- Validation loss and perplexity
- Training time (total and per step)
- Model parameters

**Memory Metrics:**
- Forward pass memory
- Backward pass memory
- Peak memory per step
- Mean and max values

**Model Configuration:**
- Architecture parameters
- Optimization flags used
- Batch size

### Expected Results (Example)

Based on typical RTX 5090 performance:

| Technique | Batch Size | Val PPL | Time (min) | Peak Mem (GB) | Speedup |
|-----------|-----------|---------|------------|---------------|---------|
| Baseline | 64 | 145.3 | 25.0 | 18.2 | 1.0x |
| BF16 | 128 | 144.9 | 16.5 | 9.5 | 1.5x |
| FlashAttn | 128 | 144.8 | 14.2 | 8.2 | 1.8x |
| Window512 | 128 | 146.2 | 13.3 | 6.8 | 1.9x |
| GradCP | 64 | 145.4 | 31.2 | 11.4 | 0.8x |

*Note: Actual results depend on dataset, model size, and GPU.*

---

## Configuration

Edit `utils/config.py` to adjust settings:

### Model Architecture
```python
embedding_dim: int = 512  # Model dimension
num_heads: int = 8        # Attention heads
num_layers: int = 6       # Transformer layers
ff_dim: int = 2048        # Feed-forward dimension
```

### Training Hyperparameters
```python
batch_size: int = 64      # Batch size
max_seq_length: int = 512 # Sequence length
learning_rate: float = 0.001
num_epochs: int = 1       # Lab 4: 1 epoch only
```

### Data Processing
```python
vocab_size: int = 10000   # BPE vocabulary size
train_split: float = 0.85
val_split: float = 0.10
test_split: float = 0.05
```

---

## Troubleshooting

### GPU Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size: `--batch-size 32` (or 16, 8)
2. Reduce sequence length in `utils/config.py`: `max_seq_length = 256`
3. Reduce model size:
   ```python
   num_layers = 4        # Instead of 6
   embedding_dim = 256   # Instead of 512
   ff_dim = 1024         # Instead of 2048
   ```
4. Use gradient checkpointing: `--gradient-checkpointing`

### FlashAttention Import Error

**Symptoms:**
```
ImportError: cannot import name 'flash_attn_func'
```

**Solutions:**
1. Install FlashAttention:
   ```bash
   pip install flash-attn --no-build-isolation
   ```
2. Check CUDA version compatibility
3. If installation fails, skip FlashAttention experiments:
   ```bash
   python scripts/run_experiments.py --skip flash_attn window_512 window_256 flash_gradcp
   ```

### Slow Training

**If training is too slow:**
1. Use smaller dataset (subsample data before preprocessing)
2. Reduce sequence length: `max_seq_length = 256`
3. Reduce model layers: `num_layers = 4`
4. Use FlashAttention for speedup

### Dataset Download Issues

**If Speakleash download fails:**
1. Check internet connection
2. Try different dataset: `python main.py --list`
3. Use smaller dataset for testing

### No Results Found

**If `compare_results.py` shows no results:**
1. Check `results/` directory exists
2. Verify experiments completed successfully
3. Look for `*_metrics.json` files in `results/`

---

## Report Requirements

For the Lab 4 report, include:

### 1. Experimental Setup
- Dataset description (size, domain)
- Model architecture (layers, dimensions)
- Hardware (GPU model, memory)

### 2. Results Tables
Use output from `compare_results.py`:
- Comparison table with all metrics
- Memory usage comparison
- Optimization techniques summary

### 3. Analysis
- Which technique is most effective?
- Memory vs speed vs accuracy trade-offs
- Why some techniques reduce memory more
- Why some techniques slow down training
- Recommendations for different scenarios

### 4. Visualizations (Optional)
- Memory usage bar charts
- Training time comparison
- Perplexity vs memory scatter plot

---

## Quick Reference

**Complete workflow (copy-paste):**
```bash
# 1. Install dependencies
uv sync
pip install flash-attn --no-build-isolation

# 2. Download dataset
python main.py shopping_1_general_corpus

# 3. Preprocess
python scripts/preprocess_data.py \
  --input data/raw/shopping_1_general_corpus.txt \
  --file-type txt

# 4. Run all experiments
python scripts/run_experiments.py

# 5. Compare results
python scripts/compare_results.py
```

**Time estimate:** 4-7 hours total (mostly training time)

---

## Additional Resources

- **FlashAttention Paper:** [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691)
- **Gradient Checkpointing:** [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)
- **Mixed Precision Training:** [PyTorch AMP Documentation](https://pytorch.org/docs/stable/amp.html)
- **Speakleash:** [Polish Language Corpora](https://github.com/speakleash/speakleash)

---
