# Memory-Efficient Transformer Training Techniques

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

This project systematically compares **5 memory optimization techniques** for Transformer training:

1. **Baseline (FP32/TF32)** - Full precision training
2. **BF16 Mixed Precision** - Automatic mixed precision with bfloat16
3. **FlashAttention** - Memory-efficient attention implementation (FA2)
4. **Windowed Attention** - Sliding window attention mechanism
5. **Gradient Checkpointing** - Trading compute for memory

**Model Configuration:**
- Architecture: Transformer decoder (6 layers, 512 d_model, 8 heads, 2048 FFN)
- Tokenizer: BPE (10K vocab) or GPT-2 (50K vocab, pre-trained)
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
│   ├── tokenizer.py                  # BPE and GPT-2 tokenizers
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
# Option A: From source (may take several minutes to compile)
pip install flash-attn --no-build-isolation

# Option B: Pre-built wheels (faster, recommended for specific configurations)
# For Python 3.12.3, PyTorch 2.9.1+cu128, CUDA 12.8:
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.5.4/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_x86_64.whl
```

**Environment Used in This Project:**
- Python: 3.12.3
- PyTorch: 2.9.1+cu128
- CUDA: 12.8.9
- FlashAttention: 2.8.3 (pre-built wheel)

**Note:** FlashAttention requires CUDA and may need compilation if installing from source. For faster installation, use pre-built wheels matching your Python/PyTorch/CUDA versions:
- Check your configuration: `python -c "import torch; print(f'Python: {torch.__version__}, CUDA: {torch.version.cuda}')"`
- Find pre-built wheels: [flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases)
- Official documentation: [flash-attn repository](https://github.com/Dao-AILab/flash-attention)

### Step 2: Verify Installation

```bash
# Test CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print(torch.__version__)"

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

#### Option A: BPE Tokenizer (Default)
```bash
python scripts/preprocess_data.py \
  --input data/raw/shopping_1_general_corpus.txt \
  --file-type txt \
  --tokenizer-type bpe
```

#### Option B: GPT-2 Tokenizer (Pre-trained)
```bash
python scripts/preprocess_data.py \
  --input data/raw/shopping_1_general_corpus.txt \
  --file-type txt \
  --tokenizer-type gpt2
```

**What this does:**
- Splits data into train (85%), val (10%), test (5%)
- Creates/loads tokenizer:
  - **BPE**: Trains custom tokenizer with 10K vocabulary from training data
  - **GPT-2**: Uses pre-trained GPT-2 tokenizer with 50K vocabulary
- Tokenizes all splits
- Saves to `data/processed/shopping_1_general_corpus_*`

**Time estimate:** 5-15 minutes depending on dataset size

**Tokenizer Comparison:**
| Feature | BPE | GPT-2 |
|---------|-----|-------|
| Vocabulary Size | 10,000 (configurable) | 50,257 (fixed) |
| Training Required | Yes (~5 min) | No (pre-trained) |
| Domain Adaptation | Optimized for your corpus | General-purpose |
| OOV Handling | May have unknown tokens | Better generalization |
| Use Case | Domain-specific text | General Polish text |

---

### **STEP 3: Run Experiments**

According to assignment requirements, you need to run a **systematic comparison** with:
- **5 optimization techniques** (Baseline, BF16, FlashAttention, Windowed Attention, Gradient Checkpointing)
- **3 batch sizes** per technique: 32, 64, 128
- **2 window sizes** for windowed attention: seq_len/4 (64) and seq_len/2 (128)

**Total experiments: 18** (5 techniques × 3 batch sizes, with windowed attention testing 2 window sizes)

---

#### **Complete Experiment List (Copy-Paste Ready)**

Run these commands in order. Each experiment takes ~15-25 minutes.

##### **1. Baseline (FP32/TF32) - 3 batch sizes**

```bash
# Baseline with BS=32
python scripts/train.py --technique baseline_bs32 --batch-size 32

# Baseline with BS=64
python scripts/train.py --technique baseline_bs64 --batch-size 64

# Baseline with BS=128
python scripts/train.py --technique baseline_bs128 --batch-size 128
```

##### **2. BF16 Mixed Precision - 3 batch sizes**

```bash
# BF16 with BS=32
python scripts/train.py --bf16 --technique bf16_bs32 --batch-size 32

# BF16 with BS=64
python scripts/train.py --bf16 --technique bf16_bs64 --batch-size 64

# BF16 with BS=128
python scripts/train.py --bf16 --technique bf16_bs128 --batch-size 128
```

##### **3. FlashAttention (with BF16) - 3 batch sizes**

```bash
# FlashAttention with BS=32
python scripts/train.py --flash-attn --bf16 --technique bf16_flash_bs32 --batch-size 32

# FlashAttention with BS=64
python scripts/train.py --flash-attn --bf16 --technique bf16_flash_bs64 --batch-size 64

# FlashAttention with BS=128
python scripts/train.py --flash-attn --bf16 --technique bf16_flash_bs128 --batch-size 128
```

##### **4. Windowed Attention - 2 windows × 3 batch sizes = 6 experiments**

**Window = 64 (seq_len / 4):**
```bash
# Window=64, BS=32
python scripts/train.py --flash-attn --bf16 --window-size 64 --technique bf16_window64_bs32 --batch-size 32

# Window=64, BS=64
python scripts/train.py --flash-attn --bf16 --window-size 64 --technique bf16_window64_bs64 --batch-size 64

# Window=64, BS=128
python scripts/train.py --flash-attn --bf16 --window-size 64 --technique bf16_window64_bs128 --batch-size 128
```

**Window = 128 (seq_len / 2):**
```bash
# Window=128, BS=32
python scripts/train.py --flash-attn --bf16 --window-size 128 --technique bf16_window128_bs32 --batch-size 32

# Window=128, BS=64
python scripts/train.py --flash-attn --bf16 --window-size 128 --technique bf16_window128_bs64 --batch-size 64

# Window=128, BS=128
python scripts/train.py --flash-attn --bf16 --window-size 128 --technique bf16_window128_bs128 --batch-size 128
```

##### **5. Gradient Checkpointing (Baseline only) - 3 batch sizes**

**Note:** Per assignment requirements, gradient checkpointing is only tested on baseline (FP32), not combined with other optimizations.

```bash
# Gradient Checkpointing with BS=32
python scripts/train.py --gradient-checkpointing --technique gradcp_bs32 --batch-size 32

# Gradient Checkpointing with BS=64
python scripts/train.py --gradient-checkpointing --technique gradcp_bs64 --batch-size 64

# Gradient Checkpointing with BS=128
python scripts/train.py --gradient-checkpointing --technique gradcp_bs128 --batch-size 128
```

---

#### **Experiment Summary Table**

| # | Technique | Window | Batch Sizes | Total Runs | Technique Names |
|---|-----------|--------|-------------|------------|-----------------|
| 1 | Baseline (FP32) | N/A | 32, 64, 128 | 3 | `baseline_bs32/64/128` |
| 2 | BF16 | N/A | 32, 64, 128 | 3 | `bf16_bs32/64/128` |
| 3 | FlashAttention | N/A | 32, 64, 128 | 3 | `bf16_flash_bs32/64/128` |
| 4 | Windowed (64) | 64 | 32, 64, 128 | 3 | `bf16_window64_bs32/64/128` |
| 5 | Windowed (128) | 128 | 32, 64, 128 | 3 | `bf16_window128_bs32/64/128` |
| 6 | Gradient Checkpointing | N/A | 32, 64, 128 | 3 | `gradcp_bs32/64/128` |
| **Total** | **6 configurations** | | | **18 experiments** | |

**Time estimate:** 18 experiments × 20 minutes = **~6 hours total**

---

#### **Why These Specific Settings?**

**Batch Sizes (32, 64, 128):**
- Tests how each optimization scales with batch size
- 128 is maximum that baseline can fit
- Shows memory/speed trade-offs at different batch sizes

**Window Sizes (64, 128 only):**
- **64 = seq_len / 4**: Aggressive windowing, significant memory savings
- **128 = seq_len / 2**: Moderate windowing, balance between memory and quality
- **Why not 256/512?** Window ≥ sequence length provides no benefit (equivalent to full attention)

**Gradient Checkpointing (Baseline only):**
- Tests pure gradient checkpointing effect without other optimizations
- Shows memory vs compute trade-off in isolation
- Per assignment requirements, not combined with BF16/FlashAttention

---

#### **Quick Copy-Paste: All 18 Experiments**

```bash
# === BASELINE (FP32) - 3 experiments ===
python scripts/train.py --technique baseline_bs32 --batch-size 32
python scripts/train.py --technique baseline_bs64 --batch-size 64
python scripts/train.py --technique baseline_bs128 --batch-size 128

# === BF16 - 3 experiments ===
python scripts/train.py --bf16 --technique bf16_bs32 --batch-size 32
python scripts/train.py --bf16 --technique bf16_bs64 --batch-size 64
python scripts/train.py --bf16 --technique bf16_bs128 --batch-size 128

# === FLASHATTENTION - 3 experiments ===
python scripts/train.py --flash-attn --bf16 --technique bf16_flash_bs32 --batch-size 32
python scripts/train.py --flash-attn --bf16 --technique bf16_flash_bs64 --batch-size 64
python scripts/train.py --flash-attn --bf16 --technique bf16_flash_bs128 --batch-size 128

# === WINDOWED ATTENTION (window=64) - 3 experiments ===
python scripts/train.py --flash-attn --bf16 --window-size 64 --technique bf16_window64_bs32 --batch-size 32
python scripts/train.py --flash-attn --bf16 --window-size 64 --technique bf16_window64_bs64 --batch-size 64
python scripts/train.py --flash-attn --bf16 --window-size 64 --technique bf16_window64_bs128 --batch-size 128

# === WINDOWED ATTENTION (window=128) - 3 experiments ===
python scripts/train.py --flash-attn --bf16 --window-size 128 --technique bf16_window128_bs32 --batch-size 32
python scripts/train.py --flash-attn --bf16 --window-size 128 --technique bf16_window128_bs64 --batch-size 64
python scripts/train.py --flash-attn --bf16 --window-size 128 --technique bf16_window128_bs128 --batch-size 128

# === GRADIENT CHECKPOINTING (baseline only) - 3 experiments ===
python scripts/train.py --gradient-checkpointing --technique gradcp_bs32 --batch-size 32
python scripts/train.py --gradient-checkpointing --technique gradcp_bs64 --batch-size 64
python scripts/train.py --gradient-checkpointing --technique gradcp_bs128 --batch-size 128
```

**Option A: Run all experiments automatically (recommended):**
```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

**Option B: Run experiments manually (copy-paste commands above)**

**Note:** The automated script (`run_all_experiments.sh`) includes progress tracking and will automatically generate the comparison report when done.

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
num_epochs: int = 1
```

### Data Processing
```python
vocab_size: int = 10000         # Vocabulary size (BPE only, GPT-2 is fixed at 50257)
tokenizer_type: str = "bpe"     # "bpe" or "gpt2"
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

## Quick Reference

### **All 18 Required Experiments - Copy-Paste Ready**

```bash
# ============================================
# BASELINE (FP32) - 3 batch sizes
# ============================================
python scripts/train.py --technique baseline_bs32 --batch-size 32
python scripts/train.py --technique baseline_bs64 --batch-size 64
python scripts/train.py --technique baseline_bs128 --batch-size 128

# ============================================
# BF16 MIXED PRECISION - 3 batch sizes
# ============================================
python scripts/train.py --bf16 --technique bf16_bs32 --batch-size 32
python scripts/train.py --bf16 --technique bf16_bs64 --batch-size 64
python scripts/train.py --bf16 --technique bf16_bs128 --batch-size 128

# ============================================
# FLASHATTENTION - 3 batch sizes
# ============================================
python scripts/train.py --flash-attn --bf16 --technique bf16_flash_bs32 --batch-size 32
python scripts/train.py --flash-attn --bf16 --technique bf16_flash_bs64 --batch-size 64
python scripts/train.py --flash-attn --bf16 --technique bf16_flash_bs128 --batch-size 128

# ============================================
# WINDOWED ATTENTION (window=64) - 3 batch sizes
# ============================================
python scripts/train.py --flash-attn --bf16 --window-size 64 --technique bf16_window64_bs32 --batch-size 32
python scripts/train.py --flash-attn --bf16 --window-size 64 --technique bf16_window64_bs64 --batch-size 64
python scripts/train.py --flash-attn --bf16 --window-size 64 --technique bf16_window64_bs128 --batch-size 128

# ============================================
# WINDOWED ATTENTION (window=128) - 3 batch sizes
# ============================================
python scripts/train.py --flash-attn --bf16 --window-size 128 --technique bf16_window128_bs32 --batch-size 32
python scripts/train.py --flash-attn --bf16 --window-size 128 --technique bf16_window128_bs64 --batch-size 64
python scripts/train.py --flash-attn --bf16 --window-size 128 --technique bf16_window128_bs128 --batch-size 128

# ============================================
# GRADIENT CHECKPOINTING (baseline only) - 3 batch sizes
# ============================================
python scripts/train.py --gradient-checkpointing --technique gradcp_bs32 --batch-size 32
python scripts/train.py --gradient-checkpointing --technique gradcp_bs64 --batch-size 64
python scripts/train.py --gradient-checkpointing --technique gradcp_bs128 --batch-size 128

# ============================================
# COMPARE ALL RESULTS
# ============================================
python scripts/compare_results.py
```

**Total:** 18 experiments × ~20 minutes = ~6 hours

### **Complete Workflow (Setup to Results)**

```bash
# 1. Install dependencies
uv sync

# Install FlashAttention (choose one):
# Option A: Pre-built wheel (recommended, faster)
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.5.4/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_x86_64.whl

# Option B: From source (if pre-built not available)
pip install flash-attn --no-build-isolation

# 2. Download dataset
python main.py shopping_1_general_corpus

# 3. Preprocess (choose tokenizer)
# Option A: BPE tokenizer (10K vocab)
python scripts/preprocess_data.py \
  --input data/raw/shopping_1_general_corpus.txt \
  --file-type txt \
  --tokenizer-type bpe

# Option B: GPT-2 tokenizer (50K vocab)
python scripts/preprocess_data.py \
  --input data/raw/shopping_1_general_corpus.txt \
  --file-type txt \
  --tokenizer-type gpt2

# 4. Run experiments (copy-paste from above section)

# 5. Compare results
python scripts/compare_results.py
```

**Time estimates:**
- Setup (steps 1-3): 30-45 minutes
- All 18 experiments: ~6 hours (18 × 20 min)
- Comparison and analysis: 10-15 minutes
- **Total: ~7 hours**

---

## Additional Resources

- **FlashAttention Paper:** [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691)
- **Gradient Checkpointing:** [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)
- **Mixed Precision Training:** [PyTorch AMP Documentation](https://pytorch.org/docs/stable/amp.html)
- **Speakleash:** [Polish Language Corpora](https://github.com/speakleash/speakleash)

---
