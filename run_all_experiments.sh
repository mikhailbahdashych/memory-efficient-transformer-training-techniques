#!/bin/bash

# ============================================
# Memory-Efficient Transformer Training
# All 18 Required Experiments
# ============================================
#
# This script runs all required experiments for the assignment:
# - 5 optimization techniques
# - 3 batch sizes each (32, 64, 128)
# - 2 window sizes for windowed attention
#
# Total: 18 experiments × ~20 minutes = ~6 hours
#
# ============================================

set -e  # Exit on error

echo "=========================================="
echo "Starting All 18 Experiments"
echo "=========================================="
echo ""
echo "Estimated total time: ~6 hours"
echo "Progress will be saved after each experiment"
echo ""

# Track progress
TOTAL_EXPERIMENTS=18
CURRENT=0

# ============================================
# BASELINE (FP32) - 3 experiments
# ============================================
echo ""
echo "=========================================="
echo "BASELINE (FP32) - 3 batch sizes"
echo "=========================================="

CURRENT=$((CURRENT + 1))
echo "[$CURRENT/$TOTAL_EXPERIMENTS] Baseline BS=32..."
python scripts/train.py --technique baseline_bs32 --batch-size 32

CURRENT=$((CURRENT + 1))
echo "[$CURRENT/$TOTAL_EXPERIMENTS] Baseline BS=64..."
python scripts/train.py --technique baseline_bs64 --batch-size 64

CURRENT=$((CURRENT + 1))
echo "[$CURRENT/$TOTAL_EXPERIMENTS] Baseline BS=128..."
python scripts/train.py --technique baseline_bs128 --batch-size 128

# ============================================
# BF16 - 3 experiments
# ============================================
echo ""
echo "=========================================="
echo "BF16 MIXED PRECISION - 3 batch sizes"
echo "=========================================="

CURRENT=$((CURRENT + 1))
echo "[$CURRENT/$TOTAL_EXPERIMENTS] BF16 BS=32..."
python scripts/train.py --bf16 --technique bf16_bs32 --batch-size 32

CURRENT=$((CURRENT + 1))
echo "[$CURRENT/$TOTAL_EXPERIMENTS] BF16 BS=64..."
python scripts/train.py --bf16 --technique bf16_bs64 --batch-size 64

CURRENT=$((CURRENT + 1))
echo "[$CURRENT/$TOTAL_EXPERIMENTS] BF16 BS=128..."
python scripts/train.py --bf16 --technique bf16_bs128 --batch-size 128

# ============================================
# FLASHATTENTION - 3 experiments
# ============================================
echo ""
echo "=========================================="
echo "FLASHATTENTION - 3 batch sizes"
echo "=========================================="

CURRENT=$((CURRENT + 1))
echo "[$CURRENT/$TOTAL_EXPERIMENTS] FlashAttention BS=32..."
python scripts/train.py --flash-attn --bf16 --technique bf16_flash_bs32 --batch-size 32

CURRENT=$((CURRENT + 1))
echo "[$CURRENT/$TOTAL_EXPERIMENTS] FlashAttention BS=64..."
python scripts/train.py --flash-attn --bf16 --technique bf16_flash_bs64 --batch-size 64

CURRENT=$((CURRENT + 1))
echo "[$CURRENT/$TOTAL_EXPERIMENTS] FlashAttention BS=128..."
python scripts/train.py --flash-attn --bf16 --technique bf16_flash_bs128 --batch-size 128

# ============================================
# WINDOWED ATTENTION (window=64) - 3 experiments
# ============================================
echo ""
echo "=========================================="
echo "WINDOWED ATTENTION (window=64) - 3 batch sizes"
echo "=========================================="

CURRENT=$((CURRENT + 1))
echo "[$CURRENT/$TOTAL_EXPERIMENTS] Window=64 BS=32..."
python scripts/train.py --flash-attn --bf16 --window-size 64 --technique bf16_window64_bs32 --batch-size 32

CURRENT=$((CURRENT + 1))
echo "[$CURRENT/$TOTAL_EXPERIMENTS] Window=64 BS=64..."
python scripts/train.py --flash-attn --bf16 --window-size 64 --technique bf16_window64_bs64 --batch-size 64

CURRENT=$((CURRENT + 1))
echo "[$CURRENT/$TOTAL_EXPERIMENTS] Window=64 BS=128..."
python scripts/train.py --flash-attn --bf16 --window-size 64 --technique bf16_window64_bs128 --batch-size 128

# ============================================
# WINDOWED ATTENTION (window=128) - 3 experiments
# ============================================
echo ""
echo "=========================================="
echo "WINDOWED ATTENTION (window=128) - 3 batch sizes"
echo "=========================================="

CURRENT=$((CURRENT + 1))
echo "[$CURRENT/$TOTAL_EXPERIMENTS] Window=128 BS=32..."
python scripts/train.py --flash-attn --bf16 --window-size 128 --technique bf16_window128_bs32 --batch-size 32

CURRENT=$((CURRENT + 1))
echo "[$CURRENT/$TOTAL_EXPERIMENTS] Window=128 BS=64..."
python scripts/train.py --flash-attn --bf16 --window-size 128 --technique bf16_window128_bs64 --batch-size 64

CURRENT=$((CURRENT + 1))
echo "[$CURRENT/$TOTAL_EXPERIMENTS] Window=128 BS=128..."
python scripts/train.py --flash-attn --bf16 --window-size 128 --technique bf16_window128_bs128 --batch-size 128

# ============================================
# GRADIENT CHECKPOINTING - 3 experiments
# ============================================
echo ""
echo "=========================================="
echo "GRADIENT CHECKPOINTING (baseline only) - 3 batch sizes"
echo "=========================================="

CURRENT=$((CURRENT + 1))
echo "[$CURRENT/$TOTAL_EXPERIMENTS] GradCP BS=32..."
python scripts/train.py --gradient-checkpointing --technique gradcp_bs32 --batch-size 32

CURRENT=$((CURRENT + 1))
echo "[$CURRENT/$TOTAL_EXPERIMENTS] GradCP BS=64..."
python scripts/train.py --gradient-checkpointing --technique gradcp_bs64 --batch-size 64

CURRENT=$((CURRENT + 1))
echo "[$CURRENT/$TOTAL_EXPERIMENTS] GradCP BS=128..."
python scripts/train.py --gradient-checkpointing --technique gradcp_bs128 --batch-size 128

# ============================================
# DONE - Generate comparison
# ============================================
echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETED!"
echo "=========================================="
echo ""
echo "Generating comparison report..."
python scripts/compare_results.py

echo ""
echo "=========================================="
echo "DONE!"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  - results/*_metrics.json (individual experiment results)"
echo "  - results/comparison_report.txt (comparison table)"
echo ""
echo "Next steps:"
echo "  1. Review results/comparison_report.txt"
echo "  2. Analyze memory vs batch size trade-offs"
echo "  3. Write your report with findings"
echo ""
