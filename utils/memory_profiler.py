"""
Memory profiling utilities for tracking GPU memory usage during training.
Optimized for CUDA (RTX 5090).
"""

import torch
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
import json


@dataclass
class MemorySnapshot:
    """Single memory measurement snapshot."""
    step: int
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    timestamp: float


class MemoryProfiler:
    """
    Profile GPU memory usage during training.

    Tracks memory at key points:
    - Before forward pass
    - After forward pass
    - After backward pass
    - Peak memory per step
    """

    def __init__(self, device: str = "cuda", warmup_steps: int = 20):
        """
        Initialize memory profiler.

        Args:
            device: Device to profile (should be "cuda")
            warmup_steps: Number of warmup steps before recording stats
        """
        self.device = device
        self.warmup_steps = warmup_steps
        self.current_step = 0

        # Memory tracking
        self.snapshots: List[MemorySnapshot] = []
        self.forward_memory: List[float] = []
        self.backward_memory: List[float] = []
        self.peak_memory: List[float] = []

        # Timing
        self.step_times: List[float] = []
        self.step_start_time: Optional[float] = None

        # Summary statistics
        self.is_recording = False

    def start_step(self):
        """Mark the start of a training step."""
        self.step_start_time = time.time()
        if self.current_step >= self.warmup_steps:
            self.is_recording = True
            torch.cuda.reset_peak_memory_stats(self.device)

    def after_forward(self):
        """Record memory after forward pass."""
        if self.is_recording:
            allocated = torch.cuda.memory_allocated(self.device) / 1024**2  # MB
            self.forward_memory.append(allocated)

    def after_backward(self):
        """Record memory after backward pass."""
        if self.is_recording:
            allocated = torch.cuda.memory_allocated(self.device) / 1024**2  # MB
            self.backward_memory.append(allocated)

            # Record peak memory for this step
            peak = torch.cuda.max_memory_allocated(self.device) / 1024**2  # MB
            self.peak_memory.append(peak)

    def end_step(self):
        """Mark the end of a training step and record statistics."""
        if self.step_start_time is not None:
            step_time = time.time() - self.step_start_time
            if self.is_recording:
                self.step_times.append(step_time)

        self.current_step += 1
        self.step_start_time = None

    def get_summary(self) -> Dict:
        """
        Get summary statistics of memory usage and timing.

        Returns:
            Dictionary with summary statistics
        """
        if not self.forward_memory:
            return {
                "status": "no_data",
                "message": "No memory data recorded (warmup not completed)"
            }

        summary = {
            "warmup_steps": self.warmup_steps,
            "recorded_steps": len(self.forward_memory),
            "memory_mb": {
                "forward_pass": {
                    "mean": sum(self.forward_memory) / len(self.forward_memory),
                    "max": max(self.forward_memory),
                    "min": min(self.forward_memory),
                },
                "backward_pass": {
                    "mean": sum(self.backward_memory) / len(self.backward_memory),
                    "max": max(self.backward_memory),
                    "min": min(self.backward_memory),
                },
                "peak_per_step": {
                    "mean": sum(self.peak_memory) / len(self.peak_memory),
                    "max": max(self.peak_memory),
                    "min": min(self.peak_memory),
                },
            },
            "timing_seconds": {
                "mean_step_time": sum(self.step_times) / len(self.step_times),
                "total_time": sum(self.step_times),
                "steps_per_second": len(self.step_times) / sum(self.step_times) if self.step_times else 0,
            },
        }

        return summary

    def save_summary(self, filepath: str):
        """Save summary statistics to JSON file."""
        summary = self.get_summary()
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)

    def reset(self):
        """Reset all recorded statistics."""
        self.current_step = 0
        self.snapshots.clear()
        self.forward_memory.clear()
        self.backward_memory.clear()
        self.peak_memory.clear()
        self.step_times.clear()
        self.is_recording = False

    def print_summary(self):
        """Print memory and timing summary to console."""
        summary = self.get_summary()

        if summary.get("status") == "no_data":
            print(f"\n{summary['message']}")
            return

        print("\n" + "=" * 80)
        print("MEMORY PROFILING SUMMARY")
        print("=" * 80)
        print(f"Warmup steps: {summary['warmup_steps']}")
        print(f"Recorded steps: {summary['recorded_steps']}")
        print()

        mem = summary['memory_mb']
        print("Memory Usage (MB):")
        print(f"  Forward Pass:")
        print(f"    Mean:  {mem['forward_pass']['mean']:.2f} MB")
        print(f"    Max:   {mem['forward_pass']['max']:.2f} MB")
        print(f"  Backward Pass:")
        print(f"    Mean:  {mem['backward_pass']['mean']:.2f} MB")
        print(f"    Max:   {mem['backward_pass']['max']:.2f} MB")
        print(f"  Peak per Step:")
        print(f"    Mean:  {mem['peak_per_step']['mean']:.2f} MB")
        print(f"    Max:   {mem['peak_per_step']['max']:.2f} MB")
        print()

        timing = summary['timing_seconds']
        print("Timing:")
        print(f"  Mean step time: {timing['mean_step_time']:.4f} s")
        print(f"  Total time:     {timing['total_time']:.2f} s")
        print(f"  Steps/second:   {timing['steps_per_second']:.2f}")
        print("=" * 80)


def get_gpu_memory_info() -> Dict:
    """
    Get current GPU memory information.

    Returns:
        Dictionary with memory statistics in MB
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    device = torch.cuda.current_device()

    return {
        "device": torch.cuda.get_device_name(device),
        "allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
        "max_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "total_memory_mb": torch.cuda.get_device_properties(device).total_memory / 1024**2,
    }


def print_gpu_memory_info():
    """Print current GPU memory information."""
    info = get_gpu_memory_info()

    if "error" in info:
        print(f"Error: {info['error']}")
        return

    print(f"\nGPU: {info['device']}")
    print(f"  Total memory:     {info['total_memory_mb']:.2f} MB")
    print(f"  Allocated:        {info['allocated_mb']:.2f} MB")
    print(f"  Reserved:         {info['reserved_mb']:.2f} MB")
    print(f"  Peak allocated:   {info['max_allocated_mb']:.2f} MB")
