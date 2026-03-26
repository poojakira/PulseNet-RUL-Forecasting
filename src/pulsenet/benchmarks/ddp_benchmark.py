# pyre-ignore-all-errors
"""
Multi-process DDP benchmark script for evaluating PulseNet GPU throughput scaling.
Usage: torchrun --nproc_per_node=NUM_GPUS src/pulsenet/benchmarks/ddp_benchmark.py
"""

from __future__ import annotations

import os
import time

import numpy as np  # pyre-ignore
import torch  # pyre-ignore
import torch.distributed as dist  # pyre-ignore

from pulsenet.models.transformer_model import TransformerModel  # pyre-ignore

# pyre-ignore-all-errors


def main():
    if not torch.cuda.is_available():
        print("CUDA not available. Exiting DDP benchmark.")
        return

    # Initialize process group
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"--- Starting DDP Throughput Benchmark (World Size: {world_size}) ---")

    # Generate synthetic telemetry payload
    batch_size = 256
    seq_len = 30
    n_features = 14
    samples = 10000

    if rank == 0:
        print(f"Generating {samples} synthetic sequences for testing...")

    # Pre-windowed shape for transformer: (B, seq_len, features)
    X = np.random.randn(samples, seq_len, n_features).astype(np.float32)

    # Initialize Model natively configures DDP inside when running via torchrun
    model = TransformerModel(batch_size=batch_size, epochs=1)

    if rank == 0:
        print("Warming up...")

    # Train invokes the DDP wrap internally
    t0 = time.perf_counter()
    model.train(X)
    dist.barrier()  # Synchronize before measuring
    elapsed = time.perf_counter() - t0

    samples_processed_global = samples * world_size
    throughput = samples_processed_global / elapsed

    if rank == 0:
        print("\n--- DDP Benchmark Results ---")
        print(f"GPUs Configured      : {world_size}")
        print(f"Per-GPU Batch Size   : {batch_size}")
        print(f"Global Batch Size    : {batch_size * world_size}")
        print(f"Time Elapsed         : {elapsed:.2f} seconds")
        print(f"Total Global Samples : {samples_processed_global}")
        print(f"Effective Throughput : {throughput:.2f} sequences/sec")
        print("-------------------------------------------\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
