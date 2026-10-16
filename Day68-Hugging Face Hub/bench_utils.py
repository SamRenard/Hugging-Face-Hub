"""
Day 68 - Hugging Face Hub
Shared helpers for memory estimation and inference benchmarking.

These functions are deliberately free of heavy dependencies (no torch, no
transformers) so the maths can be tested and reused anywhere.
"""

from __future__ import annotations

import os
import statistics
from typing import Sequence

GIB = 1024 ** 3

# Bytes needed to store one model weight in each numeric format.
BYTES_PER_PARAM = {
    "float32": 4.0,
    "float16": 2.0,
    "bfloat16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}


def estimate_weights_gib(n_params: int, dtype: str) -> float:
    """
    Memory needed just to hold the weights, in GiB.

    This is a lower bound: real inference also needs memory for activations
    and the KV cache, which grows with context length and batch size.
    """
    if dtype not in BYTES_PER_PARAM:
        supported = ", ".join(BYTES_PER_PARAM)
        raise ValueError(f"Unknown dtype {dtype!r}. Supported: {supported}")
    return n_params * BYTES_PER_PARAM[dtype] / GIB


def format_param_count(n_params: int) -> str:
    if n_params >= 1_000_000_000:
        return f"{n_params / 1e9:.2f}B"
    if n_params >= 1_000_000:
        return f"{n_params / 1e6:.1f}M"
    return str(n_params)


def summarize(values: Sequence[float]) -> dict[str, float]:
    """Mean, sample standard deviation, min, max, and count of a list of numbers."""
    if not values:
        raise ValueError("summarize() requires at least one value")
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def compute_generation_metrics(
    start: float,
    step_times: Sequence[float],
    end: float,
    n_prompt_tokens: int,
) -> dict:
    """
    Derive latency and throughput metrics from timestamps.

    start:       time just before generation began
    step_times:  one timestamp per generated token, in order
    end:         time just after generation finished
    n_prompt_tokens: number of tokens in the prompt

    Definitions:
        ttft_s       time to first token (prompt processing / "prefill")
        prefill_tps  prompt tokens processed per second during prefill
        decode_tps   steady-state generation speed after the first token
        overall_tps  new tokens divided by total wall time
    """
    if not step_times:
        raise ValueError("No generation steps were recorded")

    n_new = len(step_times)
    ttft = step_times[0] - start
    total = end - start
    decode_span = step_times[-1] - step_times[0]

    return {
        "prompt_tokens": n_prompt_tokens,
        "new_tokens": n_new,
        "ttft_s": ttft,
        "prefill_tps": n_prompt_tokens / ttft if ttft > 0 else None,
        "decode_tps": (n_new - 1) / decode_span if n_new > 1 and decode_span > 0 else None,
        "total_s": total,
        "overall_tps": n_new / total if total > 0 else None,
    }


def get_rss_mb() -> float | None:
    """Resident memory of the current process in MiB (requires psutil)."""
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def get_available_ram_gib() -> float | None:
    """Currently available system RAM in GiB (requires psutil)."""
    try:
        import psutil
    except ImportError:
        return None
    return psutil.virtual_memory().available / GIB
