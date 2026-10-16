"""
Day 68 - Hugging Face Hub
Module 1: Model Loading, Inference, device_map, and Memory Management

Covers:
    - Estimating how much memory a model needs for each numeric precision
    - Counting parameters WITHOUT downloading weights (meta device)
    - Loading with different dtypes and comparing the real memory footprint
    - What device_map does and how to inspect the placement
    - A minimal inference pass (eval mode, inference_mode, generate)
    - Releasing memory between experiments

Usage:
    python 01_model_loading_and_memory.py --offline     # estimates only, no downloads
    python 01_model_loading_and_memory.py               # also loads gpt2 (small download)
    python 01_model_loading_and_memory.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import gc
import sys
import time

from bench_utils import (
    BYTES_PER_PARAM,
    estimate_weights_gib,
    format_param_count,
    get_rss_mb,
)


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def free_memory() -> None:
    """Release Python objects and cached GPU memory between experiments."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# 1. Memory estimation (no dependencies, no downloads)
# --------------------------------------------------------------------------- #

def memory_estimate_table() -> None:
    section("1. Weight Memory by Model Size and Precision (GiB)")

    sizes = [("0.5B", 500_000_000), ("1.5B", 1_500_000_000),
             ("3B", 3_000_000_000), ("7B", 7_000_000_000)]
    dtypes = list(BYTES_PER_PARAM)

    print(f"{'model size':<12}" + "".join(f"{dtype:>11}" for dtype in dtypes))
    print("-" * (12 + 11 * len(dtypes)))
    for label, n_params in sizes:
        row = "".join(f"{estimate_weights_gib(n_params, dtype):>11.2f}" for dtype in dtypes)
        print(f"{label:<12}{row}")

    print(
        "\nRule of thumb: weights = parameters x bytes per parameter.\n"
        "These are lower bounds. Inference also needs memory for activations and\n"
        "the KV cache (which grows with context length and batch size), so leave\n"
        "headroom beyond the weight size."
    )


# --------------------------------------------------------------------------- #
# 2. Parameter count without downloading weights
# --------------------------------------------------------------------------- #

def count_parameters_on_meta_device(model_name: str) -> int:
    """
    Build the architecture on the 'meta' device: shapes only, no memory.

    Only the small config file is downloaded, so you can check whether a model
    fits in RAM BEFORE committing to a multi-gigabyte download.
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_name)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    return sum(parameter.numel() for parameter in model.parameters())


def parameter_count_demo(model_name: str) -> None:
    section(f"2. Parameter Count Without Downloading Weights: {model_name}")

    n_params = count_parameters_on_meta_device(model_name)
    print(f"Parameters: {format_param_count(n_params)} ({n_params:,})")
    for dtype in ("float32", "bfloat16", "int8", "int4"):
        print(f"  estimated weights in {dtype:<9}: {estimate_weights_gib(n_params, dtype):.2f} GiB")


# --------------------------------------------------------------------------- #
# 3. Loading with different dtypes
# --------------------------------------------------------------------------- #

def dtype_comparison_demo(model_name: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    section(f"3. Loading in Different Precisions: {model_name}")

    for label, dtype in (("float32", torch.float32), ("bfloat16", torch.bfloat16)):
        start = time.perf_counter()
        # low_cpu_mem_usage avoids holding two copies of the weights while loading.
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, low_cpu_mem_usage=True
        )
        load_seconds = time.perf_counter() - start

        footprint_mib = model.get_memory_footprint() / (1024 ** 2)
        rss = get_rss_mb()
        rss_text = f"{rss:.0f} MiB" if rss is not None else "n/a (install psutil)"
        print(f"{label:<9} load={load_seconds:>6.2f}s  weights={footprint_mib:>8.1f} MiB  "
              f"process RSS={rss_text}")

        del model
        free_memory()

    print(
        "\nbfloat16 halves the weight memory. On CPU, speed depends on hardware\n"
        "support for bfloat16, so always benchmark instead of assuming."
    )


# --------------------------------------------------------------------------- #
# 4. device_map
# --------------------------------------------------------------------------- #

def device_map_demo(model_name: str) -> None:
    from transformers import AutoModelForCausalLM

    section(f"4. device_map Placement: {model_name}")

    print(
        "device_map tells Transformers WHERE to place each part of the model:\n"
        '  "cpu" / "cuda:0"  put everything on one device\n'
        '  "auto"            fill the fastest devices first (GPU 0, GPU 1, ..., then\n'
        "                    CPU, then disk), splitting layers across them\n"
        "  a dict            explicit placement of named modules\n"
        "This is what lets models larger than one GPU's memory run at all.\n"
        "It requires the 'accelerate' package.\n"
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    print("Resulting placement (hf_device_map):")
    for module_name, device in model.hf_device_map.items():
        print(f"  {module_name or '<root>':<30} -> {device}")

    del model
    free_memory()


# --------------------------------------------------------------------------- #
# 5. Minimal inference
# --------------------------------------------------------------------------- #

def inference_demo(model_name: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    section(f"5. Minimal Inference: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()  # disable dropout and other training-only behavior

    inputs = tokenizer("The capital of France is", return_tensors="pt")

    # inference_mode disables gradient tracking: less memory, faster execution.
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,  # greedy decoding for reproducible output
            pad_token_id=tokenizer.eos_token_id,  # some models define no pad token
        )

    print("Output:", tokenizer.decode(output_ids[0], skip_special_tokens=True))

    del model
    free_memory()


def main() -> int:
    parser = argparse.ArgumentParser(description="Model loading and memory management demos.")
    parser.add_argument("--model", default="gpt2", help="Hub model ID (default: gpt2, ~124M params)")
    parser.add_argument("--offline", action="store_true",
                        help="Only run the estimation table (no libraries or downloads needed)")
    args = parser.parse_args()

    memory_estimate_table()
    if args.offline:
        return 0

    steps = [parameter_count_demo, dtype_comparison_demo, device_map_demo, inference_demo]
    for step in steps:
        try:
            step(args.model)
        except ImportError as error:
            print(f"\nSkipping {step.__name__}: missing package ({error}). "
                  "Run: pip install -r requirements.txt")
        except Exception as error:
            print(f"\n{step.__name__} failed: {type(error).__name__}: {error}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
