"""
Day 68 - Hugging Face Hub
Module 2: Download a Small Local LLM (1-3B), Run CPU Inference, Measure Speed

Task: download a small open model from the Hugging Face Hub, run inference on
the CPU, and measure how fast it is.

Metrics reported per run:
    - TTFT         time to first token (how long the prompt takes to process)
    - prefill t/s  prompt tokens processed per second
    - decode t/s   steady-state generation speed (the number people usually quote)
    - overall t/s  new tokens divided by total wall time
Plus load time, weight memory, and process memory.

Generation is greedy (deterministic) so runs are comparable. One warm-up run
is executed first and excluded from the statistics, because the first call is
slower (lazy initialization, cold caches).

Usage:
    python 02_local_llm_cpu_benchmark.py
    python 02_local_llm_cpu_benchmark.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
    python 02_local_llm_cpu_benchmark.py --dtype bfloat16 --threads 4 --runs 5
    python 02_local_llm_cpu_benchmark.py --output results.json

Check each model's card on the Hub for its license and usage terms before use.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

from bench_utils import (
    compute_generation_metrics,
    estimate_weights_gib,
    format_param_count,
    get_available_ram_gib,
    get_rss_mb,
    summarize,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_PROMPT = "Explain in three sentences why the sky is blue."


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def build_step_timer():
    """
    Create a LogitsProcessor that records a timestamp for every generated token.

    generate() calls the processor once per new token, so the recorded
    timestamps give per-token timing from a single generation run.
    """
    from transformers import LogitsProcessor

    class StepTimer(LogitsProcessor):
        def __init__(self) -> None:
            self.timestamps: list[float] = []

        def __call__(self, input_ids, scores):
            self.timestamps.append(time.perf_counter())
            return scores  # leave the scores untouched

    return StepTimer()


def count_parameters_on_meta_device(model_name: str) -> int | None:
    """Count parameters from the config alone (no weights downloaded)."""
    try:
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained(model_name)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)
        return sum(parameter.numel() for parameter in model.parameters())
    except Exception:
        return None


def print_system_info(torch, transformers) -> None:
    section("System")
    print(f"Platform     : {platform.platform()}")
    print(f"Python       : {platform.python_version()}")
    print(f"torch        : {torch.__version__}")
    print(f"transformers : {transformers.__version__}")
    print(f"CPU cores    : {os.cpu_count()}  (torch threads: {torch.get_num_threads()})")
    available = get_available_ram_gib()
    print(f"Available RAM: {available:.1f} GiB" if available is not None
          else "Available RAM: unknown (install psutil)")


def build_inputs(tokenizer, prompt: str):
    """Use the model's chat template when it has one, otherwise the raw prompt."""
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # The template already inserted any special tokens, so don't add them again.
        return tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return tokenizer(prompt, return_tensors="pt")


def run_once(model, tokenizer, inputs, max_new_tokens: int) -> tuple[dict, str]:
    import torch
    from transformers import LogitsProcessorList

    timer = build_step_timer()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    n_prompt = inputs["input_ids"].shape[1]

    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            logits_processor=LogitsProcessorList([timer]),
            pad_token_id=pad_id,
        )
    end = time.perf_counter()

    metrics = compute_generation_metrics(start, timer.timestamps, end, n_prompt)
    text = tokenizer.decode(output_ids[0][n_prompt:], skip_special_tokens=True)
    return metrics, text


def fmt(value: float | None, spec: str = ".2f") -> str:
    return format(value, spec) if value is not None else "n/a"


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark a small local LLM on CPU.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Hub model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--runs", type=int, default=3, help="Measured runs (default: 3)")
    parser.add_argument("--warmup", type=int, default=1, help="Warm-up runs, excluded from stats")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16", "auto"], default="float32",
                        help="Weight precision (default: float32, the safest on CPU)")
    parser.add_argument("--threads", type=int, default=None, help="Number of CPU threads for torch")
    parser.add_argument("--device-map", default=None,
                        help='Optional device_map passed to from_pretrained (e.g. "auto"); needs accelerate')
    parser.add_argument("--output", default=None, help="Write full results to this JSON file")
    args = parser.parse_args()

    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        print(f"Missing package: {error}. Run: pip install -r requirements.txt")
        return 1

    if args.threads:
        torch.set_num_threads(args.threads)

    print_system_info(torch, transformers)

    # ---- Memory sanity check before downloading weights ---- #
    section(f"Model: {args.model}")
    n_params = count_parameters_on_meta_device(args.model)
    if n_params is not None and args.dtype != "auto":
        needed = estimate_weights_gib(n_params, args.dtype)
        print(f"Parameters: {format_param_count(n_params)}; estimated weights in {args.dtype}: {needed:.2f} GiB")
        available = get_available_ram_gib()
        if available is not None and needed > 0.9 * available:
            print(f"WARNING: weights alone need {needed:.1f} GiB but only {available:.1f} GiB of RAM "
                  "is free. Try a smaller model or --dtype bfloat16, or expect swapping/failure.")

    # ---- Load ---- #
    dtype = args.dtype if args.dtype == "auto" else getattr(torch, args.dtype)
    load_kwargs = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
    if args.device_map:
        load_kwargs["device_map"] = args.device_map

    print("Downloading (first run only) and loading...")
    try:
        start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
        load_seconds = time.perf_counter() - start
    except Exception as error:
        print(f"Could not load '{args.model}': {type(error).__name__}: {error}")
        print("Check the model ID, your internet connection, and available memory.")
        return 1

    model.eval()
    footprint_mib = model.get_memory_footprint() / (1024 ** 2)
    rss = get_rss_mb()
    print(f"Load time      : {load_seconds:.1f} s (includes download on the first run)")
    print(f"Weights in RAM : {footprint_mib:.0f} MiB")
    print(f"Process RSS    : {fmt(rss, '.0f')} MiB")

    inputs = build_inputs(tokenizer, args.prompt)
    print(f"Prompt tokens  : {inputs['input_ids'].shape[1]}")

    # ---- Warm-up ---- #
    for index in range(args.warmup):
        print(f"Warm-up run {index + 1}/{args.warmup} (excluded from stats)...")
        run_once(model, tokenizer, inputs, max_new_tokens=min(16, args.max_new_tokens))

    # ---- Measured runs ---- #
    section("Measured Runs")
    header = f"{'run':>3} {'new tok':>8} {'TTFT (s)':>9} {'prefill t/s':>12} {'decode t/s':>11} {'overall t/s':>12}"
    print(header)
    print("-" * len(header))

    runs: list[dict] = []
    first_text = ""
    for index in range(1, args.runs + 1):
        metrics, text = run_once(model, tokenizer, inputs, args.max_new_tokens)
        runs.append(metrics)
        if index == 1:
            first_text = text
        print(f"{index:>3} {metrics['new_tokens']:>8} {fmt(metrics['ttft_s'], '.3f'):>9} "
              f"{fmt(metrics['prefill_tps'], '.1f'):>12} {fmt(metrics['decode_tps'], '.2f'):>11} "
              f"{fmt(metrics['overall_tps'], '.2f'):>12}")

    # ---- Summary ---- #
    section("Summary (mean +/- std over measured runs)")
    summary: dict[str, dict] = {}
    for key, label in (("ttft_s", "TTFT (s)"), ("prefill_tps", "Prefill tokens/s"),
                       ("decode_tps", "Decode tokens/s"), ("overall_tps", "Overall tokens/s")):
        values = [run[key] for run in runs if run[key] is not None]
        if values:
            stats = summarize(values)
            summary[key] = stats
            print(f"{label:<18}: {stats['mean']:.3f} +/- {stats['std']:.3f}  "
                  f"(min {stats['min']:.3f}, max {stats['max']:.3f})")

    print(f"\nSample output:\n{first_text.strip()}")

    if args.output:
        payload = {
            "model": args.model,
            "dtype": args.dtype,
            "threads": torch.get_num_threads(),
            "max_new_tokens": args.max_new_tokens,
            "prompt": args.prompt,
            "load_seconds": load_seconds,
            "weights_mib": footprint_mib,
            "process_rss_mib": rss,
            "runs": runs,
            "summary": summary,
            "sample_output": first_text,
        }
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
