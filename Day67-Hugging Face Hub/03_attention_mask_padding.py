"""
Day 67 - Hugging Face Hub
Module 3: Padding, Truncation, and Attention Masks with Real Tokenizers

Covers:
    - Batch-encoding texts of different lengths with padding and truncation
    - Reading input_ids, attention_mask, and token_type_ids
    - The GPT-2 gotcha: no pad token by default
    - Left vs. right padding
    - Why the attention mask matters: masked vs. unmasked mean pooling

Usage:
    python 03_attention_mask_padding.py

The first run downloads small tokenizer files from the Hugging Face Hub.
"""

from __future__ import annotations

import sys

import numpy as np

BATCH = [
    "Hello world.",
    "Tokenizers turn text into numbers that a model can process.",
    "Short.",
]


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def masked_mean_pool(embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """
    Mean-pool token embeddings over real tokens only.

    embeddings:     (batch, seq_len, dim)
    attention_mask: (batch, seq_len) with 1 for real tokens and 0 for padding
    """
    mask = attention_mask[..., None].astype(embeddings.dtype)
    summed = (embeddings * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    return summed / counts


def bert_padding_demo() -> dict:
    from transformers import AutoTokenizer

    section("1. BERT: Padding, Truncation, and Attention Mask")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoded = tokenizer(BATCH, padding=True, truncation=True, max_length=16)

    for index, text in enumerate(BATCH):
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][index])
        print(f"\nText {index}: {text!r}")
        print("  tokens        :", tokens)
        print("  input_ids     :", encoded["input_ids"][index])
        print("  attention_mask:", encoded["attention_mask"][index])
        print("  token_type_ids:", encoded["token_type_ids"][index])

    print(
        "\nThe tokenizer added [CLS] and [SEP], padded every sequence to the longest\n"
        "one in the batch (or to max_length if truncated), and built the mask."
    )
    return encoded


def gpt2_padding_demo() -> None:
    from transformers import AutoTokenizer

    section("2. GPT-2: Missing Pad Token and Padding Side")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print("pad_token before:", tokenizer.pad_token)

    # GPT-2 was trained without a pad token, so batching fails until one is set.
    # Reusing the end-of-sequence token is the standard workaround.
    tokenizer.pad_token = tokenizer.eos_token
    print("pad_token after :", tokenizer.pad_token)

    for side in ("right", "left"):
        tokenizer.padding_side = side
        encoded = tokenizer(BATCH, padding=True)
        print(f"\npadding_side={side!r}")
        for ids, mask in zip(encoded["input_ids"], encoded["attention_mask"]):
            print("  ids :", ids)
            print("  mask:", mask)

    print(
        "\nDecoder-only models generate from the end of the sequence, so batched\n"
        "generation normally uses LEFT padding to keep real tokens adjacent to the\n"
        "position where new tokens are appended."
    )


def pooling_demo(encoded: dict) -> None:
    section("3. Why the Attention Mask Matters: Pooling")

    input_ids = np.asarray(encoded["input_ids"])
    attention_mask = np.asarray(encoded["attention_mask"])

    # Stand-in for model output: one random vector per token position.
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(*input_ids.shape, 4)).astype(np.float32)

    naive = embeddings.mean(axis=1)                      # padding included (wrong)
    masked = masked_mean_pool(embeddings, attention_mask)  # padding ignored (correct)

    for index, text in enumerate(BATCH):
        difference = float(np.abs(naive[index] - masked[index]).max())
        n_real = int(attention_mask[index].sum())
        print(f"Text {index}: {n_real:>2} real tokens of {input_ids.shape[1]} "
              f"-> max difference between naive and masked mean: {difference:.3f}")

    print(
        "\nThe shorter the text, the more padding contaminates a naive mean.\n"
        "Always weight pooled representations by the attention mask."
    )


def main() -> int:
    try:
        encoded = bert_padding_demo()
        gpt2_padding_demo()
    except Exception as error:
        print(f"\nFailed: {type(error).__name__}: {error}")
        print("Check your internet connection and that 'transformers' is installed.")
        return 1

    pooling_demo(encoded)
    return 0


if __name__ == "__main__":
    sys.exit(main())
