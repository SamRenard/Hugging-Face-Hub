"""
Day 67 - Hugging Face Hub
Module 2: Tokenizer Comparison - GPT-2 (byte-level BPE) vs. BERT (WordPiece)
          plus an Azerbaijani language test

Task: tokenize the same text with different tokenizers from the Hugging Face
Hub and compare how efficiently each one handles English versus Azerbaijani.

Metrics per (model, text):
    - tokens      : number of tokens produced
    - words       : whitespace-separated words in the text
    - fertility   : tokens per word (lower = more efficient)
    - chars/token : average characters covered by one token
    - unk         : number of unknown-token placeholders

The Azerbaijani sentences below are intentional test data: they contain
characters such as the schwa that many English-centric tokenizers handle poorly.

Usage:
    python 02_tokenizer_comparison.py
    python 02_tokenizer_comparison.py --models gpt2 bert-base-uncased
    python 02_tokenizer_comparison.py --text "Any custom sentence to inspect."

The first run downloads small tokenizer files from the Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

DEFAULT_MODELS = ["gpt2", "bert-base-uncased", "bert-base-multilingual-cased"]

MODEL_LABELS = {
    "gpt2": "GPT-2 (byte-level BPE)",
    "bert-base-uncased": "BERT uncased (WordPiece)",
    "bert-base-multilingual-cased": "mBERT cased (WordPiece)",
}

# Parallel sentences: the same meaning in English and Azerbaijani.
TEXT_PAIRS = [
    (
        "Artificial intelligence plays an important role in the development of modern technologies.",
        "Süni intellekt müasir texnologiyaların inkişafında mühüm rol oynayır.",
    ),
    (
        "Baku is the capital and largest city of Azerbaijan.",
        "Bakı Azərbaycanın paytaxtı və ən böyük şəhəridir.",
    ),
    (
        "I go to work early every morning.",
        "Mən hər gün səhər tezdən işə gedirəm.",
    ),
]


@dataclass
class TokenStats:
    n_tokens: int
    n_words: int
    n_unk: int
    fertility: float
    chars_per_token: float


def analyze_tokens(text: str, tokens: list[str], unk_token: str | None) -> TokenStats:
    """Compute efficiency metrics for one tokenization result."""
    n_tokens = len(tokens)
    n_words = len(text.split())
    n_unk = sum(1 for token in tokens if unk_token is not None and token == unk_token)

    return TokenStats(
        n_tokens=n_tokens,
        n_words=n_words,
        n_unk=n_unk,
        fertility=n_tokens / n_words if n_words else 0.0,
        chars_per_token=len(text) / n_tokens if n_tokens else 0.0,
    )


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer  # imported lazily for clearer errors

    return AutoTokenizer.from_pretrained(model_name)


def print_stats_table(rows: list[tuple[str, str, TokenStats]]) -> None:
    header = f"{'model':<30} {'lang':<5} {'tokens':>7} {'words':>6} {'tok/word':>9} {'chars/tok':>10} {'unk':>4}"
    print(header)
    print("-" * len(header))
    for label, lang, stats in rows:
        print(
            f"{label:<30} {lang:<5} {stats.n_tokens:>7} {stats.n_words:>6} "
            f"{stats.fertility:>9.2f} {stats.chars_per_token:>10.2f} {stats.n_unk:>4}"
        )


def average_fertility(rows: list[tuple[str, str, TokenStats]], label: str, lang: str) -> float:
    values = [stats.fertility for row_label, row_lang, stats in rows
              if row_label == label and row_lang == lang]
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Hugging Face tokenizers.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Hub model IDs whose tokenizers to compare")
    parser.add_argument("--text", default=None,
                        help="Optional custom text to inspect token by token")
    args = parser.parse_args()

    tokenizers = {}
    for model_name in args.models:
        try:
            tokenizers[model_name] = load_tokenizer(model_name)
        except Exception as error:
            print(f"Could not load tokenizer '{model_name}': {type(error).__name__}: {error}")

    if not tokenizers:
        print("No tokenizers could be loaded. Check your internet connection and that "
              "'transformers' is installed (pip install -r requirements.txt).")
        return 1

    # ---- 1. Quantitative comparison across parallel EN / AZ sentences ---- #
    print("\n" + "=" * 60)
    print("1. Efficiency Comparison (English vs. Azerbaijani)")
    print("=" * 60)

    rows: list[tuple[str, str, TokenStats]] = []
    for model_name, tokenizer in tokenizers.items():
        label = MODEL_LABELS.get(model_name, model_name)
        for english, azerbaijani in TEXT_PAIRS:
            for lang, text in (("EN", english), ("AZ", azerbaijani)):
                tokens = tokenizer.tokenize(text)
                rows.append((label, lang, analyze_tokens(text, tokens, tokenizer.unk_token)))

    print_stats_table(rows)

    print("\nAverage tokens per word and Azerbaijani/English ratio:")
    for model_name in tokenizers:
        label = MODEL_LABELS.get(model_name, model_name)
        english = average_fertility(rows, label, "EN")
        azerbaijani = average_fertility(rows, label, "AZ")
        ratio = azerbaijani / english if english else 0.0
        print(f"  {label:<30} EN={english:.2f}  AZ={azerbaijani:.2f}  AZ/EN={ratio:.2f}x")

    print(
        "\nA higher AZ/EN ratio means the same meaning costs more tokens in Azerbaijani:\n"
        "higher API cost and less usable context window for that language."
    )

    # ---- 2. Token-by-token view of one sentence ---- #
    print("\n" + "=" * 60)
    print("2. Token-by-Token View")
    print("=" * 60)

    sample = args.text or TEXT_PAIRS[0][1]
    print(f"Text: {sample}\n")

    for model_name, tokenizer in tokenizers.items():
        label = MODEL_LABELS.get(model_name, model_name)
        tokens = tokenizer.tokenize(sample)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        print(f"{label}  (vocab size: {len(tokenizer)}, tokens: {len(tokens)})")
        print("  tokens:", tokens)
        print("  ids   :", ids)
        print()

    print(
        "How to read the tokens:\n"
        "  - GPT-2 marks a leading space with 'Ġ' and works on bytes, so it never\n"
        "    produces an unknown token, but rare characters split into many byte pieces.\n"
        "  - BERT marks word continuations with '##' and emits [UNK] for pieces\n"
        "    missing from its vocabulary.\n"
        "  - The uncased English BERT also lowercases and strips accents, which\n"
        "    can erase meaningful letters in other languages."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
