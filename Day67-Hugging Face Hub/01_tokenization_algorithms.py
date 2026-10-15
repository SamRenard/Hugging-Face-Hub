"""
Day 67 - Hugging Face Hub
Module 1: Tokenizers in Depth - BPE, WordPiece, and Attention Masks

Pure-Python, dependency-free implementations that make the algorithms behind
modern tokenizers easy to inspect:

    - BPE (Byte-Pair Encoding): learn merges from corpus statistics, then
      encode new words by replaying those merges in the order they were learned
    - WordPiece: greedy longest-match-first encoding against a fixed vocabulary
      (the inference algorithm used by BERT), plus the training-score
      difference between BPE and WordPiece
    - Padding and attention masks: how variable-length sequences are batched

Run:
    python 01_tokenization_algorithms.py
"""

from __future__ import annotations

from collections import Counter

END_OF_WORD = "</w>"


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# --------------------------------------------------------------------------- #
# BPE
# --------------------------------------------------------------------------- #

WordFreqs = dict[tuple[str, ...], int]


def build_word_frequencies(corpus: str) -> WordFreqs:
    """Split a corpus into words and represent each as a tuple of characters."""
    counts = Counter(corpus.split())
    return {tuple(word) + (END_OF_WORD,): freq for word, freq in counts.items()}


def count_pairs(word_freqs: WordFreqs) -> Counter:
    """Count how often each adjacent symbol pair occurs, weighted by word frequency."""
    pairs: Counter = Counter()
    for symbols, freq in word_freqs.items():
        for left, right in zip(symbols, symbols[1:]):
            pairs[(left, right)] += freq
    return pairs


def merge_pair(pair: tuple[str, str], word_freqs: WordFreqs) -> WordFreqs:
    """Replace every occurrence of `pair` with a single merged symbol."""
    left, right = pair
    merged: WordFreqs = {}

    for symbols, freq in word_freqs.items():
        new_symbols: list[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == left and symbols[i + 1] == right:
                new_symbols.append(left + right)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        key = tuple(new_symbols)
        merged[key] = merged.get(key, 0) + freq

    return merged


def train_bpe(corpus: str, num_merges: int) -> list[tuple[str, str]]:
    """Learn an ordered list of merge rules (most frequent pair first)."""
    word_freqs = build_word_frequencies(corpus)
    merges: list[tuple[str, str]] = []

    for step in range(1, num_merges + 1):
        pairs = count_pairs(word_freqs)
        if not pairs:
            break
        # Ties are broken lexicographically so results are deterministic.
        best_pair, best_count = max(pairs.items(), key=lambda item: (item[1], item[0]))
        word_freqs = merge_pair(best_pair, word_freqs)
        merges.append(best_pair)
        print(f"  merge {step:>2}: {best_pair[0]!r} + {best_pair[1]!r}  (count={best_count})")

    return merges


def encode_bpe(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """Encode a word by applying learned merges in the order they were learned."""
    ranks = {pair: rank for rank, pair in enumerate(merges)}
    symbols = list(word) + [END_OF_WORD]

    while len(symbols) > 1:
        candidates = [
            (ranks[pair], index)
            for index, pair in enumerate(zip(symbols, symbols[1:]))
            if pair in ranks
        ]
        if not candidates:
            break
        _, index = min(candidates)  # earliest-learned merge wins
        symbols[index:index + 2] = [symbols[index] + symbols[index + 1]]

    return symbols


def bpe_demo() -> None:
    section("1. BPE: Learning Merges From a Corpus")

    corpus = (
        "low low low low low "
        "lower lower "
        "newest newest newest newest newest newest "
        "widest widest widest"
    )
    print("Corpus word counts:", dict(Counter(corpus.split())))
    print("\nLearned merges:")
    merges = train_bpe(corpus, num_merges=10)

    print("\nEncoding words with the learned merges:")
    for word in ["lowest", "newer", "wider", "low"]:
        print(f"  {word:>8} -> {encode_bpe(word, merges)}")

    print(
        "\nNote: 'lowest' never appeared in the corpus, yet it is encoded from\n"
        "known sub-word pieces. This is how BPE handles unseen words."
    )


# --------------------------------------------------------------------------- #
# WordPiece
# --------------------------------------------------------------------------- #

def wordpiece_tokenize(
    word: str,
    vocab: set[str],
    unk_token: str = "[UNK]",
    prefix: str = "##",
    max_chars: int = 100,
) -> list[str]:
    """
    Greedy longest-match-first WordPiece encoding (BERT-style inference).

    At each position, take the longest vocabulary entry that matches. Pieces
    that do not start the word carry the continuation prefix. If any position
    cannot be matched, the whole word becomes the unknown token.
    """
    if len(word) > max_chars:
        return [unk_token]

    tokens: list[str] = []
    start = 0
    while start < len(word):
        end = len(word)
        match = None
        while start < end:
            piece = word[start:end]
            if start > 0:
                piece = prefix + piece
            if piece in vocab:
                match = piece
                break
            end -= 1

        if match is None:
            return [unk_token]

        tokens.append(match)
        start = end

    return tokens


def wordpiece_demo() -> None:
    section("2. WordPiece: Greedy Longest-Match-First Encoding")

    vocab = {
        "[UNK]", "un", "##believ", "##able", "believ", "##ing",
        "play", "##ing", "##s", "token", "##iz", "##er", "##s",
    }
    print("Toy vocabulary:", sorted(vocab))
    print()
    for word in ["unbelievable", "playing", "tokenizers", "xyz", "unbelievablex"]:
        print(f"  {word:>14} -> {wordpiece_tokenize(word, vocab)}")

    print(
        "\nNote: a word with any unmatched piece becomes a single [UNK] token\n"
        "(see 'xyz'). BPE-style byte-level tokenizers such as GPT-2's avoid\n"
        "this because every byte is always representable."
    )


def training_score_comparison() -> None:
    section("3. Training Difference: BPE Count vs. WordPiece Score")

    corpus = "hug hug hug pug pug pun pun pun pun bun bun bun hugs hugs"
    word_freqs = build_word_frequencies(corpus)

    pair_counts = count_pairs(word_freqs)
    symbol_counts: Counter = Counter()
    for symbols, freq in word_freqs.items():
        for symbol in symbols:
            symbol_counts[symbol] += freq

    # BPE picks the most frequent pair; WordPiece normalizes by how common the
    # two parts are on their own: score = count(pair) / (count(a) * count(b)).
    bpe_choice = max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]
    scores = {
        pair: count / (symbol_counts[pair[0]] * symbol_counts[pair[1]])
        for pair, count in pair_counts.items()
    }
    wordpiece_choice = max(scores.items(), key=lambda item: (item[1], item[0]))[0]

    print(f"BPE first merge       (highest raw count): {bpe_choice}")
    print(f"WordPiece first merge (highest score)    : {wordpiece_choice}")
    print(
        "\nWordPiece favors pairs whose parts rarely appear apart from each\n"
        "other, rather than simply the most frequent pair."
    )


# --------------------------------------------------------------------------- #
# Padding and attention masks
# --------------------------------------------------------------------------- #

def pad_batch(
    sequences: list[list[int]],
    pad_id: int = 0,
    padding_side: str = "right",
) -> tuple[list[list[int]], list[list[int]]]:
    """Pad sequences to equal length and build the matching attention mask."""
    if padding_side not in {"right", "left"}:
        raise ValueError("padding_side must be 'right' or 'left'")

    max_len = max(len(seq) for seq in sequences)
    padded, masks = [], []

    for seq in sequences:
        n_pad = max_len - len(seq)
        pad = [pad_id] * n_pad
        real = [1] * len(seq)
        fake = [0] * n_pad

        if padding_side == "right":
            padded.append(seq + pad)
            masks.append(real + fake)
        else:
            padded.append(pad + seq)
            masks.append(fake + real)

    return padded, masks


def attention_mask_demo() -> None:
    section("4. Padding and Attention Masks")

    batch = [[101, 7592, 2088, 102], [101, 2023, 102], [101, 102]]
    ids, mask = pad_batch(batch, pad_id=0)

    print("input_ids:")
    for row in ids:
        print("  ", row)
    print("attention_mask (1 = real token, 0 = padding):")
    for row in mask:
        print("  ", row)

    left_ids, left_mask = pad_batch(batch, pad_id=0, padding_side="left")
    print("\nLeft padding (preferred for decoder-only generation):")
    for row_ids, row_mask in zip(left_ids, left_mask):
        print("  ", row_ids, row_mask)

    print(
        "\nThe attention mask tells the model to ignore padding positions.\n"
        "Without it, padding tokens would influence attention and pooled outputs."
    )


if __name__ == "__main__":
    bpe_demo()
    wordpiece_demo()
    training_score_comparison()
    attention_mask_demo()
