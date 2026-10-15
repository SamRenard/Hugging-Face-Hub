# Day 67 — Hugging Face Hub: Tokenizers in Depth

Part of the *NIZAM AI — 150-Day AI Engineering Protocol*, Month 3
("LLM & RAG Systems"), Block Day 2 of 5.

## Overview

This module explains how text becomes model input. It implements BPE and
WordPiece from scratch so the algorithms can be inspected, then uses real
tokenizers from the Hugging Face Hub to compare GPT-2 (byte-level BPE) with
BERT (WordPiece) on English and Azerbaijani text, and finally examines padding,
truncation, and attention masks.

## Learning Objectives

By the end of this module you should be able to:

- Explain how BPE learns merge rules and encodes unseen words
- Explain WordPiece's greedy longest-match-first encoding and how its training
  criterion differs from BPE's
- Describe why byte-level tokenizers never produce unknown tokens while
  WordPiece can
- Measure tokenizer efficiency with tokens-per-word (fertility) and
  characters-per-token
- Explain why tokenizer efficiency affects API cost and usable context length
  for a given language
- Read `input_ids`, `attention_mask`, and `token_type_ids`, and apply the
  attention mask correctly when pooling
- Handle the GPT-2 missing-pad-token case and choose left vs. right padding

## File Structure

```
day67_hf_hub/
├── 01_tokenization_algorithms.py  # Pure-Python BPE, WordPiece, padding and masks (offline)
├── 02_tokenizer_comparison.py     # GPT-2 vs BERT vs mBERT, English vs Azerbaijani metrics
├── 03_attention_mask_padding.py   # Padding, truncation, attention masks, masked pooling
├── requirements.txt               # Python dependencies
├── .gitignore                     # Ignores caches
└── README.md                      # This file
```

## Requirements

- Python 3.9+
- `transformers` and `numpy` (PyTorch is not required)
- Internet access on the first run of scripts 02 and 03, to download small
  tokenizer files from the Hugging Face Hub (script 01 runs fully offline)

```bash
pip install -r requirements.txt
```

## Usage

```bash
python 01_tokenization_algorithms.py
python 02_tokenizer_comparison.py
python 02_tokenizer_comparison.py --models gpt2 bert-base-uncased
python 02_tokenizer_comparison.py --text "Any sentence you want to inspect."
python 03_attention_mask_padding.py
```

## About the Azerbaijani Test

The task requires an Azerbaijani language test, so `02_tokenizer_comparison.py`
contains three parallel English/Azerbaijani sentence pairs as test data. This
is the only non-English content in the code; all identifiers, comments, and
output text are English. The script reports metrics computed live rather than
asserting expected results, so outcomes depend on the tokenizer versions you
download. Add your own pairs to `TEXT_PAIRS` to extend the test.

## Key Concepts Reference

| Concept            | Idea                                                                 |
|---------------------|-----------------------------------------------------------------------|
| BPE                  | Repeatedly merge the most frequent adjacent symbol pair                |
| WordPiece            | Greedy longest-match against a vocabulary; `##` marks continuations    |
| Byte-level BPE       | Works on bytes, so every input is representable (no unknown token)     |
| Fertility            | Tokens per word; higher means less efficient for that language         |
| Attention mask       | 1 for real tokens, 0 for padding; tells the model what to ignore       |
| Padding side         | Left padding is typical for batched decoder-only generation            |
| Masked pooling       | Average only over real tokens, never over padding                      |

## Suggested Workflow (matches the daily plan)

1. **Deep theory (2h):** Work through the tokenizer chapter of the Hugging Face
   Course (free), running script 01 alongside it and tracing a merge by hand.
2. **Practical coding (2h):** Run script 02, then add Azerbaijani sentences of
   your own and compare models; run script 03 to see padding and masks.
3. **Consolidation:** Write a 5–10 sentence summary in your own words plus
   the key terms below (second-brain habit).
4. **Commit:** Push this folder to GitHub, wait 5 minutes, re-read your own
   code, and improve one thing before finalizing.

## Key Terms

`tokenizer`, `BPE`, `WordPiece`, `sub-word`, `vocabulary`, `[UNK]`,
`fertility`, `attention mask`, `padding`, `truncation`, `token_type_ids`

## Reference Sources

- Hugging Face Course (free)
- Hugging Face Hub

## Progress

Day 67 / 150 — Overall completion: 45%
