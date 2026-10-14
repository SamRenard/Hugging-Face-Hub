# Day 66 - Hugging Face Hub: Five Pipelines

[![CI](https://github.com/<your-username>/<your-repo>/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/<your-repo>/actions)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Transformers](https://img.shields.io/badge/transformers-v4%20%26%20v5-yellow)
![License](https://img.shields.io/badge/license-MIT-green)

Part of the **NIZAM AI - 150 Day AI Engineering Protocol** (Month 3: LLM & RAG Systems).

Run five different NLP tasks from the Hugging Face Hub (sentiment analysis, named entity recognition,
summarization, question answering and text generation) with one command, and save a report with
the outputs and timings.

```bash
python run_pipelines.py            # writes outputs/results.md and outputs/results.json
```

## Important: Transformers v5 removed two of these pipelines

The v5 migration guide removed the `question-answering`, `summarization`, `translation` and
`text2text-generation` pipelines ("for almost all text-to-text or question-answering tasks a modern chat
model and `TextGenerationPipeline` will provide much higher quality output"). Code from older tutorials such as
`pipeline("summarization")` fails on v5 with `KeyError: Unknown task ...`.
See the [migration guide](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md#pipelines).

This project therefore runs the tasks like this, and works on Transformers v4 and v5:

| Key | Task | Model | How it runs |
|-----|------|-------|-------------|
| `sentiment` | Sentiment analysis | [distilbert/distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) | `pipeline("sentiment-analysis")` |
| `ner` | Named entity recognition | [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER) | `pipeline("ner", aggregation_strategy="simple")` |
| `summarization` | Summarization | [sshleifer/distilbart-cnn-12-6](https://huggingface.co/sshleifer/distilbart-cnn-12-6) | `AutoModelForSeq2SeqLM` + `generate()` |
| `qa` | Question answering | [distilbert/distilbert-base-cased-distilled-squad](https://huggingface.co/distilbert/distilbert-base-cased-distilled-squad) | `AutoModelForQuestionAnswering` + span decoding |
| `generation` | Text generation | [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) | `pipeline("text-generation")` |

Other options if you prefer them: `pip install "transformers<5"` brings back the classic
`pipeline("summarization")` and `pipeline("question-answering")`, or use a chat model through the
`text-generation` pipeline as the migration guide suggests.

Models are **pinned by name** on purpose: a pipeline's default model can change between library versions.

## Setup

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The first run downloads the models (about 3 GB in total) into `~/.cache/huggingface/hub`
(`%USERPROFILE%\.cache\huggingface\hub` on Windows); later runs reuse the cache. Set `HF_HOME` to move it.
On Windows, `HF_HUB_DISABLE_SYMLINKS_WARNING=1` silences a harmless symlink warning.

## Usage

```bash
python run_pipelines.py                         # all five tasks
python run_pipelines.py sentiment ner           # a selection
python run_pipelines.py --device cuda           # GPU (cpu is the default)
python run_pipelines.py --model qa=deepset/roberta-base-squad2   # swap a model
python run_pipelines.py --list                  # tasks and default models
```

At the start it prints which `pipeline()` tasks your installed Transformers supports (so you can see the v5
removals yourself). One failing task never stops the others: the failure is reported and the exit code is `1`.
Outputs go to `outputs/results.md` (human readable) and `outputs/results.json`.

## How it works

A pipeline is three steps in one call: **tokenizer** (text to token ids), **model** (ids to logits) and
**post-processing** (logits to a label and score, entities, text, ...).

- **Sentiment / NER / generation** use `transformers.pipeline`. NER uses `aggregation_strategy="simple"` to
  merge word pieces into whole entities. Generation samples with `temperature=0.8`, `top_k=50` and a fixed
  seed, so the text is reproducible.
- **Summarization** (`hf_tasks/direct.py`): tokenize (truncated at 1024 tokens), `generate()` with beam search (4 beams),
  decode.
- **Question answering** (`hf_tasks/qa.py`): the model outputs a "start" and an "end" score for every token.
  Keep only context tokens, apply a softmax to each list, and choose the span `(i, j)` with the highest
  `P_start[i] * P_end[j]` where `j >= i` and the span is at most 30 tokens. Character offsets from the
  tokenizer map the span back to the original text. This is the decoding step the removed pipeline did.

## Project structure

```
.
├── hf_tasks/
│   ├── registry.py     # the five tasks: model, loader, runner, example input
│   ├── direct.py       # summarization and QA with the Auto classes
│   ├── qa.py           # answer-span decoding (pure NumPy)
│   ├── runner.py       # run tasks, time them, isolate failures
│   ├── report.py       # output formatting, JSON and Markdown reports
│   └── cli.py          # command line interface
├── run_pipelines.py
├── tests/
│   ├── test_qa.py
│   ├── test_tasks.py
│   └── test_integration.py     # opt-in: downloads the real models
├── notes/day66_summary.md      # short concept notes (Azerbaijani)
├── .github/workflows/ci.yml
├── requirements.txt
└── README.md
```

## Testing

```bash
python -m pytest -v                                          # offline, no models needed
RUN_HF_INTEGRATION=1 python -m pytest tests/test_integration.py -v   # real models (about 3 GB)
```

On Windows PowerShell set the variable first: `$env:RUN_HF_INTEGRATION="1"`.

| Area | What is verified |
|------|------------------|
| Answer-span decoder (10) | Character offsets, multi-token answers, question/special tokens never chosen, no end-before-start, length limit, uniform logits give `1/36`, shift invariance, probabilities in `(0, 1]`, invalid input |
| Registry (4) | Five tasks in order, namespaced model ids, summarization and QA use the Auto classes, importing the package never imports `transformers` or `torch` |
| Runner (6) | Model/device pass-through, exact timing, load and run failures captured, `KeyboardInterrupt` not swallowed, one failure does not stop the rest, selection and overrides |
| Formatting (5) | Each task's documented output shape, fallback for unexpected shapes, JSON with NumPy values, Markdown success and failure rows, UTF-8 files |
| CLI (5) | Reports written, exit code `1` on failure, task selection / `--model` / `--device`, `--list`, bad arguments |
| Integration (5, opt-in) | Real models: sentiment labels, entities found, summary shorter than input, QA finds the library, generation continues the prompt |

CI runs the offline tests only, so it needs no model download.

## Design decisions

- **Lazy imports.** `transformers` and `torch` load only when a model is used, so everything else is testable
  in milliseconds.
- **One interface for every task.** `TaskSpec` holds the model, how to load it, how to run it and an example,
  so adding a task means adding one entry.
- **Failures are data.** A task that cannot download or run produces a result with the error, not a crash.
- **Pinned models, explicit versions.** Reproducible results and a clear note about what each API version supports.

## Limitations

- QA truncates the context at 384 tokens (no sliding window), and summarization truncates at 1024 tokens.
- All models are English-only and small; GPT-2 text quality is limited and the output is sampled.
- One example per task; no batching or benchmarking.
- `distilbart-cnn-12-6` ships `.bin` weights only, which is why `torch>=2.6` is required.

## Roadmap

- [ ] QA with a sliding window for long contexts
- [ ] Summarization and translation through a chat model (the v5 recommendation) for comparison
- [ ] Zero-shot classification and fill-mask pipelines
- [ ] A small Gradio demo that lets you choose the task

## References

- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course) (chapter 1)
- [Hugging Face Hub documentation](https://huggingface.co/docs/hub)
- [Transformers pipeline tutorial](https://huggingface.co/docs/transformers/pipeline_tutorial)
- [Transformers v5 migration guide](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md)

## License

Released under the [MIT License](LICENSE).
