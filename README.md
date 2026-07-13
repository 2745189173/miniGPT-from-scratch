# miniGPT-from-scratch

[English](README.md) | [简体中文](README.zh-CN.md)

A small Decoder-only Transformer language model built from first principles in
PyTorch, including byte-level BPE tokenization, training, resumable checkpoints,
generation, and controlled experiments.

This repository is an educational implementation and experiment platform. It is
designed to make GPT internals inspectable; it is not a production chatbot or a
replacement for a broadly pretrained language model.

## Current Version

`v1.0.1` completes the intended learning scope and public-release polish. The
final model is trained on Tiny
Shakespeare with a learned 512-token byte-level BPE vocabulary.

## Highlights

- deterministic character and byte-level BPE tokenizers implemented from scratch;
- causal single-head and multi-head self-attention;
- pre-norm Transformer blocks with residual connections and GELU FeedForward;
- YAML-driven AdamW training and periodic validation;
- best-checkpoint saving, optimizer restoration, RNG state, and resume support;
- autoregressive generation with temperature and top-k sampling;
- isolated experiment artifacts and twelve documented controlled experiments;
- backward compatibility for legacy tokenizer, activation, and weight-tying states;
- focused checks, Ruff lint/format, and GitHub Actions CI.

## Final Results

| Metric | Character E010 | BPE E012 |
|---|---:|---:|
| Vocabulary size | 65 | 512 |
| Parameters | 816,705 | 931,584 |
| Corpus tokens | 1,115,394 | 568,210 |
| Estimated characters per 64-token context | 64.0 | 125.6 |
| Estimated bits per character | 2.4516 | 2.3374 |

BPE reduced the token count by **49.06%**, nearly doubled effective context
coverage, and improved estimated bits per character by **4.66%**.

Example output from the final BPE checkpoint:

```text
they hands that can trumble with laughter,
Captain is the daughter, hearty of your swork;
Well I have duke me, and you are a maids the devil'd, and these a fear!

KING RICHARD II:
Now, go my lord, and can her give me business to his troge about bod,
```

## Architecture

```text
UTF-8 text -> byte-level BPE -> token ids
-> token + learned positional embeddings
-> 4 x pre-norm Transformer blocks
   -> 4-head causal self-attention
   -> GELU FeedForward
-> final LayerNorm -> LM head -> next-token logits
```

Final configuration: context 64, 4 layers, 4 heads, embedding width 128,
dropout 0.1, and 931,584 parameters.

## Project Goal

This project aims to implement the core mechanisms of GPT-style language models from scratch, including:

- character-level tokenizer
- token embedding
- positional embedding
- causal self-attention
- multi-head attention
- Transformer block
- next-token prediction training
- autoregressive text generation
- decoding strategies
- tokenizer and architecture ablations

## Roadmap

- [x] Project structure
- [x] Character-level tokenizer
- [x] Dataset and batch construction
- [x] Minimal language model baseline
- [x] Causal self-attention
- [x] Multi-head attention
- [x] Transformer block
- [x] Training loop
- [x] Validation loss and checkpoint saving
- [x] Autoregressive text generation
- [x] Temperature and top-k sampling
- [x] Loss curve visualization
- [x] Decoding strategy comparison
- [x] Byte-level BPE tokenizer core
- [x] BPE training and generation integration
- [x] Ablation study

Post-v1.0 ideas include KV caching, RoPE, top-p sampling, mixed precision, and larger-scale training.

## Setup

Create and activate a Python environment, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download Tiny Shakespeare:

```powershell
python scripts\download_data.py
```

Train the deterministic BPE-512 tokenizer configured in `configs/tiny.yaml`:

```powershell
python scripts\train_tokenizer.py
```

The tracked tokenizer artifact is written to:

```text
artifacts/tokenizers/tiny_shakespeare_bpe_512.json
```

Tiny Shakespeare is downloaded from Andrej Karpathy's `char-rnn` repository.
See [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md) for source and usage notes.

## Validation

Run the focused checks:

```powershell
python scripts\check_data.py
python scripts\check_bpe.py
python scripts\check_tokenizer_integration.py
python scripts\check_attention.py
python scripts\check_block.py
python scripts\check_model.py
python scripts\check_gradients.py
```

## Train and Generate

Train the active run from `configs/tiny.yaml`:

```powershell
python scripts\train.py
```

The published character baseline can be reproduced separately:

```powershell
python scripts\train.py --config configs\char_baseline.yaml
```

Render its loss curve and generate text from its best checkpoint:

```powershell
python experiments\loss_curves.py
python scripts\generate.py
```

Use `--config configs/char_baseline.yaml` to generate from the local E010 character checkpoint. The tracked `experiments/final_comparison.json` contains the published comparison; recalculating it requires both local E010 and E012 checkpoints.

Compare tokenizers and final models:

```powershell
python experiments\tokenizer_comparison.py
python experiments\final_comparison.py
```

Checkpoints, histories, plots, and generated samples are local experiment artifacts and are intentionally ignored by git.

## Repository Layout

```text
src/          model, attention, blocks, tokenizers, and generation
scripts/      data download, tokenizer training, model training, and checks
configs/      final BPE and character-baseline configurations
experiments/  comparisons, plots, generation experiments, and experiment log
artifacts/    tracked deterministic BPE merge rules
docs/         Chinese project journey
report/       final technical report
```

## Why this project matters

This project is designed to demonstrate a bottom-up understanding of Decoder-only Transformer language models, rather than simply using existing high-level frameworks.

Meaningful training and generation runs are summarized in [experiments/EXPERIMENT_LOG.md](experiments/EXPERIMENT_LOG.md).

The final design, results, limitations, and completion scope are summarized in [report/technical_report.md](report/technical_report.md).

A Chinese narrative of the complete learning process, decisions, mistakes, and milestones is available in [docs/PROJECT_JOURNEY.md](docs/PROJECT_JOURNEY.md).

## Experiment Artifacts

Set `experiment.run_name` in `configs/tiny.yaml` before training. Checkpoints and loss artifacts are isolated by that name:

```text
experiments/checkpoints/<run_name>.pt
experiments/loss_curves/<run_name>.json
experiments/loss_curves/<run_name>.png
```

`scripts/generate.py` and `experiments/loss_curves.py` use the active run name from the config. A different recorded curve can also be rendered with `python experiments/loss_curves.py --run-name <run_name>`.

## License

The source code is released under the [MIT License](LICENSE). Dataset attribution
is documented separately in [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md).
