# miniGPT-from-scratch

A from-scratch implementation of a small Decoder-only Transformer language model in PyTorch.

## Current Version

`v1.0` completes the intended learning project: a Decoder-only Transformer with character and byte-level BPE tokenizers, training, resumable checkpoints, generation, controlled experiments, and reproducible documentation.

It includes:

- character-level tokenizer
- dataset and batch construction
- token and learned positional embeddings
- single-head causal self-attention
- multi-head causal self-attention
- pre-norm Transformer blocks
- residual connections
- position-wise feed-forward networks
- stacked Transformer blocks
- final LayerNorm
- language-modeling head
- next-token cross-entropy loss
- modular sanity checks for data, attention, blocks, and model forward pass
- YAML-driven AdamW training loop
- periodic train and validation loss estimation
- best-validation checkpoint saving and loading
- autoregressive generation with temperature and top-k sampling
- run-isolated checkpoints, histories, plots, and generation samples
- controlled temperature and top-k comparison experiments
- model/optimizer restoration with continued global step numbering
- inherited loss histories and CPU/CUDA RNG state saving for resumable runs
- configurable ReLU/GELU feed-forward activations stored in model checkpoints
- backward-compatible loading of legacy ReLU checkpoints
- optional token-embedding/language-head weight tying with shared storage checks
- backward-compatible loading of legacy untied checkpoints
- character-level and byte-level BPE tokenizer implementations
- BPE compression, mixed-language round-trip, save, and load checks

The final model is trained on Tiny Shakespeare with a learned 512-token byte-level BPE vocabulary.

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

## Why this project matters

This project is designed to demonstrate a bottom-up understanding of Decoder-only Transformer language models, rather than simply using existing high-level frameworks.

Meaningful training and generation runs are summarized in [experiments/EXPERIMENT_LOG.md](experiments/EXPERIMENT_LOG.md).

The final design, results, limitations, and completion scope are summarized in [report/technical_report.md](report/technical_report.md).

## Experiment Artifacts

Set `experiment.run_name` in `configs/tiny.yaml` before training. Checkpoints and loss artifacts are isolated by that name:

```text
experiments/checkpoints/<run_name>.pt
experiments/loss_curves/<run_name>.json
experiments/loss_curves/<run_name>.png
```

`scripts/generate.py` and `experiments/loss_curves.py` use the active run name from the config. A different recorded curve can also be rendered with `python experiments/loss_curves.py --run-name <run_name>`.
