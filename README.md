# miniGPT-from-scratch

A from-scratch implementation of a small Decoder-only Transformer language model in PyTorch.

## Current Version

`v0.7` adds checkpoint-compatible configurable feed-forward activations and adopts GELU for the current GPT-style model.

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

The current toy corpus is intentionally tiny, so generated text demonstrates the complete pipeline rather than strong language quality.

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
- KV cache acceleration

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
- [ ] KV cache benchmark
- [ ] Ablation study

## Why this project matters

This project is designed to demonstrate a bottom-up understanding of Decoder-only Transformer language models, rather than simply using existing high-level frameworks.

Meaningful training and generation runs are summarized in [experiments/EXPERIMENT_LOG.md](experiments/EXPERIMENT_LOG.md).

## Experiment Artifacts

Set `experiment.run_name` in `configs/tiny.yaml` before training. Checkpoints and loss artifacts are isolated by that name:

```text
experiments/checkpoints/<run_name>.pt
experiments/loss_curves/<run_name>.json
experiments/loss_curves/<run_name>.png
```

`scripts/generate.py` and `experiments/loss_curves.py` use the active run name from the config. A different recorded curve can also be rendered with `python experiments/loss_curves.py --run-name <run_name>`.
