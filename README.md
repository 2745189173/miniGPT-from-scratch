# miniGPT-from-scratch

A from-scratch implementation of a small Decoder-only Transformer language model in PyTorch.

## Current Version

`v0.3` adds end-to-end training, best-checkpoint selection, and autoregressive text generation.

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
- [ ] Loss curve visualization
- [ ] Decoding strategy comparison
- [ ] KV cache benchmark
- [ ] Ablation study

## Why this project matters

This project is designed to demonstrate a bottom-up understanding of Decoder-only Transformer language models, rather than simply using existing high-level frameworks.
