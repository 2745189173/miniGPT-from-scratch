# miniGPT-from-scratch

A from-scratch implementation of a small Decoder-only Transformer language model in PyTorch.

## Current Version

`v0.2` implements the complete Decoder-only Transformer architecture.

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

Training and autoregressive text generation are not implemented yet.

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
- [ ] Training loop
- [ ] Loss curve visualization
- [ ] Decoding strategy comparison
- [ ] KV cache benchmark
- [ ] Ablation study

## Why this project matters

This project is designed to demonstrate a bottom-up understanding of Decoder-only Transformer language models, rather than simply using existing high-level frameworks.
