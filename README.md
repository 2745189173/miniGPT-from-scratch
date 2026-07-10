# miniGPT-from-scratch

A from-scratch implementation of a small Decoder-only Transformer language model in PyTorch.

## Current Version

`v0.1` is a minimal character-level language modeling baseline.

It includes:

- character-level tokenizer
- dataset and batch construction
- token embedding
- learned positional embedding
- language-modeling head
- next-token cross-entropy loss
- sanity checks for data and model forward pass

It does not yet include Transformer blocks, causal self-attention, multi-head attention, or text generation.

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
- [ ] Causal self-attention
- [ ] Multi-head attention
- [ ] Transformer block
- [ ] Training loop
- [ ] Loss curve visualization
- [ ] Decoding strategy comparison
- [ ] KV cache benchmark
- [ ] Ablation study

## Why this project matters

This project is designed to demonstrate a bottom-up understanding of Decoder-only Transformer language models, rather than simply using existing high-level frameworks.
