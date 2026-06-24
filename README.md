# miniGPT-from-scratch

A from-scratch implementation of a small Decoder-only Transformer language model in PyTorch.

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

- [x] Project stucture
- [ ] Character-level tokenizer
- [ ] Dataset and batch construction
- [ ] Causal self-attention
- [ ] Multi-head attention
- [ ] Transformer block
- [ ] Loss curve visualization
- [ ] Decoding strategy comparison
- [ ] Decoding strategy comparison
- [ ] KV cache benchmark
- [ ] Ablation study

## Why this project matters

This project is designed to demonstrate a bottom-up understanding of Decoder-only Transformer language model, rather than simply using existing high-level frameworks.