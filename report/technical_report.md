# miniGPT-from-scratch Technical Report

## Objective

This project builds a small Decoder-only GPT language model from first principles in PyTorch. The implementation exposes tokenization, causal attention, Transformer blocks, optimization, checkpointing, and autoregressive decoding instead of delegating them to a high-level language-model library.

## Final Architecture

- byte-level BPE tokenizer with 512 tokens;
- learned token and positional embeddings;
- context length 64 tokens;
- 4 pre-norm Transformer blocks;
- 4 causal attention heads per block;
- embedding width 128;
- GELU feed-forward activation;
- residual connections and final LayerNorm;
- untied token-embedding and language-head weights;
- 931,584 trainable parameters.

The final data flow is:

```text
UTF-8 text -> byte BPE -> token ids -> embeddings
-> 4 Transformer blocks -> final LayerNorm -> vocabulary logits
-> cross-entropy training / autoregressive sampling
```

## Dataset and Training

- corpus: Tiny Shakespeare, 1,115,394 characters;
- split: 90% train, 10% validation after tokenization;
- optimizer: AdamW;
- learning rate: 0.0003;
- batch size: 32;
- training steps: 6,000;
- dropout: 0.1;
- seed: 1337;
- decoding: temperature 0.8, top-k 20.

Training saves the best validation checkpoint together with model configuration, optimizer state, tokenizer state, step, loss, and CPU/CUDA RNG state. Runs, checkpoints, loss histories, plots, and samples are isolated by `experiment.run_name`.

`configs/tiny.yaml` reproduces the final BPE run configuration, while `configs/char_baseline.yaml` preserves the preferred character baseline. Training and generation accept `--config` so the two pipelines remain independently reproducible.

## Key Experiments

The project progressed from a 193-character pipeline check to Tiny Shakespeare and controlled architecture experiments. Important findings include:

- the toy corpus overfit after roughly 100 steps;
- expanding to Tiny Shakespeare removed the immediate overfitting failure;
- longer training, context 64, width 128, and depth 4 each improved validation quality;
- GELU slightly outperformed ReLU;
- weight tying reduced parameters by 1.02% but worsened validation loss by 6.80%, so it remains optional and disabled;
- temperature and top-k demonstrated the expected coherence/diversity tradeoff.

Full experiment records are in `experiments/EXPERIMENT_LOG.md`.

## Character vs BPE Tokenization

The trained BPE-512 tokenizer reduced the corpus from 1,115,394 character tokens to 568,210 BPE tokens, a 49.06% reduction. Average token payload increased to 1.963 bytes, so a 64-token context covers approximately 125.6 characters instead of 64.

Token-level cross-entropy cannot be compared directly across tokenizers. Using an estimated bits-per-character normalization:

| Model | Vocab | Parameters | Val loss/token | Estimated bits/character |
|---|---:|---:|---:|---:|
| Character E010 | 65 | 816,705 | 1.6994 | 2.4516 |
| BPE E012 | 512 | 931,584 | 3.1803 | 2.3374 |

The BPE model improves estimated bits per character by approximately 4.66% while nearly doubling effective context coverage. Its generated sample also exhibits longer coherent spans, more stable words, and stronger play-like structure.

## Limitations

- This is a character/byte-subword educational model, not a production LLM.
- BPE training uses a clear but non-optimized repeated-scan implementation.
- The model uses learned absolute positions and recomputes the full context during generation.
- The estimated BPC comparison uses corpus-average token compression rather than a dedicated byte-normalized evaluation pass.
- Tiny Shakespeare is small and domain-specific.

## Completion and Extensions

Version 1.0 completes the intended learning scope: tokenizer, data pipeline, Decoder-only Transformer, training, checkpoint resume, generation, controlled experiments, BPE integration, and documentation.

Possible post-v1.0 extensions include KV caching, RoPE, top-p sampling, mixed precision, optimized BPE training, and a small interactive interface.
