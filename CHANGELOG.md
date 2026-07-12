# Changelog

All notable project milestones are summarized here. Detailed experiment evidence
is available in `experiments/EXPERIMENT_LOG.md`.

## [Unreleased]

## [1.0.1] - 2026-07-12

- Added MIT license and Tiny Shakespeare attribution.
- Added Ruff lint/format configuration and GitHub Actions CI.
- Standardized Python formatting, imports, and modern type annotations.
- Added contributing guidance, changelog, line-ending policy, and public-facing
  README positioning.
- Added reproducible character/BPE config selection via `--config`.
- Removed stale empty placeholders and corrected dependency-file naming/version
  constraints.

## [1.0] - 2026-07-12

- Integrated byte-level BPE-512 into dataset, training, checkpoint, resume, and
  generation workflows.
- Added tracked tokenizer merges and reproducible character/BPE configurations.
- Added normalized character/BPE comparisons and final technical documentation.
- Completed experiments E001-E012.

## [0.9] - 2026-07-12

- Added deterministic educational byte-level BPE training, encode/decode,
  persistence, and round-trip checks.

## [0.8] - 2026-07-12

- Added optional token-embedding/language-head weight tying.
- Preserved legacy untied checkpoint compatibility and documented the negative
  E011 ablation result.

## [0.7] - 2026-07-12

- Added configurable ReLU/GELU FeedForward activation with legacy checkpoint
  compatibility.

## [0.6] - 2026-07-12

- Added resumable model/optimizer training, inherited loss history, global step,
  and CPU/CUDA RNG state.

## [0.5] - 2026-07-12

- Added controlled temperature and top-k decoding experiments.

## [0.4] - 2026-07-12

- Added run-isolated experiment artifacts, loss history, and curve rendering.

## [0.3] - 2026-07-12

- Added AdamW training, best-validation checkpoints, and autoregressive text
  generation.

## [0.2] - 2026-07-11

- Completed the Decoder-only Transformer architecture with causal multi-head
  attention, pre-norm blocks, residual connections, and FeedForward layers.

## [0.1] - 2026-07-11

- Recorded the minimal character-level embedding-to-logits language-model
  baseline.
