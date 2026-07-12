# Experiment Log

This document records meaningful miniGPT experiments so results remain comparable after local checkpoints and generated artifacts are overwritten.

## E001 - Toy Corpus Pipeline Baseline

- Date: 2026-07-11
- Purpose: verify the end-to-end training, checkpoint, and generation pipeline.
- Corpus: custom toy English text, 193 characters, vocabulary size 24.
- Split: 170 training tokens, 19 validation tokens.
- Model: block size 16, 2 layers, 4 heads, embedding size 64, dropout 0.1.
- Training: batch size 32, 500 steps, AdamW, learning rate 0.0003, weight decay 0.1, seed 1337.
- Parameters: 103,832.
- Initial loss: train 3.3406, validation 3.2537.
- Best checkpoint: step 100, train loss 2.0817, validation loss 2.9744.
- Final loss: step 500, train 0.3921, validation 4.8702.
- Result: the pipeline worked, but severe overfitting began after roughly step 100 because the corpus and validation split were extremely small.
- Generation: character distribution and fragments of corpus words were learned, but output was largely incoherent.
- Local artifacts: `e001_toy_baseline.pt`, `e001_toy_baseline.json`, and `e001_toy_baseline.png` in their respective ignored experiment directories.

## E002 - Tiny Shakespeare Data Expansion Baseline

- Date: 2026-07-12
- Purpose: isolate the effect of replacing the toy corpus while keeping the model and training settings unchanged.
- Corpus: Tiny Shakespeare, 1,115,394 characters, vocabulary size 65.
- Model: block size 16, 2 layers, 4 heads, embedding size 64, dropout 0.1.
- Training: batch size 32, 500 steps, AdamW, learning rate 0.0003, weight decay 0.1, seed 1337.
- Parameters: 109,121.
- Initial loss: train 4.3218, validation 4.3186.
- Best checkpoint: step 500, train loss 2.5043, validation loss 2.5130.
- Final loss: identical to the best checkpoint because validation loss improved at every evaluation.
- Result: train and validation losses stayed close and decreased throughout the run, with no observed overfitting by step 500. More training is justified.
- Generation sample:

```text
the wint ak brid owind t son, bth

Hise myobe t eranthend my thachanhe ar hthare wearthe.
War dilas ate are than.
```

- Interpretation: output remains incoherent, but word-like fragments, punctuation, capitalization, line breaks, and play-like rhythm improved substantially over E001.
- Local artifacts: `e002_shakespeare_baseline.pt`, `e002_shakespeare_baseline.json`, and `e002_shakespeare_baseline.png` in their respective ignored experiment directories.

## Recording Policy

Record an experiment when at least one meaningful variable or outcome changes, such as:

- corpus or train/validation split;
- model architecture or parameter scale;
- optimizer or training schedule;
- decoding strategy under a controlled comparison;
- best validation loss or generation quality sufficient to change the next decision.

Do not create a new entry for formatting changes, repeated runs with no analytical value, or simple bug-fix checks.
