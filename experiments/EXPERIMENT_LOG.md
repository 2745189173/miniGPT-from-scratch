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

## E003 - Tiny Shakespeare 3000-Step Training

- Date: 2026-07-12
- Purpose: test whether E002 was undertrained by changing only the training duration from 500 to 3,000 steps.
- Corpus: Tiny Shakespeare, 1,115,394 characters, vocabulary size 65.
- Model: unchanged from E002: block size 16, 2 layers, 4 heads, embedding size 64, dropout 0.1.
- Training: unchanged from E002 except for 3,000 steps; batch size 32, AdamW, learning rate 0.0003, weight decay 0.1, seed 1337.
- Parameters: 109,121.
- Initial loss: train 4.3218, validation 4.3186.
- Best checkpoint: step 2,900, train loss 2.1349, validation loss 2.1292.
- Final loss: step 3,000, train 2.1357, validation 2.1481.
- Comparison with E002: best validation loss improved from 2.5130 to 2.1292 (15.3% lower); validation perplexity improved from approximately 12.34 to 8.41 (31.9% lower).
- Result: E002 was undertrained. Train and validation curves remained close throughout E003, and no sustained overfitting was observed. The small rise at step 3,000 is consistent with evaluation noise rather than a clear reversal.
- Generation sample:

```text
the wint if bridchwill thall, be madisen bube to tarther the catatands:
Whit he us he hert?

MEXENDHate awick my thand a wize mus
Youns, to fit he cove thelill, at miree seng, will is tharev
```

- Interpretation: character-level coherence improved noticeably. The model now reproduces play-like speaker labels, punctuation, capitalization, questions, line breaks, and longer word-like fragments, but semantics and spelling remain unstable.
- Local artifacts: `e003_shakespeare_3000_steps.pt`, `e003_shakespeare_3000_steps.json`, and `e003_shakespeare_3000_steps.png` in their respective ignored experiment directories.
- Next decision: hold the E003 checkpoint fixed and run controlled inference-time temperature/top-k comparisons before changing architecture or training settings.

## E004 - Temperature Comparison

- Date: 2026-07-12
- Purpose: isolate the effect of sampling temperature on coherence and diversity.
- Fixed conditions: E003 step-2,900 checkpoint, prompt `the`, seed 1337, 300 new tokens, top-k 40.
- Compared temperatures: 0.5, 0.8, 1.1, and 1.4.
- Temperature 0.5: produced the most stable word-like fragments, but collapsed toward high-frequency words and repeated `the` heavily. Diversity and play-like structure were weak.
- Temperature 0.8: produced the best balance in this run, with line breaks, questions, speaker-like labels, and moderate lexical variety while retaining some local structure.
- Temperature 1.1: increased formatting and lexical diversity, but spelling fragmented more often and local coherence weakened.
- Temperature 1.4: produced the highest diversity but noticeably more broken words, abrupt punctuation, rare character combinations, and unstable structure.
- Result: increasing temperature traded local coherence for diversity as expected. Temperature 0.8 is selected as the controlled value for the next top-k comparison; this is a run-specific engineering choice, not a universal optimum.
- Representative contrast:

```text
temperature 0.5: ... in the the the me not the wat thell ...
temperature 0.8: MEXENDHate awick my thand a wize mus ...
temperature 1.1: DUKENWIO: Welll no me litenser cechiry ...
temperature 1.4: Yatauss: Vanthat usqor, vet? ...
```

- Local artifact: `experiments/generation_samples/e004_temperature_comparison.json`.
- Next decision: hold temperature at 0.8 and compare top-k values with every other generation condition fixed.

## E005 - Top-k Comparison

- Date: 2026-07-12
- Purpose: isolate the effect of top-k filtering after selecting temperature 0.8 in E004.
- Fixed conditions: E003 step-2,900 checkpoint, prompt `the`, seed 1337, 300 new tokens, temperature 0.8.
- Compared top-k values: 1, 5, 10, 20, 40, and no filtering.
- Top-k 1: collapsed into a deterministic `the the the ...` loop. Greedy next-token choice could not escape the locally most likely continuation.
- Top-k 5: produced recognizable word-like fragments but remained conservative and repetitive, heavily favoring common words such as `the` and `and`.
- Top-k 10: improved variety while retaining relatively stable local word shapes, but still showed repeated sentence patterns.
- Top-k 20: provided the preferred balance for the next model experiment, with more varied continuations than 5/10 and fewer highly fragmented speaker-like artifacts than 40.
- Top-k 40: increased structural and lexical variety, including questions and speaker-like labels, but also introduced more unstable invented words.
- No top-k filtering: produced exactly the same sampled text as top-k 40 for this seed. This does not make the settings equivalent; it indicates the excluded tail probability did not alter this particular sampling path.
- Result: overly restrictive top-k causes mode collapse and repetition. Increasing the candidate set improves diversity but admits less stable continuations. Top-k 20 is selected as a controlled engineering choice for the next experiment.
- Representative contrast:

```text
top-k 1: the the the the the ...
top-k 5: Thow and the a be madisen and shan and and ...
top-k 20: This that you whave reand meall and bar bapte ...
top-k 40: MEXENDHate awick my thand a wize mus ...
```

- Local artifact: `experiments/generation_samples/e005_top_k_comparison.json`.
- Next decision: increase context length before model width/depth, because block size 16 restricts the model to approximately 16 characters of usable context.

## E006 - Context Length 64

- Date: 2026-07-12
- Purpose: isolate the effect of increasing context length from 16 to 64 characters.
- Corpus: Tiny Shakespeare, 1,115,394 characters, vocabulary size 65.
- Model: block size 64, 2 layers, 4 heads, embedding size 64, dropout 0.1. Width and depth are unchanged from E003.
- Training: 3,000 steps, batch size 32, AdamW, learning rate 0.0003, weight decay 0.1, seed 1337.
- Parameters: 112,193, an increase of 3,072 learned positional-embedding parameters over E003.
- Best checkpoint: step 3,000, train loss 2.1819, validation loss 2.1918, approximate validation perplexity 8.95.
- Comparison with E003: E003 reached validation loss 2.1292 and perplexity 8.41. At 3,000 steps, E006 validation loss is 2.94% higher.
- Curve interpretation: train and validation losses remain close, and E006 reaches its best validation result at the final evaluation. The model has not shown sustained overfitting or clear convergence, so the 3,000-step comparison is inconclusive rather than evidence that longer context is harmful.
- Generation settings: prompt `the`, seed 1337, temperature 0.8, top-k 20, 300 new tokens.
- Generation sample:

```text
the wint wand incerd,
The lay be mad ser bube to tanthend my dall ands ard thie us hithe.

F CORNDHININGE:
Har he fald to horm he offor tof is he me mil;
```

- Generation interpretation: the sample has stable lines, a speaker-like label, punctuation, and less obvious repetition, but one sample cannot establish a context-length advantage.
- Local artifacts: `e006_shakespeare_block64.pt`, `e006_shakespeare_block64.json`, and `e006_shakespeare_block64.png` in their respective ignored experiment directories.
- Next decision: add checkpoint resume support and continue the E006 training trajectory to 6,000 total steps before judging block size 64.

## Recording Policy

Record an experiment when at least one meaningful variable or outcome changes, such as:

- corpus or train/validation split;
- model architecture or parameter scale;
- optimizer or training schedule;
- decoding strategy under a controlled comparison;
- best validation loss or generation quality sufficient to change the next decision.

Do not create a new entry for formatting changes, repeated runs with no analytical value, or simple bug-fix checks.
