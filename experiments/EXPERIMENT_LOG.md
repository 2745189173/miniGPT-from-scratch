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

## E007 - Resumed Context-64 Training to 6000 Steps

- Date: 2026-07-12
- Purpose: determine whether E006 was undertrained and validate model/optimizer checkpoint resume support.
- Resume source: E006 step 3,000 checkpoint, including model and AdamW optimizer states. E006 did not contain RNG states, so this first continuation cannot reconstruct an unavailable bitwise-identical E006 random stream.
- Corpus and model: unchanged from E006: Tiny Shakespeare, block size 64, 2 layers, 4 heads, embedding size 64, dropout 0.1, 112,193 parameters.
- Continued training: step 3,000 to total step 6,000, batch size 32, learning rate 0.0003, weight decay 0.1.
- History integrity: 61 unique evaluation records cover steps 0 through 6,000 at intervals of 100, with exactly one step-3,000 record.
- Best checkpoint: step 5,600, train loss 1.9819, validation loss 2.0095, approximate validation perplexity 7.46.
- Final loss: step 6,000, train 1.9449, validation 2.0227.
- Comparison with E006: validation loss improved from 2.1918 to 2.0095 (8.3% lower).
- Comparison with E003: validation loss improved from 2.1292 to 2.0095 (5.6% lower), and perplexity improved from approximately 8.41 to 7.46.
- Result: E006 was undertrained. With sufficient optimization, block size 64 surpassed the block-16 baseline under this setup. Validation loss flattened near step 5,600 while train loss continued decreasing, indicating diminishing returns but no severe overfitting.
- Generation settings: prompt `the`, seed 1337, temperature 0.8, top-k 20, 300 new tokens.
- Generation sample:

```text
the with affer
Thow and is by be madient bube to tanthend my dagay uss;
Whith fould a he that dill ware awch, my thenstake hour he ofform'd fition.

VORWINIO:
What miree sen cintlaivent drove the the me nown is warmes like dience.
```

- Generation interpretation: longer sentence spans, speaker-like labels, punctuation, contractions, and dialogue formatting are more stable, although semantic coherence and spelling remain limited.
- Local artifacts: `e007_shakespeare_block64_6000.pt`, `e007_shakespeare_block64_6000.json`, and `e007_shakespeare_block64_6000.png` in their respective ignored experiment directories.
- Engineering result: resumed checkpoints now preserve model state, optimizer state, global step, inherited loss history, and CPU/CUDA RNG states for future continuations.
- Next decision: keep block size 64 and compare model capacity, beginning with increased embedding width while holding depth and training conditions controlled.

## E008 - Embedding Width 128

- Date: 2026-07-12
- Purpose: isolate model-width scaling by increasing `n_embd` from 64 to 128 while retaining the established context length and depth.
- Corpus: Tiny Shakespeare, 1,115,394 characters, vocabulary size 65.
- Model: block size 64, 2 layers, 4 heads, embedding size 128, dropout 0.1.
- Training: from scratch for 6,000 steps, batch size 32, AdamW, learning rate 0.0003, weight decay 0.1, seed 1337.
- Parameters: 420,929, approximately 3.75 times E007's 112,193 parameters.
- Initial loss: train 4.3924, validation 4.3960.
- Best checkpoint: step 5,900, train loss 1.6549, validation loss 1.8122, approximate validation perplexity 6.12.
- Final loss: step 6,000, train 1.6543, validation 1.8125.
- Comparison with E007: validation loss improved from 2.0095 to 1.8122 (9.8% lower), while approximate perplexity improved from 7.46 to 6.12 (17.9% lower).
- Curve interpretation: the wider model develops a moderate train/validation gap, but validation loss remains near its best at the end of training and does not show sustained deterioration.
- Generation settings: prompt `the`, seed 1337, temperature 0.8, top-k 20, 300 new tokens.
- Generation sample:

```text
the will affery come,
The lady the dispon of our take and my dalied
My art that us hath be are that ane are the hearts
a with our to the to find come milending,
We mire, sends night staid overs, and the now
```

- Generation interpretation: compared with E007, common words, multi-line syntax, punctuation, and sentence-like continuity are substantially more stable, although grammar and semantics remain imperfect.
- Local artifacts: `e008_shakespeare_width128.pt`, `e008_shakespeare_width128.json`, and `e008_shakespeare_width128.png` in their respective ignored experiment directories.
- Next decision: retain block size 64 and width 128, then increase only depth from 2 to 4 layers to isolate depth scaling.

## E009 - Transformer Depth 4

- Date: 2026-07-12
- Purpose: isolate depth scaling by increasing `n_layer` from 2 to 4 while keeping width, context, heads, and training settings fixed.
- Corpus: Tiny Shakespeare, 1,115,394 characters, vocabulary size 65.
- Model: block size 64, 4 layers, 4 heads, embedding size 128, dropout 0.1.
- Training: from scratch for 6,000 steps, batch size 32, AdamW, learning rate 0.0003, weight decay 0.1, seed 1337.
- Parameters: 816,705, approximately 1.94 times E008's 420,929 parameters.
- Best checkpoint: step 6,000, train loss 1.5468, validation loss 1.7097, approximate validation perplexity 5.53.
- Comparison with E008: validation loss improved from 1.8122 to 1.7097 (5.7% lower), while approximate perplexity improved from 6.12 to 5.53 (9.7% lower).
- Curve interpretation: train and validation losses continue trending downward through the final evaluation. A moderate generalization gap remains, but there is no sustained validation deterioration.
- Generation settings: prompt `the`, seed 1337, temperature 0.8, top-k 20, 300 new tokens.
- Generation sample:

```text
the will affend comford, and, but
be sence be to take and my call and barnith;
Nurse the trace that ane are the confallack, on my
to them of it heart my for like enour entent,
And them in overs, and the now of you so. Ahave you hearns;
```

- Generation interpretation: longer grammatical spans, punctuation, line continuity, and sentence-like structure improve over E008, though invented words and semantic instability remain.
- Local artifacts: `e009_shakespeare_depth4.pt`, `e009_shakespeare_depth4.json`, and `e009_shakespeare_depth4.png` in their respective ignored experiment directories.
- Next decision: stop pure capacity scaling temporarily and run an activation-function ablation, replacing ReLU with GELU to more closely match GPT-style feed-forward networks.

## E010 - GELU Activation Ablation

- Date: 2026-07-12
- Purpose: compare GPT-style GELU against the existing ReLU FeedForward activation while holding all other conditions fixed.
- Corpus: Tiny Shakespeare, 1,115,394 characters, vocabulary size 65.
- Model: unchanged from E009 except for GELU activation: block size 64, 4 layers, 4 heads, embedding size 128, dropout 0.1.
- Training: from scratch for 6,000 steps, batch size 32, AdamW, learning rate 0.0003, weight decay 0.1, seed 1337.
- Parameters: 816,705, identical to E009 because the activation function has no learned parameters.
- Best checkpoint: step 6,000, train loss 1.5307, validation loss 1.6994, approximate validation perplexity 5.47.
- Comparison with E009: validation loss improved from 1.7097 to 1.6994 (0.60% lower), while approximate perplexity improved from 5.53 to 5.47 (1.03% lower).
- Curve interpretation: GELU produces a small but consistent improvement under fixed conditions. Validation loss reaches its best at the final evaluation without sustained deterioration.
- Generation settings: prompt `the`, seed 1337, temperature 0.8, top-k 20, 300 new tokens.
- Generation sample:

```text
the wind if thy cowing to life
the dispoker expeecanting to me?
I'll such him fould word. What that a fraw you had
fall to my must of the cause of me mine dill,
We misters not not time in overs, and the news!
```

- Generation interpretation: questions, contractions, punctuation, speaker-like labels, and sentence continuity remain strong; invented words and semantic errors persist.
- Engineering result: activation is now stored in `GPTConfig` and checkpoints. Legacy checkpoints without the field default to ReLU, while E010 explicitly records GELU. E009/ReLU and E010/GELU loading were both verified.
- Local artifacts: `e010_shakespeare_gelu.pt`, `e010_shakespeare_gelu.json`, and `e010_shakespeare_gelu.png` in their respective ignored experiment directories.
- Next decision: retain GELU and evaluate another standard GPT technique, token-embedding/language-head weight tying, as a controlled parameter-sharing ablation.

## E011 - Token Embedding and LM-Head Weight Tying

- Date: 2026-07-12
- Purpose: evaluate GPT-style parameter sharing between `token_embedding.weight` and `lm_head.weight` while holding the E010 architecture and training conditions fixed.
- Corpus: Tiny Shakespeare, 1,115,394 characters, vocabulary size 65.
- Model: block size 64, 4 layers, 4 heads, embedding size 128, GELU, dropout 0.1, with token/output weights tied.
- Training: from scratch for 6,000 steps, batch size 32, AdamW, learning rate 0.0003, weight decay 0.1, seed 1337.
- Parameters: 808,385 versus E010's 816,705, a reduction of 8,320 parameters (1.02%).
- Best checkpoint: step 6,000, train loss 1.6748, validation loss 1.8149, approximate validation perplexity 6.14.
- Comparison with E010: validation loss worsened from 1.6994 to 1.8149 (6.80% higher), while approximate perplexity worsened from 5.47 to 6.14 (12.25% higher).
- Curve interpretation: validation loss decreases through the final step, so the negative result is not caused by late overfitting. E011 shows an early optimization plateau and remains behind E010 throughout training.
- Initialization finding: directly tying PyTorch's default Embedding matrix produced step-0 validation loss around 82.7 because its scale was unsuitable as an output projection. Copying the original lm-head initialization into the shared matrix restored a healthy step-0 loss around 4.35.
- Generation sample:

```text
The lay be madise a way do take reath the changs
You that she rece. When that a full crown.

HENRY VI:
Would off hearfuit heart my well
Which misters, and that stay to by preather preason
```

- Result: weight tying works technically and saves parameters, but it harms quality under the current character-level setup and initialization recipe. Keep the feature available but disabled in the preferred baseline.
- Local artifacts: `e011_shakespeare_weight_tying.pt`, `e011_shakespeare_weight_tying.json`, and `e011_shakespeare_weight_tying.png` in their respective ignored experiment directories.
- Next decision: restore E010 (GELU, untied weights) as the preferred character-level baseline and begin the BPE tokenizer phase.

## E012 - Byte-level BPE-512 Model

- Date: 2026-07-12
- Purpose: complete the tokenizer phase by integrating a learned byte-level BPE vocabulary into dataset construction, checkpointing, training, restoration, and generation.
- Corpus: Tiny Shakespeare, 1,115,394 UTF-8 bytes/characters.
- Tokenizer: byte-level BPE with 256 base bytes and 256 learned merges, vocabulary size 512.
- Compression: 568,210 BPE tokens versus 1,115,394 character tokens, a 49.06% token-count reduction and 1.963 bytes per BPE token.
- Context coverage: block size 64 covers an estimated 125.6 characters with BPE versus 64 characters with character tokenization, a 1.96x increase.
- Model: block size 64, 4 layers, 4 heads, embedding size 128, GELU, untied weights, dropout 0.1.
- Parameters: 931,584. The increase over E010 comes from the larger 512-token input/output vocabulary.
- Training: from scratch for 6,000 steps, batch size 32, AdamW, learning rate 0.0003, weight decay 0.1, seed 1337.
- Best checkpoint: step 6,000, train loss 2.7847, validation loss 3.1803.
- Cross-tokenizer evaluation: token-level losses are not directly comparable. Estimated bits per character improved from E010's 2.4516 to E012's 2.3374, approximately 4.66% better.
- Generation settings: prompt `the`, seed 1337, temperature 0.8, top-k 20, 300 new BPE tokens.
- Generation sample:

```text
they hands that can trumble with laughter,
Captain is the daughter, hearty of your swork;
Well I have duke me, and you are a maids the devil'd, and these a fear!
I am as fearth: if Warwick, thou'st thou liege, and the heart of
put scart and thank you happs and gods with him laught.

KING RICHARD II:
Now, go my lord, and can her give me business to his troge about bod,
```

- Result: BPE nearly halves sequence length, improves effective context coverage and estimated bits per character, and produces visibly stronger word-level and dialogue structure. The complete BPE checkpoint independently restores its tokenizer merges and model.
- Local artifacts: `e012_shakespeare_bpe512.pt`, `e012_shakespeare_bpe512.json`, and `e012_shakespeare_bpe512.png` in their respective ignored experiment directories.
- Tracked tokenizer artifact: `artifacts/tokenizers/tiny_shakespeare_bpe_512.json`.
- Next decision: finalize documentation and release v1.0. Further architecture and inference improvements are optional extensions.

## Recording Policy

Record an experiment when at least one meaningful variable or outcome changes, such as:

- corpus or train/validation split;
- model architecture or parameter scale;
- optimizer or training schedule;
- decoding strategy under a controlled comparison;
- best validation loss or generation quality sufficient to change the next decision.

Do not create a new entry for formatting changes, repeated runs with no analytical value, or simple bug-fix checks.
