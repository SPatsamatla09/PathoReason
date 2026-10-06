# Competence search: final result (negative)

Dates: 2026-10-03 to 2026-10-05. Protocol: `PLAN.md`, written before any call, with its
step-3 and step-4 addenda and amendments, each committed before the data it governs.

## Outcome

- **Steps 1 to 4 all fail.** No configuration passed dev, so no VLM configuration was run on
  the 977-tile test set. Kiran items 1 and 5 are **not achieved**.
- By the brief this is a stop point: the negative result is written up here and the search
  stops. The faithfulness pipeline was **not** rerun on a new model.
- **Step 6 (sourced ResNet-18) is done:** 75.9% ± 0.4% test accuracy, AUC 0.848.
- **No image control was run,** because no configuration passed dev.
- **Spend:** API $0.32 of the $1.00 cap (steps 1–2 only). Steps 3–4 ran on this Mac and on
  the user's private Kaggle account: 15.4 of the week's 30 GPU hours by Kaggle's quota
  readout on 2026-10-06.

## Every configuration tried

The bars, unchanged: screen accuracy ≥ 65% to advance; dev accuracy ≥ 72% **and** balanced
accuracy ≥ 65%. Unparseable answers count as wrong. Every row uses the pipeline's gridded
tile. **Controls:** none was run for any row, because none passed dev. The 14 earlier model
× prompt cells of the pilot and the prompt diagnostic (4 + 10) ran before this search on the
same 100 test-partition tiles, not on screen or dev. They are in `runs/PILOT_RESULTS.json` and
`runs/prompt_diagnostic/RESULTS.md`; the best reached 61 of 100, below the 63 always-HP.

| step | model (ID, revision, runtime) | prompt (hash) | decoding | set | n | acc | acc 95% CI | bal acc | recall HP | recall SSA | controls | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | gemma-4-31b-it ᵃ | neutral labels + criteria (51293f8e / b4ba5883) | T 1.0 | screen | 100 | 0.470 | 0.375–0.567 | 0.563 | 0.21 | 0.92 | not run | stops |
| 1 | Qwen3-VL-235B ᵇ | ″ | T 1.0 | screen | 100 | 0.370 | 0.282–0.468 | 0.500 | 0.00 | 1.00 | not run | stops |
| 2 | gemma-4-31b-it ᵃ | real names + criteria + 6 examples (e0ef4bbe) | T 1.0 | screen | 100 | 0.600 | 0.502–0.691 | 0.599 | 0.60 | 0.59 | not run | stops |
| 2 | gemma-4-31b-it ᵃ | neutral labels + criteria + 6 examples (51293f8e / b4ba5883) | T 1.0 | screen | 100 | 0.650 | 0.553–0.736 | 0.694 | 0.52 | 0.86 | — | advances |
| 2 | ″ | ″ | T 1.0 | **dev** | **300** | **0.573** | 0.517–0.628 | **0.626** | 0.42 | 0.83 | not run | **fail** |
| 2 | Qwen3-VL-235B ᵇ | real names + criteria + 6 examples (e0ef4bbe) | T 1.0 | screen | 100 | 0.400 | 0.309–0.498 | 0.524 | 0.05 | 1.00 | not run | stops |
| 2 | Qwen3-VL-235B ᵇ | neutral labels + criteria + 6 examples (51293f8e / b4ba5883) | T 1.0 | screen | 100 | 0.400 | 0.309–0.498 | 0.524 | 0.05 | 1.00 | not run | stops |
| 3 | MedGemma 1.5 4B ᶜ | classify only, `co_p1` (80c1ce12) | tier 1 | screen | 100 | 0.620 | 0.522–0.709 | 0.520 | 0.90 | 0.14 | not run | stops |
| 3 | ″ | ″ | tier 2 | screen | 100 | 0.630 | 0.532–0.718 | 0.500 | 1.00 | 0.00 | not run | stops |
| 3 | ″ | classify then explain, `cte_p1` (eb0e4d85) | tier 1 | screen | 100 | 0.570 | 0.472–0.663 | 0.452 | 0.90 | 0.00 | not run | stops |
| 3 | ″ | ″ | tier 2 | screen | 100 | 0.490 | 0.394–0.587 | 0.389 | 0.78 | 0.00 | not run | stops |
| 3 | ″ | neutral labels + criteria, `neutral_cte` (51293f8e / b4ba5883) | tier 1 | screen | 100 | 0.490 | 0.394–0.587 | 0.495 | 0.48 | 0.51 | not run | stops |
| 3 | ″ | ″ | tier 2 | screen | 100 | 0.490 | 0.394–0.587 | 0.495 | 0.48 | 0.51 | not run | stops |
| 3 | ″ | neutral labels + criteria + 6 examples, `neutral_crit_fs` (51293f8e / b4ba5883) | tier 1 | screen | 100 | 0.450 | 0.356–0.548 | 0.441 | 0.48 | 0.41 | not run | stops |
| 3 | ″ | ″ | tier 2 | screen | 100 | 0.490 | 0.394–0.587 | 0.495 | 0.48 | 0.51 | not run | stops |
| 4 | MedGemma 1.5 4B + LoRA ᵈ | classify then explain, `cte_p1` (eb0e4d85) | tier 1 | label-free format check, 20 training-pool tiles | 20 | — | — | — | — | — | not run | **fails the check (12 of 20 count, 18 needed); dev not run** |

- ᵃ `google/gemma-4-31b-it` via OpenRouter, pinned to Friendli. ᵇ `qwen/qwen3-vl-235b-a22b-instruct`
  via OpenRouter, pinned to Alibaba. Neither provider exposes a model revision. Per call, the
  records hold the requested model ID and the serving provider; no served model ID was recorded.
- ᶜ `google/medgemma-1.5-4b-it`, Hugging Face revision `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`,
  converted to GGUF Q8_0 (sha256 `2b24548a…c026d605`) with the f16 vision projector
  (`5cfd0926…383e8443`), run locally by llama.cpp (commit `7f2dd88b`, Metal, with the macOS 13
  patch in `llama_cpp_macos13_metal.patch`).
- ᵈ The same base revision, language-model linear layers in 4-bit nf4 with float32 arithmetic
  (the T4 has no bf16; recorded as a forced change), plus a LoRA adapter (sha256
  `13b0dbb0a807…`), run with transformers on a private Kaggle T4.
- Prompt texts are in `prompts/rendered_competence/` (real names + criteria e0ef4bbe; neutral,
  A carries HP 51293f8e; neutral, A carries SSA b4ba5883) and in `prompts/rendered/` (`co_p1`
  80c1ce12; `cte_p1` eb0e4d85). A hash is the first 8 hex digits of the SHA-256 of the prompt
  text; every step-1 to step-3 record carries it as `prompt_sha256`. The step-4 records carry a
  hash of the tokenised prompt instead; the text hash eb0e4d85 is in its `train_summary.json`. The 6 examples are
  unanimous train-pool tiles: MHIST_dpv, ctz and aul (HP); eqa, dqq and bxb (SSA).
- Decoding: "T 1.0" is temperature 1.0 at the API defaults. Tier 1 is temperature 1.0, top_k 64,
  top_p 0.95; tier 2 is temperature 0, run because every tier-1 configuration failed. Thinking
  token banned, at most 1,500 new tokens (PLAN step-3 addendum and Amendment 1).

## Diagnostics, without gating

**Steps 1–2.**

| model | prompt | set | parse | SSA-class calls | cost |
|---|---|---|---|---|---|
| gemma | neutral labels + criteria | screen | 1.00 | 0.84 | $0.019 |
| Qwen | neutral labels + criteria | screen | 1.00 | 1.00 | $0.036 |
| gemma | real names + criteria + examples | screen | 1.00 | 0.47 | $0.042 |
| gemma | neutral labels + criteria + examples | screen | 1.00 | 0.62 | $0.042 |
| gemma | ″ | dev | 1.00 | 0.67 | $0.126 |
| Qwen | real names + criteria + examples | screen | 1.00 | 0.97 | $0.048 |
| Qwen | neutral labels + criteria + examples | screen | 1.00 | 0.97 | $0.047 |

**Step 3 (MedGemma, local).** As the step-3 addendum requires: thinking-trace rate, answers cut
off at the token limit, and answers citing at least 1 and at least 3 valid grid cells.

| prompt | decoding | parse | SSA-class calls | cut off | thinking trace | cites ≥ 1 cell | cites ≥ 3 cells |
|---|---|---|---|---|---|---|---|
| `co_p1` | tier 1 | 1.00 | 0.11 | 4 | 0.00 | — | — |
| `co_p1` | tier 2 | 1.00 | 0.00 | 0 | 0.00 | — | — |
| `cte_p1` | tier 1 | 0.92 | 0.00 | 10 | 0.02 | 0.92 | 0.91 |
| `cte_p1` | tier 2 | 0.75 | 0.00 | 24 | 0.00 | 0.75 | 0.75 |
| `neutral_cte` | tier 1 | 0.99 | 0.53 | 4 | 0.05 | 0.99 | 0.98 |
| `neutral_cte` | tier 2 | 1.00 | 0.52 | 0 | 0.00 | 1.00 | 1.00 |
| `neutral_crit_fs` | tier 1 | 0.97 | 0.48 | 6 | 0.02 | 0.97 | 0.97 |
| `neutral_crit_fs` | tier 2 | 1.00 | 0.52 | 0 | 0.00 | 1.00 | 1.00 |

The 9 thinking traces (all tier 1: 2 `cte_p1`, 5 `neutral_cte`, 2 `neutral_crit_fs`) contain
only the closing thinking token, which is not banned; none contains the banned opening token.
Each opens with a complete JSON answer, then (in 6 of them) some explanation, then the closing
token, then a second JSON answer. The scorer reads the text after that token, as the step-3
addendum fixes. In all 9 the label before the token equals the label after it, so this choice
changes no score. `co_p1` asks for no grid cells.

**Counterbalance halves (descriptive, PLAN "Counterbalancing of neutral labels").** Screen
halves have 48 tiles where Class A carries the HP criteria and 52 where it carries the SSA
criteria; the dev halves have 151 and 149.

| model | prompt | decoding | set | acc, A = HP | acc, A = SSA | bal acc, A = HP | bal acc, A = SSA |
|---|---|---|---|---|---|---|---|
| gemma | neutral labels + criteria | T 1.0 | screen | 0.542 | 0.404 | 0.600 | 0.530 |
| Qwen | neutral labels + criteria | T 1.0 | screen | 0.375 | 0.365 | 0.500 | 0.500 |
| gemma | neutral + examples | T 1.0 | screen | 0.604 | 0.692 | 0.650 | 0.735 |
| gemma | neutral + examples | T 1.0 | dev | 0.563 | 0.584 | 0.618 | 0.634 |
| Qwen | neutral + examples | T 1.0 | screen | 0.417 | 0.385 | 0.533 | 0.515 |
| MedGemma | neutral labels + criteria | tier 1 | screen | 0.646 | 0.346 | 0.528 | 0.474 |
| MedGemma | neutral labels + criteria | tier 2 | screen | 0.625 | 0.365 | 0.500 | 0.500 |
| MedGemma | neutral + examples | tier 1 | screen | 0.542 | 0.365 | 0.433 | 0.455 |
| MedGemma | neutral + examples | tier 2 | screen | 0.625 | 0.365 | 0.500 | 0.500 |

## Step 4 in detail: LoRA fine-tune of MedGemma 1.5 4B, private Kaggle T4 (cost $0)

| | |
|---|---|
| adapter | LoRA r 16, alpha 16, dropout 0.05 on the 238 language-model linear layers; vision tower frozen |
| data | `fewshot_pool` (train minus dev): 1,594 training tiles, 281 held out for epoch selection (seed 20261004). No dev or test tile |
| target | label only: `{\n  "label": "HP|SSA` (8 tokens), class-weighted loss |
| schedule | 4 epochs, 800 optimiser steps, lr 2e-4 cosine; two Kaggle sessions: session 1 paused by itself when its 11.0 h budget ran out, at step 678 of 800 (78 steps into epoch 4, epochs 1–3 validated); session 2 resumed from that checkpoint and ran the remaining 122 steps (about 3.1 h; 14.1 h in total); epoch 4 chosen by validation balanced accuracy |
| label-free format check | 20 training-pool tiles, `cte_p1`, tier 1, sdpa attention, batch 4: **12 of 20 count, 18 needed → fails** |
| consequence | dev evaluation **not run** (PLAN step-4 Amendments 1 A and 2 A); one step-4 configuration only, so step 4 ends without a pass |

**Format check detail** (`STEP4_RESULT.json`). An answer counts when it ends normally and the
pipeline's parser reads a valid label and at least one valid grid cell from it. 14 answers
ended normally and 6 reached the 1,500-token limit; 12 count. Mean length 706 new tokens.
How the 8 others fail:

- in 4 cut-off answers the evidence list keeps growing (28 to 46 entries) until the limit;
- 1 writes the answer, then prose, then copies of the JSON block (12 blocks in all) until the
  limit;
- 1 writes valid JSON and then prose until the limit (valid label and cells, but no normal
  stop);
- 2 end normally and begin with a complete, valid JSON answer, but it is followed by a stray
  closing code fence and prose. The pipeline's parser reads the fenced text first, which is the
  prose, and finds no label.

Even a lenient count (any complete JSON object with a valid label and cell, anywhere in an
answer that ended normally) gives 14 of 20, still below 18. The outcome does not depend on
the parser.

Exploratory: two of the runaway answers describe "periapical bone loss" and "teeth", which
have nothing to do with colon tissue.

**Exploratory, not a dev or test result.** On the 281 held-out training-pool tiles used to
choose the epoch, the fine-tuned model's label (read from the HP / SSA logits) scored:

| epoch | accuracy | balanced accuracy | recall HP | recall SSA |
|---|---|---|---|---|
| 0 (untrained) | 0.722 | 0.500 | 1.00 | 0.00 |
| 1 | 0.790 | 0.768 | 0.82 | 0.72 |
| 2 | 0.819 | 0.701 | 0.97 | 0.44 |
| 3 | 0.819 | 0.732 | 0.93 | 0.54 |
| **4 (chosen)** | **0.833** | **0.801** | 0.87 | 0.73 |

These tiles may share slides with dev and test (MHIST has no slide ids), the numbers chose the
epoch, and the label is read from logits, not from a generated answer. They cannot be read as
competence.

## What steps 1–4 show (all exploratory: none of these comparisons was pre-registered)

- **The class name does not drive the SSA preference.** Qwen calls SSA with the disease names
  (100 of 100 pilot tiles under `cte_p1`; 97 of 100 with names + criteria + examples) and picks
  the SSA-criteria letter on 100 of 100 screen tiles with neutral labels + criteria and on 97 of
  100 with neutral labels + criteria + examples. gemma has no SSA default with the
  names on the gridded tile (SSA on 32 of 100 pilot tiles under `cte_p1`, 47 of 100 with names +
  criteria + examples); replacing the names with Class A/B plus criteria raises its choice of the
  SSA-criteria letter to 84%.
- **MedGemma leans the other way.** With the names it calls SSA on 0–11% of tiles. With neutral
  labels it answers "A" almost always: on 98–100 of 100 tiles in three runs, and on 89 of 100
  (87 of them parseable) with examples at tier 1. The class behind "A" is fixed per tile by the
  counterbalancing, so those rows only reflect that assignment; three of them have the same
  accuracy, balanced accuracy and recalls to four decimal places, and their predictions agree on
  98 to 100 of 100 tiles.
- **Examples helped gemma on the screen but did not hold.** On the same 100 screen tiles, neutral
  labels + criteria went from 47 to 65 correct with 6 examples. On the other 200 dev tiles that
  configuration scored 107 of 200 (53.5%), 57.3% on all 300.
- **The fine-tuned adapter fails the format check, but the fine-tune cannot be named as the
  cause.** The untrained model never generated answers in the 4-bit Kaggle runtime: it ran there
  only to have its HP / SSA logits scored on the 281 validation tiles (the epoch-0 row above), so
  there is no untrained format check to compare with. A 4-step smoke adapter
  in that runtime also failed (16 of 20). And the unmodified Q8_0 model already meets the same
  counting rule on only 89 of 100 (tier 1) and 75 of 100 (tier 2) dev-screen answers under
  `cte_p1`, in its local Q8_0 runtime: run-on answers predate the fine-tune.
- No configuration of this search (15: 6 + 8 + 1) passed dev; only one (gemma, neutral labels
  + criteria + examples) passed the screen. The 14 earlier pilot and diagnostic cells had no
  screen or dev run; the best reached 61 of 100 on their 100 test-partition tiles.
- **No claim of diagnostic validity is made**, for any model here.

**Interruptions.** The Mac restarted on 2026-10-04 at 22:31 and on 2026-10-06 around 14:00.
The runners resume from their records: every step-3 file holds 100 unique tiles and no error
record. The per-run console logs of three tier-1 runs that were resumed on 2026-10-05 were
overwritten on resume (the driver wrote them with `>`; now `>>`); the data files are intact.

## Step 6: ResNet-18 baseline (done, local)

**Design:**

- ResNet-18 in plain PyTorch, trained **from scratch**, with no downloads
- train on train minus dev (1,875 tiles)
- choose the epoch by dev AUC, over 30 epochs
- 3 seeds, with **one test evaluation per seed**

Results: `runs/resnet18/RESULTS.json`.

| seed | chosen epoch | test acc | test AUC | balanced acc | recall HP | recall SSA |
|---|---|---|---|---|---|---|
| 0 | 28 | 76.0% | 0.848 | 0.754 | 0.78 | 0.73 |
| 1 | 24 | 75.4% | 0.849 | 0.757 | 0.75 | 0.77 |
| 2 | 27 | 76.2% | 0.846 | 0.754 | 0.78 | 0.73 |
| **mean ± SD** | | **75.9 ± 0.4%** | **0.848 ± 0.002** | **0.755** | 0.77 | 0.74 |

**Seed 0 checkpoint incident.** iCloud "Optimize Mac Storage" renamed seed 0's
checkpoint (`best.pt` became `best 9.pt`), so the script's own test step crashed.

- The surviving file's stored epoch (28) equals the log's best-dev-AUC epoch. Its single
  test evaluation was run from that file and documented in `seed0/test.json`.
- Seeds 1 and 2 ran normally.
- Checkpoints now live outside iCloud.

**This replaces the unsourced ResNet figures.** The earlier 81.8% figure probably came
from an ImageNet-pretrained model, which was not re-run; that would need about 45 MB of
weights. ViT-B/16 was not re-run either.

## Kiran items 1 and 5, updated

| # | item | status |
|---|---|---|
| 1 | fix the classifier | ❌ **Not achieved.** Pre-registered search on a 300-tile dev set from the train split: prompting gemma and Qwen (6 configurations), pathology-tuned MedGemma 1.5 4B (8 configurations), and a LoRA fine-tune of MedGemma (fails the pre-dev format check, 12 of 20). Best dev result 57.3% (gemma, neutral labels + criteria + 6 examples). No VLM test run. |
| 5 | competent VLM + rerun the pipeline | ❌ **Not run.** No configuration qualified, so the ordering and masking pipelines were not rerun. gemma stays the only VLM in the faithfulness results. |

## Summary for Kiran (5 lines)

1. We pre-registered a search for a VLM that beats MHIST's 63.2% always-HP rate (300 dev tiles from the train split; bars ≥ 72% accuracy and ≥ 0.65 balanced; test only after a frozen prereg). Nothing qualified, so no VLM configuration was run on the test set.
2. Prompting failed: Qwen3-VL calls SSA almost always, with or without the disease names; gemma-4-31B picks the SSA-criteria letter on 84 of 100 screen tiles once the names become Class A/B with criteria, and adding 6 examples gives its best dev result, 57.3% (SSA-criteria letter on 201 of 300 dev tiles); pathology-tuned MedGemma 1.5 4B leans the other way, to HP, or simply answers "Class A" (8 configurations, none above 63% on the screen).
3. A LoRA fine-tune of MedGemma on 1,594 train tiles moved the label (exploratory: 0.80 balanced on 281 held-out train tiles that may share slides with dev), but only 12 of 20 of its answers passed the pre-dev format check (normal stop, parseable and cited; 18 needed), so by protocol it never ran on dev; exploratory: run-on answers already affect the unmodified model in its local runtime (89 of 100 dev-screen answers pass the same rule at tier 1).
4. So items 1 and 5 are documented negative results: gemma remains the only VLM in the faithfulness results, and a sourced ResNet-18 trained from scratch (75.9% ± 0.4% test accuracy, AUC 0.848, 3 seeds) replaces the unsourced CNN numbers.
5. No claim of diagnostic validity; protocol, amendments and every configuration are in `runs/competence/` (PLAN.md, RESULTS.md, STEP4_RESULT.json).

## ETA for the remaining steps

None. Steps 1–4 failed, which the brief makes a stop point. Step 5 does not run. Step 6 is
done.

## What needs you

- **Whether to stop here (recommended, and what the brief says) or to authorize a new,
  separately pre-registered follow-up.** Any follow-up is post hoc: its result would have to be
  reported as such, next to this negative result. The options:
  1. **Constrained decoding for the same adapter** (force the `cte_p1` JSON shape, e.g. at most
     3 evidence entries and a stop at the closing brace), the same label-free check (about 0.3
     GPU hours), then dev. This sends an adapter that failed a pre-registered gate to dev under
     a new decoding rule, so it reverses that gate and goes past the brief's stop. It is the
     cheapest route to item 5.
  2. **Retraining with the explanation in the target.** There are no reference explanations, so
     the target would come from the model's own text. A larger change.
  3. **Classification alone** (label read from the same adapter's logits) on dev. This also
     runs the adapter that failed the format check on dev, so it too reverses that gate and goes
     past the brief's stop. Under the current PLAN a classify-only configuration cannot close
     item 1 (the pipeline needs classify-then-explain answers), and it does not help item 5.
- Kaggle: 14.6 GPU hours are left this week (the quota resets on 2026-10-10). Datasets and
  notebooks are private; the MedGemma weights copy and the adapter must stay private (HAI-DEF
  terms). Delete them on kaggle.com when you no longer need them.
- Still open from before: revoke the Hugging Face token pasted in chat; rotate the Cerebras key;
  the pathologist and resident emails; iCloud "Optimize Mac Storage" on Documents. If you ran
  `sudo pmset -a disablesleep 1`, undo it with `sudo pmset -a disablesleep 0`.

**No claim of diagnostic validity is made. Exploratory remarks above are labelled as such.**
