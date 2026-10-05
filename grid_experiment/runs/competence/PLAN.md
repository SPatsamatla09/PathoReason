# Competence search: protocol

Written 2026-10-03, before any call. It covers Kiran items 1 and 5.

## Data, fixed once

Source: `runs/competence/splits.json`, seed 20261003.

| set | tiles | HP / SSA | built from |
|---|---|---|---|
| dev | 300 | 189 / 111 | the train split, stratified by label × agreement band to the test set's distribution |
| screen | 100 | 63 / 37 | a stratified subset of dev |
| dev_rest | 200 | — | the rest of dev |
| fewshot_pool | 1,875 | — | train minus dev; few-shot examples and any fine-tuning data come only from here |

- **All prompt and model selection happens on dev.**
- **The 977 test tiles** get exactly one confirmatory run per final configuration. The
  configuration is frozen first, and a pre-registration file is committed before that
  run.

## What a configuration is

A configuration is a model (with a pinned provider and revision) plus a prompt.

**Images.** The pipeline's gridded tile, rendered exactly as for the gemma runs.

**Prompts.** The pipeline's classify-then-explain JSON format (label, confidence, evidence
with grid cells), so a winning configuration can drive the ordering and masking pipeline
unchanged. Only the class block, the label tokens and any examples change.

**Counterbalancing of neutral labels.** Each tile gets a fixed assignment: Class A = the
HP criteria or Class A = the SSA criteria. The assignments are 50/50 within each label ×
band stratum, seeded. That gives one call per tile, and the assignment is kept for that
tile in any later pipeline run. The two halves are also reported separately, as
descriptive results.

## Scoring

- **Accuracy.** Correct answers over **all** n tiles. An unparseable or invalid answer
  counts as wrong.
- **Balanced accuracy.** (HP recall + SSA recall) / 2, where an unusable answer counts as
  a miss for its tile's true class.
- **Also reported:** per-class recall, parse rate, SSA-call rate, and cost.

## Pass bars, fixed now (not lowered, no metric changes)

- **Screen (100 tiles).** Accuracy ≥ 65% means the configuration also runs on dev_rest.
  Otherwise it stops.
- **Dev pass (all 300 tiles).** Accuracy ≥ 72% **and** balanced accuracy ≥ 65%.
- **Image controls,** for any configuration that passes dev. Both are run on all 300
  dev tiles:
  1. **No image:** the same prompt with the tile omitted.
  2. **Mismatched image:** each tile is shown another dev tile's image (a seeded
     derangement), and scored against its own label.

  A control is OK only if its balanced accuracy has fallen to about 50%. Concretely:
  the 95% bootstrap CI includes 0.50 **and** the point estimate is ≤ 0.58. If either
  control holds up, the model is not using the image, and the configuration **fails**.
- **Test, confirmatory (977 tiles, one run).** The configuration is competent if the
  Wilson 95% lower bound on accuracy is above 0.632 **and** the bootstrap 95% lower bound
  on balanced accuracy is above 0.50.

## Order of steps

Each step stops at the first configuration that passes dev (including the controls),
then goes to step 5.

**Step 1: neutral labels.** "Class A" / "Class B" with written criteria, in parallel
structure and of similar length. The prompt contains no "hyperplastic" and no "sessile
serrated". Models: gemma-4-31b-it on Friendli first, then the cheapest other model by
measured cost.

**Step 2: criteria plus few-shot.**

- The same WHO-based criteria, plus 6 labelled example tiles (3 HP, 3 SSA).
- The examples are drawn once from fewshot_pool, with a fixed seed, and are the same for
  every call.
- Two label variants: the real class names, and neutral labels (counterbalanced as
  above).
- The same model order as step 1.

**Steps 3–4:** a local pathology-tuned model, then a local LoRA fine-tune. Both run only
on local or free compute, and both are subject to the stop points.

## Budget

- **Hard cap:** $1.00 of API spend in total (balance before this protocol: $1.41).
- **Before every paid batch:** measure the cost on 5 calls, print the estimate, and skip
  the batch if it would break the cap.
- **Before anything long:** time 20 calls and print the ETA.

**Exploratory analyses are labelled as such. No claim of diagnostic validity is made.**

## Addendum: step 3, local MedGemma (written 2026-10-04, before any dev call with this model)

A pre-flight review (model-card research and a protocol critique) was run first. Its
changes are adopted below. No bar is lowered and no metric is changed.

### Runtime, frozen now

| | |
|---|---|
| model | `google/medgemma-1.5-4b-it`, HF revision `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b` (Gemma-3 architecture, SigLIP 896 px, 256 tokens per image) |
| weights | official safetensors, converted locally with llama.cpp `convert_hf_to_gguf.py` |
| text model | **Q8_0**, sha256 `2b24548a…c026d605` |
| vision projector | f16, sha256 `5cfd0926…383e8443` |
| engine | llama.cpp commit `7f2dd88b0ac393357ae6a9e1992185a48c20b13b`, Metal, with one local patch (below) |
| server | `llama-server -ngl 99 -c 8192 -np 1 --jinja --special`, bound to 127.0.0.1; embedded chat template; no system prompt |
| input | the pipeline's gridded 224×224 PNG, image before text in one user turn; the runtime resizes to 896×896 (bilinear, as the model's own preprocessor does) |

Everything runs on this Mac. No image leaves it.

**Forced changes vs the gemma runs:**

- **Local runtime** instead of a hosted API.
- **Q8_0 text weights.** The Mac has 16 GB of RAM and a history of crashes under memory
  pressure. Q8_0 is the only quantization used, and it is not changed after any dev
  call. It was not compared against f16, for the same memory reason.
- **Local Metal patch** (`llama_cpp_macos13_metal.patch`). On macOS 13.2 the engine's
  no-copy GPU read-back returns nil. The patch reads back through a temporary buffer
  instead; the values are the same, only read synchronously.
- **Explicit sampling values** (see Decoding). The gemma runs sent temperature only, and
  the host defaults are unknown.

**Thinking traces.** MedGemma 1.5 can emit an unprompted trace
(`<unused94>thought … <unused95>`). It is stripped before parsing, and its rate is
reported per configuration. An unclosed trace leaves nothing to parse, and counts as
wrong.

### Decoding, decided now

- **Tier 1:** temperature 1.0, top_k 64, top_p 0.95, min_p 0, repeat penalty off, and
  a fixed per-tile seed. Temperature 1.0 is what the pipeline uses; the top_k and top_p
  values are the Gemma family's published ones.
- **Tier 2,** run only if no tier-1 classify-then-explain configuration passes dev: the
  same configurations in the same order at **temperature 0**. That is the setting
  Google documents for this model.
  - If a tier-2 configuration is the one that passes, the pipeline rerun uses
    temperature 0.
  - Its repeat samples are then identical, so the ordering experiment has K = 1.
  - That is recorded as a forced change.

### Configurations and stopping rule

One draw per tile per configuration. No reruns, except for logged infrastructure faults.

1. **`co_p1`**, the pipeline's classify-only prompt. It is run and reported, but it is
   **not a stopping configuration**: the masking pipeline needs classify-then-explain
   answers, and competence is prompt-specific.
2. **`cte_p1`**, the pipeline's classify-then-explain prompt.
   - It is added to the brief's list because it is the prompt the masking pipeline uses
     unchanged.
   - It comes first in the stopping order for that reason.
3. **`neutral_cte`**, Class A/B plus criteria: the best of step 1.
4. **`neutral_crit_fs`**, Class A/B plus criteria plus 6 examples: the best of step 2.
   - Few-shot prompting with images is outside the settings Google evaluated.

**The search stops** at the first of 2–4 that passes dev including both controls. If
`co_p1` passes and `cte_p1` fails, that is reported, and the search continues.

**At most one step-3 configuration goes to the test set.**

**If the passing configuration is 3 or 4,** every forced pipeline change is listed in the
pre-registration before the test run:

- re-rendered prompts
- label mapping
- multi-image calls

**Before the test run,** the measured per-call time and the full-pipeline ETA are
printed. An unworkable ETA is a stop point for the user.

### Bars and scoring

**Bars are unchanged:**

- **Screen:** accuracy ≥ 65% to advance. A screen of 64/100 or lower is final. The
  screen is a futility gate only; no claim rests on it.
- **Dev:** accuracy ≥ 72% and balanced accuracy ≥ 65%, on all 300.

**Scoring:**

- Only a response the model produced is scored. An unparseable model response counts
  as wrong.
- A transport or server failure stops the run and is never scored.
- Every call must show at least 256 prompt tokens per image, or the run stops. That
  proves the image was ingested.

**Image controls: a clarification made before any control call.** A control is OK if
its balanced accuracy is **not above chance**: the bootstrap 95% lower bound is ≤ 0.50
and the point estimate is ≤ 0.58. A model that refuses to answer without an image scores
below chance and should not fail the control.

- Parse rates are reported beside each control.
- For few-shot prompts, the no-image control omits the target tile only.
- The mismatched-image control is the decisive one.

**Also reported, without gating:**

- thinking-trace rate
- answers cut off at the token limit
- the share of answers citing at least one and at least three valid grid cells

### Smoke and timing

Smoke and timing calls use 20 training-pool tiles: never dev, never test, and never the
six few-shot examples. They are written to `runs/competence/smoke/`, which is never
scored. They check label-free properties only:

- valid JSON
- a normal stop
- image tokens present
- per-call time

### Context from the model card, not a claim

- Google reports no training or evaluation on MHIST or on HP-vs-SSA polyp subtyping.
- Its nearest official result is colorectal tissue typing on 224-px patches, which is
  flagged out-of-distribution and is poor zero-shot.
- The model is a research starting point, not for clinical use.

### Amendment 1 to the step-3 addendum (2026-10-04, after the first 3 smoke calls, before any dev call)

**What the smoke test showed.** These were training-pool tiles, checked on label-free
properties only. With `cte_p1`, 2 of the first 3 calls opened an unprompted thinking
trace of about 1,300 tokens. Each exhausted the 1,500-token limit before the JSON
answer was complete, and took about 4 minutes.

**Change: thinking is switched off.** Every call bans the token that opens a trace
(`<unused94>`, id 100) through `logit_bias`. The reasons:

- Google's image-classification evaluations ran with thinking off.
- Hidden reasoning before the label would void the classify-then-explain and ordering
  design.
- With thinking on, a call takes about 4 minutes, which makes even the dev set
  impractical on this machine.

**A thinking-on variant is not run.** That is recorded as untested. It is not a tested
negative.

**The earlier smoke file** is moved to `smoke/superseded_thinking_on/`.

### Amendment 2 to the step-3 addendum (2026-10-04, still before any dev call)

These changes come from the third pre-flight reviewer (runtime research). All calls so
far used training-pool tiles only.

**Prompt format: now token-identical to Google's own preprocessing.**

- The chat endpoint drops the blank lines that HF's `Gemma3Processor` puts around each
  image.
- Local calls now use the raw `/completion` endpoint, with a prompt built to match:
  text parts stripped and concatenated, each image wrapped in blank lines, no system
  prompt.
- It was checked against the official processor for single-image, few-shot and text-only
  layouts. The token ids are identical (822 tokens for `cte_p1` with one image).

**Server flags added:** `--swa-full --cache-ram 0 -fit off`.

- `--swa-full` gives prefix reuse for the few-shot examples.
- `--cache-ram 0` switches off the host-RAM prompt cache.
- `-fit off` prevents any silent CPU fallback.

The full line: `llama-server -ngl 99 -c 8192 -np 1 --jinja --special --swa-full
--cache-ram 0 -fit off`.

**Prompt caching** is left on. Reusing a cached few-shot prefix can change logits at
float level, and that is accepted and recorded.

**The Metal patch** was reviewed and tested independently, and found sound. Its
read-back returns matching data. The crash is known upstream: issue #16266, open PR
#29814.

**Dissent recorded.** The runtime reviewer advised against suppressing thinking if the
goal were to reproduce the model's default behaviour. Amendment 1 stands, for the
reasons given there. The competence result therefore describes MedGemma 1.5 4B with
thinking off.

**Smoke files from the chat-endpoint format** are moved to
`smoke/superseded_chat_endpoint/`.

## Addendum: step 4, LoRA fine-tune on private cloud GPU (written 2026-10-04, before any step-4 training or evaluation)

**Why cloud.** PyTorch has no GPU on this Mac, so step 4 cannot run locally. On
2026-10-04 the user approved Kaggle or Colab under strict privacy rules:

- a private Kaggle dataset and a private notebook on the user's account only, or the
  user's private Drive on a private Colab runtime
- never public, and never link-shared

No dev or test tile is used for training or for model selection.

**Step order is unchanged.** Step-4 training may run while step 3 is still screening,
to save time. But the step-4 dev evaluation is run only if step 3 ends with no passing
configuration. If step 3 passes, step 3 is the winner whatever step 4 would have scored.

### Model and data, fixed now

- **Base model:** `google/medgemma-1.5-4b-it`, revision `91850547…` (official weights,
  transformers, GPU).
- **Training set:** `fewshot_pool`, i.e. the train split minus dev (1,875 tiles).
  - A validation slice of 15% is held out from the pool, stratified by label ×
    agreement band, with seed 20261004.
  - Checkpoint and epoch selection use that validation slice only.
- **Input:** the pipeline's gridded tile plus the pipeline's `cte_p1` prompt, in the
  official chat format (image first, no system prompt).
- **Target, label-only.** The assistant text is `{\n  "label": "<HP|SSA>"`. Loss is
  taken on those target tokens only. No explanation text is ever supervised.
- **Class imbalance:** handled by a class-weighted loss, with inverse-frequency weights.

### Hyperparameters, one setting, no search on dev

- LoRA rank 16, alpha 16, dropout 0.05, on the language model's linear layers.
- The vision tower is frozen.
- Learning rate 2e-4 with a cosine schedule and 5% warm-up.
- Effective batch size 8.
- Up to 4 epochs.
- The checkpoint with the best validation balanced accuracy is kept. The label is read
  from the HP-vs-SSA logits at the label position.
- 4-bit loading may be used if memory requires it. Whether it was is recorded.
- Seed 20261004.

### Dev evaluation, one run per decoding tier

Same bars, same parser.

- **What is run:** the selected checkpoint generates the full `cte_p1` answer on all
  300 dev tiles.
- **Decoding:**
  - Tier 1: temperature 1.0, top_k 64, top_p 0.95, thinking token banned, up to 1,500
    new tokens.
  - Tier 2 (temperature 0): only if tier 1 fails.
- **Pass:**
  - accuracy ≥ 72% and balanced accuracy ≥ 65%;
  - **format:** at least 90% of dev answers parse with a valid label and at least one
    valid cited grid cell, because the masking parser needs citations;
  - both image controls not above chance: no image, and mismatched image, by the rule
    in the step-3 addendum.
- **Reporting:** the 100-tile screen subset is reported, but with a fast GPU the whole
  dev set is run in one pass.

**At most one step-4 configuration goes to the test set.** That happens only after a
pre-registration is committed.

### Privacy and licence

- Kaggle datasets and notebooks are created private.
- The MedGemma weights are not redistributed. They are either uploaded to the user's own
  private dataset, or downloaded inside the private notebook with the user's token held
  as a Kaggle secret.
- Any LoRA adapter stays private (HAI-DEF terms).
- No claim of diagnostic validity is made.


### Amendment 1 to the step-4 addendum (2026-10-04, before any step-4 training or dev evaluation)

These points come from the adversarial review of the cloud fine-tune code. No bar is
lowered and no metric is changed.

**A. What the dev result can and cannot show**

- **Slides.** Dev was carved from the train partition tile by tile, and `annotations.csv` has no slide id. Pool
  tiles used for fine-tuning may therefore come from the same slides as dev tiles, and a step-4 dev result can be
  optimistic. The dev result is a gate only. The confirmatory test run is the evidence. (Checked on
  2026-10-04: no two of the 3,152 tiles in pool, dev and test have identical pixels.)
- **Format bar, where it is applied.** `comp_analyze.py` applies the accuracy bars only. The format bar (at least
  90% of all 300 dev answers with a valid label and at least one valid cited grid cell) is computed by
  `kaggle_ft/import_results.py` and written to `runs/competence/step4_gate__<config>__<model>__<tier>.json`. A
  step-4 configuration passes dev only if `comp_analyze.py` says PASS, that file says `"format_ok": true`, and
  both image controls are OK.
- **Label-free check before dev.** Before any dev job, the final adapter generates the full `cte_p1` answer for
  the 20 smoke tiles of the step-3 addendum (training-pool tiles; no label is used). Every answer must end with a
  normal stop and at least 90% must have a valid label and a valid grid cell. If not, the dev evaluation is not
  run with that adapter.
- **One adapter on dev.** Only the final adapter of the completed protocol run (the epoch chosen on the
  validation slice after all four epochs) is evaluated on dev. Epoch checkpoints and the best-so-far adapter of
  an unfinished run are refused by the code, and every dev import is logged in
  `runs/competence/step4_imports.ndjson`.
- **Step order.** Dev job files are built only once step 3 has ended with no passing configuration
  (`build_dev_jobs.py --step3-closed`).

**B. Choices the addendum did not fix, fixed now**

| | |
|---|---|
| gradient clipping | max norm 0.3 (Google's notebook) |
| optimiser | AdamW, betas 0.9 / 0.999, eps 1e-8, weight decay 0 |
| loss | mean cross-entropy over the target tokens of an example, times its class weight, averaged over the 8 examples of an optimiser step |
| LoRA targets | the 7 linear layers of each of the 34 language-model blocks (238 modules); `lm_head` is not adapted |
| ties | equal validation balanced accuracy: the earlier epoch |
| 4-bit scope | only where memory requires it: the language model's linear layers. The SigLIP vision tower, the projector, the embeddings and `lm_head` are not quantised. Checked on the loaded model and recorded |
| validation read-out | argmax of the HP / SSA logits at the label position (as in the addendum). The expected accuracy under tier-1 sampling is also logged, for information only; it is not used for selection |

**C. The supervised target, amended before any training**

The assistant text is `{\n  "label": "<HP|SSA>`: 8 tokens, ending at the label token.

- The closing quote is no longer part of the target. In a full answer the label is
  followed by the single token `",`, not by a bare quote.
- Loss is taken on those 8 target tokens only.
- The label is still read from the HP-vs-SSA logits at the label position.

**GPU and precision.** Kaggle's free GPU is the T4, which has no native bf16. The
language-model linear layers are therefore loaded in 4-bit (nf4), with float32
arithmetic, for training and for every later evaluation. The vision tower and the
projector stay unquantised. This is recorded as a forced change: the competence result
then describes that 4-bit model plus its adapter.


### Amendment 2 to the step-4 addendum (2026-10-04, 22:10 EDT, while the protocol training run is in its first epochs; before the final adapter exists)

**A. The label-free check before dev: the brief's 90% bar, not 20 of 20**

Amendment 1 said: before any dev job, the final adapter answers the 20 smoke tiles, and
"every answer must end with a normal stop and at least 90% must have a valid label and a
valid grid cell". The "every answer" clause came from the code review. It is stricter
than the user's brief, whose format bar is a parse rate of at least 90%.

The user decided on 2026-10-04 (before the final adapter existed): *"Run the format check
at the 90% bar from my original brief: the fine-tuned model passes if at least 18 of 20
answers parse correctly and end normally. Do not use the stricter 20 of 20 version."*

**Rule from now on.** An answer counts only if it ends with a normal stop **and** has a
valid label **and** at least one valid cited grid cell. The check passes if at least 18
of the 20 answers count (90%). A cut-off answer never counts, whatever can be read from
it. If fewer than 18 count, the dev evaluation is not run with that adapter. Code:
`kaggle_ft/import_results.py`, `smoke_check()`.

**What was known when this was changed** (disclosed because it bears on the old clause):

- The unmodified model (local Q8_0, `cte_p1`, tier 1) reached the 1,500-token limit in 10
  of 100 dev-screen answers. That run is already scored and stopped at the screen.
- The 32-tile smoke adapter (`protocol_run: false`, never scored) reached the limit in 4
  of the same 20 smoke tiles. Under the new rule it has 16 of 20 and fails, as it did
  under the old one.
- No output of the final adapter existed. The protocol run had used about 3 of its
  roughly 12.5 GPU hours.

**What is not changed.** The dev bars (accuracy at least 72%, balanced accuracy at least
65%), the dev format bar (at least 90% of all 300 dev answers with a valid label and a
valid grid cell; an answer cut off before its JSON closes does not parse and counts
against it and against accuracy), the image-control rule, the test rule, one adapter on
dev, and the step order. This check is label-free and is not a reported outcome; it only
decides whether dev is run.

**B. Inference settings for every evaluation run of the fine-tuned model, fixed now**

The addendum did not fix the batch size or the attention implementation. Measured on
Kaggle's T4 x2 with the smoke adapter on the 20 smoke tiles (no label used):

| setting | result |
|---|---|
| batch size 1, eager attention | 20/20 jobs, 82 s per job per GPU |
| batch size 4, eager attention | out of GPU memory |
| batch size 4, sdpa attention | the same 20 answers token for token; 1.43 times faster |
| batch size 8, sdpa attention | the same 20 answers token for token; no faster than batch 4 on this sample |

All runs of the fine-tuned model (format check, dev, controls, test, pipeline) use
`--attn-implementation sdpa --batch-size 4`, tier-1 sampler `per-job`. Chosen for speed on
pool tiles before any dev tile was run. Both are written into every output line.

**C. Deviations from the runbook, recorded**

- The protocol training run was started when the smoke training run had passed and the
  smoke inference run had just been started (the runbook asks for all three smoke runs
  first). The smoke inference run then completed 20/20 jobs. Nothing was changed in the
  trainer between the smoke run and the protocol run (same file hash).
- Kaggle mounts attached datasets at `/kaggle/input/datasets/<owner>/<slug>`. The header
  that `kaggle_push.py` inserts now resolves input paths at run time. The trainer and the
  inference script are unchanged.
