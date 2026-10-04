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
