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
