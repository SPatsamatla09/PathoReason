# Step-4 LoRA adapter: training config and tile lists

Everything needed to understand, reproduce or extend the step-4 fine-tune of
MedGemma 1.5 4B on MHIST, **except the adapter weights**. Results and the protocol are in
`../RESULTS.md`, `../STEP4_RESULT.json` and `../PLAN.md` (step-4 addendum and Amendments
1–2).

## What is here

| file | what it is |
|---|---|
| `training_config.json` | the run's own `train_summary.json`: hyperparameters, LoRA settings, 4-bit setup, tokenisation and target token ids, package versions, data hashes, validation per epoch, timing |
| `adapter_config.json` | the PEFT config of the final adapter (r 16, alpha 16, dropout 0.05, 238 target modules) |
| `step4_meta.json` | the adapter's metadata: chosen epoch (4), final, protocol run, scoring token ids, adapter hash |
| `adapter_weights.json` | name, size and sha256 of the weights file, which is **not** in this repository |
| `split_train_val.json` | the 1,594 training and 281 validation tile names (seed 20261004; aggregate stratum counts only, no per-tile labels) |
| `base_weights_sha256.json` | sha256 of the two base-model weight files that were used |
| `train_log.jsonl`, `train_steps.jsonl` | per-epoch and per-step training logs |

Other tile lists already in this repository: `../splits.json` (the 300 dev tiles, the 100-tile
screen, the other 200 dev tiles and the 1,875-tile pool) and `../fewshot_examples.json`. The
20 tiles of the label-free format check are listed in `../STEP4_RESULT.json`.

## Two changes made for this copy

- In `adapter_config.json`, `base_model_name_or_path` was the private Kaggle mount path. It now
  reads `google/medgemma-1.5-4b-it`, and `revision` is set to the revision that was used,
  `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`. Nothing else in the file was changed.
- In `training_config.json` the same mount path is replaced by that model id and revision.

## The weights are private

`adapter_model.safetensors` (119,280,712 bytes, sha256 `13b0dbb0a807…`) is not here, for three
reasons:

1. it is a derivative of MedGemma, which is under the Health AI Developer Foundations terms;
2. it was trained on MHIST, which is under a data-use agreement;
3. it is larger than GitHub's 100 MB file limit.

It is kept in a private Kaggle dataset on the project owner's account. Ask the owner for
access; check the file against the sha256 in `adapter_weights.json`.

To use it you also need:

- your own MHIST access;
- the base model, after accepting the terms at
  <https://huggingface.co/google/medgemma-1.5-4b-it> (revision `91850547…`).

## Reproducing or extending it

- **Training:** `../../kaggle_ft/train_lora.py`, with the data bundle built by
  `../../kaggle_ft/build_bundle.py` from your own MHIST copy. The runbook is
  `../../kaggle_ft/README.md`, section 5d.
- **Loading for inference:** `../../kaggle_ft/infer_jobs.py`. It loads the base model in the
  same 4-bit nf4 setup (float32 arithmetic, vision tower not quantised) and puts the adapter on
  top without merging. Use `--attn-implementation sdpa --batch-size 4`; it gave the same answers
  as batch size 1.

## What is known about it, and the rules for a new attempt

- **Exploratory, validation only:** on the 281 validation tiles, epoch 4 scores 0.833 accuracy
  and 0.801 balanced accuracy (label read from the HP / SSA logits). These tiles may share
  slides with dev; this is not a dev or test result.
- **It failed the pre-registered label-free format check:** 12 of 20 answers ended normally
  with a parseable, cited JSON answer, and 18 were needed. So it was never run on dev. The
  answers run on (evidence lists keep growing). Forcing the JSON shape during generation is
  the obvious next thing to try.
- **For Kiran's item 1 and item 5, a new attempt is comparable only if it:**
  - uses the same dev tiles (`../splits.json`) and the same bars: dev accuracy ≥ 72% and
    balanced accuracy ≥ 0.65, then the no-image and mismatched-image controls;
  - commits a pre-registration before its one run on the 977 test tiles;
  - writes the pipeline's grid-cell JSON explanations, which the masking and ordering
    pipelines parse.

  Any re-use of this adapter on dev is a post-hoc step, because it failed a gate set in
  advance, and must be reported as such.

No claim of diagnostic validity is made.
