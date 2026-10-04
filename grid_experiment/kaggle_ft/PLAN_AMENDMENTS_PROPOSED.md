# Proposed text for `runs/competence/PLAN.md` (step-4 addendum)

`kaggle_ft/` code may not edit `PLAN.md`, so the paragraphs below are a draft for you to paste into the PLAN
**before any step-4 training or dev evaluation**. Sections A and B describe what the code already does. Section C
is a decision: the code stays on the PLAN's wording until you take it.

---

### Amendment 1 to the step-4 addendum (date it; before any step-4 training or dev evaluation)

These points come from the pre-flight review of the Kaggle code. No bar is lowered and no metric is changed.

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

---

### C. Decision needed: the supervised target (only with a dated amendment, before training)

**The issue.** The addendum's target is `{\n  "label": "<HP|SSA>"`, 9 tokens, and its last token is the bare
quote `"` (token id 236775). In a full `cte_p1` answer the label is followed by the single token `",` (id 827).
So the target trains a token the full answer never uses, and nothing after the label is trained, while LoRA
changes all 238 language-model layers for 800 steps. Whether the adapter still writes confidence and evidence
with grid cells after that is not known. The 90% format bar depends on it.

**Option 1: keep the PLAN's target** (the code's current setting, `TARGET_VARIANT = "plan"`). Nothing to amend.
The label-free check above tells you, before dev is touched, whether the format survived. If it did not, a
retrain needs option 2 and costs another 7 to 9.5 GPU hours.

**Option 2: stop the target at the label.** Amendment text:

> **Target, amended before any training.** The assistant text is `{\n  "label": "<HP|SSA>` (8 tokens, ending at
> the label token). The closing quote is no longer part of the target, because in a full answer the label is
> followed by the single token `",`, not by a bare quote. Loss is taken on those 8 target tokens only. The
> label is still read from the HP-vs-SSA logits at the label position.

Then set `TARGET_VARIANT = "label_end"` in `kaggle_ft/train_lora.py` (one line) and re-run
`test_train_data_local.py`, which checks both variants.
