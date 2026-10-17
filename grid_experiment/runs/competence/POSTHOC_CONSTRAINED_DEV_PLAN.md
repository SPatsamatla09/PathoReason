# Prospective dev evaluation of post-hoc constrained decoding

Prepared for Rohan Ravulapelli on 2026-10-17 (America/Chicago).

**Status: draft, not yet preregistered.** The remote Git commit containing the
completed plan and manifest will establish the prospective timestamp. Do not
describe this draft as registered or the whole study as prospectively designed.

## Scope and prior knowledge

This is an explicitly post-hoc follow-up to the original MedGemma LoRA format
check. The original unconstrained result was reported as 12/20 and is retained,
not replaced. Constrained smoke outputs have already been inspected: the supplied
file contains 20 unique answers with normal stops and 20 successful separate
constraint audits. The project's official import/provenance check is pending.
The supplied smoke jobs match these outputs; identity with the original
unconstrained smoke tile list still needs verification before a paired claim.

This plan fixes the subsequent dev evaluation before its predictions or scores
are inspected. It does not preregister the completed smoke run. Record any prior
team exposure to results on these same dev tiles in the final manifest; this
follow-up is not a new independently held-out dataset.

## Model and fixed implementation

- Base model: google/medgemma-1.5-4b-it.
- Base revision: 91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b.
- Adapter weights SHA-256:
  13b0dbb0a80767f698bbc2f851db6c583cd00ec5b8b6dbe719a1f9f18f854444.
- The adapter metadata identifies the final protocol adapter at epoch 4. Verify
  this against the original completed training run's train_summary.json before
  dev inference; do not manufacture a replacement summary.
- Tested runner: infer_jobs_posthoc.py, SHA-256
  8b2e3fdeea023e261d64f8b610e1fc80e6908cb02e74090163979214af7bef46.
- Constraint: cte-v1, lm-format-enforcer 0.11.3; schema SHA-256
  8210906ed6904eb9a6fb260357d103f328e5fd414672edde5077bb760b55e4b7.
- Use the tested runner's Transformers tokenizer compatibility shim. Record it as
  an implementation compatibility fix, not a change to the model or prompts.
- Fixed prompt/configuration: cte_p1; preserve all job-specific seeds and prompts.
- GPU setup: T4 x2, two worker shards, per-worker batch size 4, SDPA attention,
  per-job sampling; 4-bit NF4 double quantization of language-model linear layers,
  float32 arithmetic/unquantized modules, unquantized vision tower.
- Tier 1: temperature 1.0, top_k 64, top_p 0.95, thinking opener banned,
  maximum 1,500 new tokens. Tier 2: greedy, only under the decision rule below.
- Grammar restricts field structure/order, feature names, and A1-D4 citations.
  Confidence bounds are audited after generation, not enforced by LMFE.
  Constraint validity is reported separately from the project's format gate.
- No response repair, answer-based retries, new prompts, checkpoint selection,
  font changes, or decoding tuning after dev results are inspected.

## Data and rendering

Use the existing fixed 300-tile dev split (100 screen + 200 dev_rest), as generated
by build_dev_jobs.py. Run all 300 once per permitted tier; report the screen
subset descriptively, following the step-4 addendum's full-GPU evaluation rule.
Primary job file: dev__cte_p1__none__tier1.jsonl.

The bundle was rendered on a Mac. Its grid-coordinate label font differs from
the font used for the adapter's training images, as reported by Aditya. Hold this
rendering fixed for this follow-up and disclose the difference as a potential
source of distribution shift. Do not attribute any difference from the original
experiment solely to constrained decoding without controlling other differences.

Keep dev labels and local sidecars outside Kaggle and the public repository.
No test tiles enter the private dev bundle. Model, adapter, inputs, and Kaggle
notebooks remain private. Score downloaded outputs locally.

## Preconditions and registration record

Before dev inference:

1. Confirm and record that Step 3 ended without a passing configuration. The
   builder's --step3-closed flag is an assertion, not independent evidence.
2. Obtain the original train_summary.json; verify adapter hash and final status.
3. Run the project's official smoke import/check with the local sidecar and
   training summary: at least 18/20 normal-stop answers with valid label and at
   least one valid cell. The separate 20/20 grammar audit is not a substitute.
4. Commit this completed plan and a non-sensitive manifest to remote Git before
   any dev inference. Record the repository commit, dev job SHA-256, bundle
   MANIFEST.json SHA-256, input dataset versions, adapter version, runner hash,
   exact command, scorer source commit, and software environment. Record the
   training summary hash locally; do not publish private training artifacts.
5. State the already-observed smoke results and any earlier exposure to the
   fixed dev set. If these prerequisites cannot be established, pause and
   document the missing evidence rather than silently bypassing checks.

## Metrics and decision rules

Use the repository's existing parser, import_results.py, and comp_analyze.py;
freeze their source version in the manifest. Unusable model answers count as
wrong and as misses for the relevant class. Infrastructure failures stop the run
and are not scored as model answers. Preserve raw responses and provenance.

The main dev run qualifies for image controls only if all three hold:

- Accuracy >= 0.72 on all 300 tiles.
- Balanced accuracy >= 0.65 on all 300 tiles.
- At least 90% (270/300) have a valid label and at least one valid cited cell,
  under the project's format rule.

For a qualifying tier, run both no-target-image and mismatched-target-image
controls on the same dev tiles using the existing deterministic job builder and
the same tier/constraint settings. Both controls must satisfy the repository rule:
balanced-accuracy bootstrap 95% lower bound <= 0.50 AND point estimate <= 0.58.
Report control parse rates. Use the existing bootstrap (5,000 resamples,
seed 20261003), without replacing it based on results.

Evaluate Tier 1 first. If it passes the main dev and both control gates, select
it and do not run Tier 2. If Tier 1 fails a required gate, Tier 2 is the sole
predeclared fallback, with the same main/control gates. Do not add further tiers
or variants. If neither tier passes all gates, stop this follow-up and report
the failure; do not proceed to a new test evaluation with this configuration.

Report accuracy with Wilson 95% CI, balanced accuracy with bootstrap 95% CI,
HP/SSA recalls, label parse rate, format rate, constraint-audit rate, truncation
rate, thinking-trace rate, predicted-label distribution, and runtime. Format
success does not establish visual grounding, explanation faithfulness, or
diagnostic usefulness.

## Failures, test evaluation, and reporting

Allow resumption only for logged infrastructure faults with unchanged job file,
runner/settings and compatible resume signature. Keep completed answers. Never
rerun to select a more favorable draw; log any necessary implementation amendment
before restarting and retain failed-run artifacts.

At most one passing configuration may advance. Before any new 977-tile test run,
commit a separate frozen test plan, verify prior test exposure and team protocol
eligibility, and record the pipeline runtime estimate. This document does not
authorize repeated use of an already evaluated test set as a fresh holdout.

Retain original and post-hoc outputs separately. Report the completed smoke
follow-up, dev results, controls, stopping decisions, font difference, software
fixes, deviations, and limitations regardless of whether the gates pass.

## Registration fingerprints

- Dev jobs SHA-256: `0d5caca7c8b6dd22a3102137532800ecf968d6bcf34d95b76125f2e8dfc880e9`

- Bundle manifest SHA-256: `b10a72aa93d2926c67353391b7921e53f826dda6c406c2e4cffd3cd7191b7227`

- Training summary source: repository commit cc78b86b580ea004a580ab8b4dac7e0bd738562e,
  grid_experiment/runs/competence/step4_adapter/train_summary.json.
- Summary identifies completed protocol training, selected epoch 4, and an
  adapter hash matching the constrained smoke predictions.
- Official smoke import remains required before dev inference.
- Exact Kaggle input versions and launch command will be recorded before launch.

## Pre-inference launch clarification

Official smoke dry-run passed: 20/20 qualifying answers, 18 required.
Kaggle notebook: rohanravulapelli/notebook5eac37b441.
Internet OFF; GPU T4 x2.
Dataset version labels were not readily exposed in the notebook interface.
Input identity is therefore recorded through the registered dev-job and
bundle-manifest SHA-256 hashes, the fixed runner hash, and the runner's
model-weight, adapter, and image-byte verification.
The first two setup cells passed before dev inference.
Launch uses the prepared notebook's final cell: cte_p1, tier 1, SDPA,
batch size 4 per worker, per-job sampler, and --json-constraint cte-v1.
No evaluation thresholds or stopping rules are changed.
