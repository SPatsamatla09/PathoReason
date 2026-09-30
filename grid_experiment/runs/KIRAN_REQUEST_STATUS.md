# Kiran's four requests: status as of 2026-09-29, 22:10

**Update.** An OpenRouter key was added, and the same weights (`google/gemma-4-31b-it`) are
now pinned to the Friendli upstream.

- **Request 1:** complete.
- **Request 2:** running.

The "blocker" section below is kept for the record.

| # | request | status | file to pull into the write-up |
|---|---|---|---|
| 1 | four new ordering conditions, n ≥ 100, flips vs classify-only | **done** 2026-09-29 on OpenRouter/Friendli; verified by independent recomputation | **`runs/ordering_controls/RESULTS__friendli.md`** (write-up-ready); full output `analysis__openrouter__google-gemma-4-31b-it__friendli.json` |
| 2 | masking sweep to 400–500 tiles, pooled tile-level p | **running** on OpenRouter/Friendli (started 21:55) | now: `runs/masking_pooled_legacy_analysis.json`; after the run: `runs/masking_pooled_analysis.json` |
| 3 | two-item pathologist rating instrument | **done** | `outreach/` (the rating instrument; local only) |
| 4 | no second VLM, no tissue-type-matched controls | respected | — |

## The blocker for 1 and 2

Cerebras has archived gemma-4-31b. Re-checked today: the chat endpoint returns
HTTP 404 `model_archived`, and the account's model list now has only
`gpt-oss-120b` and `qwen-3.8-27b`.

The same weights are served elsewhere, for example OpenRouter
(`google/gemma-4-31b-it`) and Google AI Studio. No key for any of those hosts
exists on this machine. Local inference is not an option: the Mac has 16 GB of RAM.

Nothing was run on a substitute model. That would be the "second VLM run" that
request 4 rules out.

## 1. Ordering conditions

**What is ready.** The prompts are rendered in `prompts/rendered_ordering_controls/`
and all design invariants pass. The runner, analyzer and pre-registration
(`runs/ordering_controls/PLAN.md`) are written.

**Conditions.** Kiran's four:

- `ins_filler`: unrelated text
- `describe_first`: describe the image without diagnostic reasoning
- `ins_checklist`: generic pathology checklist
- `ins_copied`: another tile's explanation

One secondary condition is added: `ins_self`, the tile's own explanation copied in.
It separates "text about this tile" from "text the model generated".

**Sample.** The same 100 abl100 tiles as the existing ordering sweep, K = 3 samples
per tile per condition.

**Calls.** 8 conditions × 100 × 3 = 2,400.

**Baseline has to be re-run.** Because the old host is gone, `co`, `cte` and `etc`
are re-run on the new host, and every comparison stays within that one host. The
Cerebras figures (28/3 and 41/0) appear only as a labelled reference.

**Reproduction check first.** The report opens with a pre-registered check: does
explain-first still shift labels toward SSA on the new host? If it does not, the
controls cannot be read as controls for that effect, and the report says so first.

**Output, in the requested format.** For each condition vs classify-only:

- replicate-1 HP→SSA and SSA→HP flip counts, flip rate and exact McNemar p
- the pre-registered Δ SSA-rate with 95% CI and Holm p
- a shift / equivalent / inconclusive reading. "Inconclusive" is never reported as
  "no shift".

**Design resolution.** Simulated at K = 3, 100 tiles:

Simulated operating characteristics: K = 3, 100 tiles, 400 simulations per row.

| true Δ (SSA-rate points) | reads "shift" | reads "equivalent" | reads "inconclusive" |
|---|---|---|---|
| 0 | 1.0% | 96.3% | 2.8% |
| 3 | 7.5% | 82.5% | 10.0% |
| 5 | 18.2% | 54.7% | 27.0% |
| 7.5 | 52.2% | 19.3% | 28.5% |
| 10 (the margin) | 81.5% | **2.5%** | 16.0% |
| 12.5 | 92.7% | 0% | 7.2% |
| 15 | 99.0% | 0% | 1.0% |

**Simulation setup.** Tile base rates are 30% near-certain HP (0.03), 30%
near-certain SSA (0.97) and 40% uniform on 0.15–0.85. The true shift is exact, and
larger on HP-leaning tiles. The other three members of the Holm family are
independent nulls. Script and raw output: `runs/ordering_controls/oc_power2.py`, `oc_power2_K3.txt`. Monte-Carlo error is about ±2.5 points.

**How to read it.**

- **Equivalence error is controlled.** At a true shift exactly on the ±10-point
  margin, "equivalent" is returned 2.5% of the time, within the TOST α of 5%.
- **"Equivalent" means under 10 points, not zero.** A real 5-point shift is called
  "equivalent" about half the time.
- **Power.** The design has about 80% power to call a 10-point shift.

## 2. Masking extension

**Current citable result.** Sweeps B + C on Cerebras, one unit per tile:

- **185 unique tiles**: 54 cited-more, 47 control-more, 84 tied
- **sign test p = 0.55**

Source: `runs/masking_pooled_legacy_analysis.json`. The earlier "220 tiles, 61 vs 57"
counted 15 tiles twice (sweep C reused B's records) and the pilot's 18 re-tested
tiles as separate units.

**Extension, ready to launch.** 340 new tiles drawn from test tiles never masked
before, proportional by label × agreement band, with seed 20260929
(`runs/.mask_ext_tiles.json`). The design is unchanged:

- mean-fill, blur and black occlusions
- the three highest-tissue cited cells
- area-matched tissue controls

**Expected size.** About 313 of the 340 should have usable controls, giving **about
498 unique tiles pooled**.

**Calls.** 340 baseline calls, then about 1,880 masked calls (6 per usable tile).

**Citations are regenerated on the new host.** Each extension tile gets fresh
classify-then-explain citations and a fresh baseline on the new host. A gate stops
the chain before masking if the new host agrees with Cerebras labels on fewer than
60% of these tiles.

**Mixed hosts, stated in advance.** The pooled ~498-tile figure will mix two
deployments: Cerebras for B + C, the new host for the extension. The pooled file
reports each host separately and runs a Fisher heterogeneity test. If the hosts
disagree (p < .05), report per host, not the pooled p.

## To run 1 and 2 (one command, resumable, both experiments independent)

Needs an OpenRouter key (or an equivalent host) and a pinned upstream provider:

```bash
export PATHO_BASE_URL=https://openrouter.ai/api/v1
export PATHO_MODEL=google/gemma-4-31b-it
export PATHO_PROVIDER=<one upstream slug>
export PATHO_KEY_ENV=OPENROUTER_API_KEY OPENROUTER_API_KEY=<key>
export PATHO_MIN_INTERVAL_S=2
python3 probe_host.py
screen -dmS kiran caffeinate -dimsu bash -c 'cd /Users/adityak/Documents/mhist/grid_experiment && ./run_kiran_batch.sh'
```

The probe checks four things before any money is spent: text, vision, the JSON
format, and that the served provider matches the pin. The chain logs to
`runs/logs/kiran_batch__<host>.log`.

## 3. Rating instrument

`outreach/` (the rating instrument; local only) has two separate items per case:

- **(a) visibility:** is the named feature, as described, visible in the outlined
  cells? Scored 1–4 or NA.
- **(b) diagnostic support:** assuming the feature is present as described, does it
  support the model's claimed diagnosis? Scored 1–5, and answered even when (a) is
  "not visible".

The 30-case packet, the answer sheet and the internal analysis plan are also in
`outreach/`. That folder is gitignored and never public, because it contains MHIST
pixels.

## Verification done without the model

All of this used a stubbed model. Planted effects were recovered:

- a +0.30 donor-direction effect came back as +0.28 at the donor-text level
  (p = 0.0008)
- the planted null conditions (filler, describe) came back at −0.04 and −0.06. That
  matches the stub's classify-only baseline, which drew +0.05 high by chance.

The gates refused when they should:

- under 95 complete tiles
- a missing condition
- a second provider
- unattempted baseline tiles

Resume re-ran exactly the missing calls, and the batch script ran masking even when
ordering failed.
