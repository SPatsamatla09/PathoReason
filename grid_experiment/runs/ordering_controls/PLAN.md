# Ordering controls: pre-registered analysis plan

Written before any data. Revised twice on 2026-09-29, both times before any data:
first after an adversarial design review (confounds, power, references), then after
a re-verification pass (the `ins_self` contrasts, an incomplete-data rule, the
simulated operating characteristics below). No calls have been made under this plan.

## Question (Kiran)

On the 100 abl100 tiles (Cerebras gemma-4-31b), asking the model to explain before
classifying (`etc`) moved its label toward SSA:

- **etc vs classify-only (`co`):** 28 HP→SSA / 3 SSA→HP. The SSA rate went from
  55% to 80%.
- **etc vs classify-then-explain (`cte`):** 41 HP→SSA / 0 SSA→HP.

*The 41–0 figure is the cte contrast, not the co contrast.*

Does the shift need the model's own diagnostic explanation, or does *any* text before
the label produce it?

## Host

Cerebras has archived gemma-4-31b. Every condition, including `co`, `cte` and `etc`,
therefore runs on **one** replacement host serving the same weights, with the
upstream provider pinned (`PATHO_PROVIDER`, fallbacks disabled) and logged per call.
**Nothing on the new host is compared with the Cerebras numbers.** Those appear only
as a labelled historical reference.

## Conditions

All conditions run on the 100 abl100 tiles, with **K = 3 independent samples** per
tile per condition at temperature 1.0. That is 8 × 100 × 3 = 2,400 calls.

| condition | text before the label | diagnostic? | this tile? | written by |
|---|---|---|---|---|
| `co` | none (baseline) | — | — | — |
| `cte` | none; explanation after the label | — | — | — |
| `etc` | the model's own explanation | yes | yes | model |
| `ins_filler` | 52-word lighthouse paragraph | no | no | copied |
| `ins_checklist` | 54-word symmetric polyp checklist, HP-first or SSA-first (50/50) | yes, both directions | no | copied |
| `ins_copied` | another tile's explanation, median 51 words | yes, one direction | no | copied |
| `ins_self` | this tile's own `etc` explanation, median 52 words | yes | yes | copied |
| `describe_first` | a plain-visual description of this tile | no | yes | model |

The design holds several things fixed across conditions:

- The prompt blocks outside the instruction and output format are byte-identical to
  `co_p1`.
- The four copy conditions share one wrapper, word for word.
- Every control puts a `preamble` field *before* `label`, the position the
  explanation occupies in `etc`.
- Inserted texts are single-line.

For `ins_copied`, donor explanations are balanced 50 HP-labelled and 50 SSA-labelled,
crossed 25/25/25/25 with the recipient's true label. All donor texts avoid the
words SSA, SSL and sessile. The 50 HP-direction recipients share 9 distinct HP texts,
each reused up to 6 times.

Calls are interleaved tile-major, with a seeded condition order within each tile, so
drift in the served model cannot alias onto a condition.

## Reproduction gate

**This is checked first.**

- **Pass criterion:** same-host `co` vs `etc`, replicate 1, exact McNemar p < .05
  with HP→SSA > SSA→HP.
- **Also reported:** `cte` vs `etc` and `co` vs `cte`.
- **If the gate fails:** the report says so first, and the four controls are not
  interpreted as controls for the explain-first effect.

## Primary analysis

The primary family is the four controls: `ins_filler`, `ins_checklist`, `ins_copied`
and `describe_first`.

- **Requested format:** replicate-1 paired flips vs `co`, HP→SSA and SSA→HP, each with
  an exact McNemar test.
- **Estimand, per control c:** Δ_c = the mean over tiles of p_c(t) − p_co(t), where
  p(t) is the fraction of a tile's K samples labelled SSA.
  - It is reported with a 95% tile-bootstrap CI and a two-sided sign-flip permutation
    p.
  - The p is Holm-corrected over a **fixed** family of 4. A missing condition still
    counts against the family.
- **Reading, pre-registered:**
  - **shift:** the 95% CI excludes 0 and the Holm p is below .05.
  - **equivalent to classify-only:** the 90% CI lies inside ±10 percentage points
    (TOST, α = .05). The margin is about 40% of the Cerebras explain-first shift of
    +25 points.
  - **inconclusive:** anything else. **An inconclusive control is never reported as
    "no shift".**

**Power:** a single sample per tile detects only net shifts of about 13–22 tiles at
80% power, depending on resampling noise. K = 3 per-tile proportions are why the
estimand can separate "equivalent" from "inconclusive". The simulated operating
characteristics below give the resolution of the design.

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

## Noise floor

`co` replicate 1 vs replicates 2 and 3, compared both ways. This should be roughly
symmetric and null.

## Secondary contrasts

These are reported, are outside the Holm family, and are **all descriptive**.

- `ins_self` vs `co`.
- `ins_self` − `ins_copied`, paired. **This is not a clean tile-specificity test.**
  The two conditions differ in more than whether the text is about this tile:
  - `ins_self` texts are 80% SSA-direction; donor texts are 50/50.
  - 15 of the 100 `ins_self` texts name SSA lexically; no donor text does.
  - Both sets of texts were generated on Cerebras, not on the new host.
- **Direction-matched versions (unpaired, different tiles):** `ins_self` on tiles
  whose own text is HP-direction (20) vs `ins_copied` on tiles given an HP-direction
  donor (50); likewise for SSA-direction texts (80 vs 50).
- `ins_self` vs `co` with the 15 lexically-SSA texts removed.
- `ins_copied` split by whether the donor's cited grid cells land on the recipient's
  background (blank glass) or all on tissue.
- `etc` − `ins_self`: whether generating the text adds anything over copying the same
  text. It is confounded by host: `etc` text is generated on the new host, while the
  `ins_self` text came from Cerebras.
- `etc` − each control: whether the control moves the label as far as explaining
  does. The prompts also differ in instruction wording and output format, not only
  content.

## Exploratory

These are underpowered and labelled as such.

- **Donor direction.** The unit is the donor *text*: 9 HP-direction texts and 50
  SSA-direction texts. Each recipient's SSA-shift is centred within its
  classify-only-majority stratum. Each text's value is the mean over its recipients.
  - Statistic: Welch t between HP-text and SSA-text means.
  - p: from permuting the HP/SSA label over texts.
- **Checklist order:** the SSA-shift under the HP-first variant minus that under the
  SSA-first variant, with a tile-level permutation p.

## Manipulation checks and sensitivity

The primary analysis uses every valid response. The sensitivity analysis drops any
response that failed its manipulation:

- key order does not put `preamble` or `evidence` before `label`
- copy fidelity is below 0.90
- a `describe_first` preamble contains either:
  - histological or diagnostic vocabulary (for example crypt, gland, serrated,
    basal, dilated, benign), or
  - a plain-language paraphrase of an HP/SSA criterion (for example tooth, jagged,
    widen, sideways, L-shaped, anchor, branch).

  Both lists are matched with word boundaries. Counts of each kind, and the ten
  most common terms, are reported.

Median preamble length in words is reported per condition, so that length can be
compared with the ~52-word inserted texts.

That filter is post-treatment, so the sensitivity result is flagged as such.

Per-condition counts of lenient parses, truncations, errors and duplicates are
always reported.

## Gates that stop the analysis

The analyzer refuses to produce results, and exits non-zero, when:

- fewer than **95** of the 100 tiles have a valid label in every condition and every
  replicate, or
- any condition has no output at all, or
- records on a router host come from more than one upstream provider.

When at least 95 tiles are complete, **every** analysis is restricted to those tiles,
and the dropped tiles are listed with per-condition missing counts. "Missing" can
only mean the runner gave up after 3 unusable responses (unparseable JSON, or a
label other than HP/SSA) for that tile-replicate. Transient HTTP failures are
retried without limit.

## How outcomes are read

This table is fixed before the data. Mixed outcomes are reported as they come.

| outcome | reading |
|---|---|
| reproduction gate fails | the explain-first effect is host-specific; controls are not interpretable as controls |
| all four controls "equivalent"; `etc` shifts | the shift needs diagnostic text that is model-generated and/or about this tile; `ins_self` separates which |
| `ins_self` shifts, `ins_copied` does not | consistent with text *about this tile* driving it, but confounded by direction (80% SSA) and lexical SSA mentions: the reading stands only if the direction-matched and lexical-split rows agree |
| `ins_self` ≈ `etc` | generating the text adds little over copying it, with the caveat that the two texts come from different hosts |
| `ins_checklist` or `ins_copied` "shift" | diagnostic content before the label shifts it, even when not about this tile |
| `ins_filler` "shift" | any text before the label shifts it: a position effect, not reasoning |
| `describe_first` "shift", `ins_filler` not | model-generated attention to the image before the label shifts it, without diagnostic words |
| any control "inconclusive" | it cannot support either reading, and that is reported explicitly |
