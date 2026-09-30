# Ordering controls: results

## Setup

| | |
|---|---|
| model | gemma-4-31b-it, via OpenRouter, pinned to the Friendli upstream |
| run | 2026-09-29, 19:45–21:55 EDT |
| tiles | the 100 abl100 tiles (50 HP / 50 SSA by consensus) |
| sampling | K = 3 samples per tile per condition, temperature 1.0 |
| calls | 8 conditions × 100 × 3 = 2,400. All valid: strict JSON parse, HP/SSA label, one provider, no retries given up |
| analysis plan | `PLAN.md`, fixed before any data |
| verification | numbers recomputed from the raw records by independent code; every value matched |

`†` marks numbers computed after the data came in. They are not part of the
pre-registration; report them as exploratory.

## Table 1: flips and direction vs classify-only (the requested format)

- **Flips** are paired over the same 100 tiles, replicate 1, with an exact McNemar test.
- **SSA rate (rep 1)** is directly comparable to the earlier Cerebras numbers.
- **SSA rate (K = 3)** is the rate over all 300 calls.

| condition | HP→SSA | SSA→HP | flip rate | McNemar p | SSA rate (rep 1) | SSA rate (K = 3) |
|---|---|---|---|---|---|---|
| classify-only (`co`), baseline | — | — | — | — | 50% | 50.0% |
| **noise floor:** `co` rep 1 vs rep 2 | 5 | 11 | 0.16 | 0.21 | — | — |
| **noise floor:** `co` rep 1 vs rep 3 | 12 | 6 | 0.18 | 0.24 | — | — |
| classify-then-explain (`cte`) | 3 | 24 | 0.27 | 4.9 × 10⁻⁵ | 29% | 33.3% |
| explain-then-classify (`etc`) | 23 | 9 | 0.32 | 0.020 | 64% | 67.3% |
| unrelated filler (`ins_filler`) | 9 | 8 | 0.17 | 1.0 | 51% | 51.0% |
| generic checklist (`ins_checklist`) | 18 | 4 | 0.22 | 0.004 | 64% | 63.0% |
| other tile's explanation (`ins_copied`) | 28 | 22 | 0.50 | 0.48 | 56% | 56.0% |
| describe the image first (`describe_first`) | 21 | 10 | 0.31 | 0.071 | 61% | 60.7% |
| own explanation copied (`ins_self`), secondary | 35 | 4 | 0.39 | 3.4 × 10⁻⁷ | 81% | 81.0% |

**Classify-then-explain vs explain-then-classify: 36 HP→SSA / 1 SSA→HP**
(McNemar p = 5.5 × 10⁻¹⁰).

**Cerebras reference, same tiles, replicate 1.** For reference only; never compared
statistically:

- classify-only → explain-then-classify: 28 / 3
- classify-then-explain → explain-then-classify: 41 / 0
- SSA rate: 55% classify-only, 39% classify-then-explain, 80% explain-then-classify

## Table 2: the pre-registered test

- **Δ** is the mean over tiles of the change in each tile's SSA rate across its 3
  samples, vs classify-only.
- **CI** is a 95% tile-bootstrap interval.
- **Holm p** is corrected over the fixed family of the four controls.
- **Reading, fixed in advance:**
  - *shift:* the CI excludes 0 and Holm p < .05.
  - *equivalent:* the 90% CI is inside ±10 points.
  - *inconclusive:* anything else.

| condition | Δ SSA rate | 95% CI | Holm p | pre-registered reading |
|---|---|---|---|---|
| explain-then-classify (reproduction check) | **+17.3 pts** | [+9.0, +25.7] | (not in family; sign-flip p = 0.0003) | **reproduction check PASSED** |
| unrelated filler | +1.0 | [−5.0, +7.0] (90%: [−4.0, +6.0]) | 0.83 | **equivalent to classify-only** |
| generic checklist | **+13.0** | [+6.7, +19.3] | 0.0004 | **shift toward SSA** |
| other tile's explanation | +6.0 | [−7.3, +19.0] | 0.80 | **inconclusive** (see finding 4) |
| describe the image first | **+10.7** | [+4.7, +17.0] | 0.0033 | **shift toward SSA** (see finding 5) |

Design resolution, simulated before the data: this design calls a true 10-point
shift a "shift" about 80% of the time. It calls a true 5-point shift "equivalent"
about half the time. "Equivalent" therefore means *under 10 points*, not *zero*.

## Findings

### 1. The explain-first shift reproduces on the new host

Explaining before classifying moved labels toward SSA:

- Δ = +17.3 points [+9.0, +25.7].
- Replicate 1: 23 HP→SSA vs 9 SSA→HP.

The pre-registered check passed, so the four controls can be read as controls for
this effect.

**What this does not show.** It shows the direction is present on this host. It is
not a statistical comparison with Cerebras, and the plan rules that out. The
replicate-1 counts also depend on which sample is used: replicate 2 gives 34/6 and
replicate 3 gives 21/11†. The K = 3 Δ is the stable measure.

### 2. Classify-then-explain is not a neutral baseline

With no text before the label, merely asking for the explanation *after* the label
moves labels toward HP:

- vs classify-only: 3 HP→SSA vs 24 SSA→HP.
- Δ = −16.7 points†.

So the 36/1 contrast (41/0 on Cerebras) adds two opposite shifts together, and
overstates the explain-first effect. **Classify-only is the right baseline**, and
the paper's headline contrast should use it.

### 3. Unrelated text before the label does not produce the shift

Copying a 52-word lighthouse paragraph into the same slot left the label
statistically equivalent to classify-only:

- Δ = +1.0 points, 90% CI [−4.0, +6.0].
- Flip rate 0.17, at the noise floor.

The filler used the same wrapper, slot and length (51–54 words) as the other copy
conditions. Their effects are therefore attributable to the *content* of the text,
not to its position or length.

**Scope.** This rules out copied, off-topic text. It does not rule out model-written
or on-topic non-diagnostic text.

### 4. The label follows the diagnostic text before it, whichever tile that text is about

**Primary result:** inconclusive. Copying another tile's explanation gave a net
Δ = +6.0 points [−7.3, +19.0]. By the pre-registered rule this must not be reported
as "no shift".

**Why the average is uninformative.** The donor texts were balanced by design: 50
argue HP and 50 argue SSA. The average therefore cancels two large, opposite effects.

**Exploratory analyses (pre-specified):**

- By direction of the copied text:
  - HP-direction text: −38 points [−54.7, −21.3].
  - SSA-direction text: +50 points [+38.7, +61.3].
- Donor-text-level permutation test: p ≤ 0.0002 (9 HP texts, 50 SSA texts).
- The label matched the copied text's direction in **282/300** calls†:
  - SSA-direction texts: 150/150.
  - HP-direction texts: 132/150. All 18 misses came from one text (`MHIST_bal`),
    which is labelled HP but reads SSA-like.

**The tile's own explanation behaves the same way.** When the tile's own earlier
explanation was copied in, the label matched its direction in **297/300** calls†.
Matched by direction, own and other-tile texts are followed at nearly identical
rates:

- SSA-direction: 100% vs 100%.
- HP-direction: 95% vs 88%, and the gap comes from the single anomalous text.

So these data give no evidence that text *about this tile* is special.

### 5. Describe-first shifts toward SSA, but the description often was not free of diagnostic content

**Primary result:** a shift. Δ = +10.7 points [+4.7, +17.0], Holm p = 0.0033.

**But the manipulation often failed.** 130 of 300 descriptions described the tissue
as "jagged", "saw-tooth", "tooth-like", "zig-zag" or "branching":

- They cover 81 of the 100 tiles.
- A spot-check found no false flags.
- A further 39 of the 170 unflagged descriptions mention "wavy" edges†.

**Sensitivity analysis (pre-registered, post-treatment):** dropping the flagged
responses gives Δ = +4.8 points [−2.4, +11.9] on 90 tiles, which is inconclusive.

**What can be said:** a model-written description of the image before the label
moves it toward SSA. Whether a strictly non-diagnostic description would do so is
unresolved. The filter selects on an outcome of the prompt, so it cannot show that
the leaked wording *causes* the shift either.

### 6. A direction-balanced checklist still shifts toward SSA

The generic checklist names HP and SSA criteria in balanced form. It is not about
this tile, yet it shifted labels toward SSA:

- Δ = +13.0 points [+6.7, +19.3].
- About three-quarters of the explain-first shift.

The difference from explain-first is +4.3 points [−3.0, +11.7]†, unresolved either
way.

**Checklist order (exploratory):** HP-first vs SSA-first gave +4.2 points, p = 0.51.
This test is underpowered; an order effect as large as the main effect is not
excluded.

**Mechanism: not identified.** The checklist is only approximately symmetric. The
SSA line lists three positive abnormal findings and uses "serrat" twice; the HP line
describes confinement and orderliness.

### 7. The shifts move the HP/SSA decision threshold; they do not improve accuracy†

| condition | accuracy | sensitivity for SSA | specificity for HP |
|---|---|---|---|
| classify-only | 65.3% | 65.3% | 65.3% |
| explain-then-classify | 60.0% | 77.3% | 42.7% |
| checklist | 63.0% | 76.0% | 50.0% |
| describe-first | 54.7% | 65.3% | 44.0% |

The extra SSA calls land mostly on true-HP tiles. On the 26 tiles that at most 1 of
7 pathologists called SSA, explain-first adds +25.6 points of SSA.

## Answer to Kiran's question

Does the explain-first SSA shift need the model's own diagnostic explanation, or
does any text before the label produce it?

**Neither.**

- **Not any text:** unrelated filler is equivalent to classify-only.
- **Not the model's own explanation either:** diagnostic text the model did not
  write in the call also moves the label.
  - A generic checklist moves it toward SSA.
  - Another tile's explanation moves it strongly *in whichever direction that text
    argues*, followed in 282 of 300 calls.

The label follows the diagnostic content placed before it, largely regardless of
which image that content describes. The explain-first shift toward SSA is therefore
consistent with the model's explanations tending to affirm SSA criteria, and the
label then following the explanation. It is not consistent with the explanation
surfacing image evidence the model would otherwise miss: accuracy does not improve
(finding 7).

## Caveats a reader must see

- **Scope.**
  - One model on one host (Friendli), 100 selected tiles, temperature 1.0, K = 3.
  - The tiles are 50/50 HP/SSA, and 52 of them are contested (2–5 of 7 pathologist
    SSA votes). Full MHIST is about 31% SSA.
  - Classify-only accuracy is only 65%. That is the regime where supplied text is
    most likely to dominate.
- **Host confound.** The copied texts (own and other-tile) were generated earlier on
  Cerebras. Any contrast between them and explain-then-classify, which is generated
  on Friendli, is also a host contrast. That includes the explain-then-classify vs
  own-copied difference of −13.7 points.
- **Output position vs prompt content.** The copied texts appear in the prompt as
  well as in the output before the label. This design does not separate the two.
- **Different wording as well as content.** Contrasts between explain-then-classify
  and each control also differ in instruction wording and output format.
  Classify-then-explain shows that wording alone can move the label by about 17
  points.
- **What "direction" means.** Direction for the copied texts is defined by the
  Cerebras label on the donor tile, not by reading the text. One "HP" text reads
  SSA-like. 7 of the 50 SSA-direction texts say "serrated lesion", close to the
  prompt's own "sessile serrated lesion".
- **Uncorrected rows.** The direction splits, follow rates, the accuracy table and
  the pathologist-agreement stratum are exploratory or post hoc (†). Secondary rows
  are uncorrected.
- **Monte Carlo floors.** Some p-values sit at the floor of their permutation tests
  and are reported as "≤".
- **Plan vs code, recorded rather than fixed.** `PLAN.md` says the describe-first
  term lists were matched with word boundaries. The code matched several terms
  (e.g. "jagged", "gland") as plain substrings. Every hit was checked in context,
  with no false positives, so the numbers are unaffected.

## Files

| file | contents |
|---|---|
| `analysis__openrouter__google-gemma-4-31b-it__friendli.json` / `.md` | full analyzer output, including every secondary row |
| `<condition>__openrouter__google-gemma-4-31b-it__friendli.jsonl` | raw call records |
| `PLAN.md` | pre-registration, including the simulated operating characteristics |
| `oc_power2.py`, `oc_power2_K3.txt` | the design-resolution simulation |
