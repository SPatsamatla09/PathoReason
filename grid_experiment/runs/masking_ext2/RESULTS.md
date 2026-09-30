# Masking on the remaining 450 test tiles: results

## Setup

| | |
|---|---|
| date | 2026-09-30 |
| plan | `PLAN.md`, written before any call; timestamps verified |
| verification | every number recomputed independently |
| integrity | 450/450 valid baselines; 3,624 masked calls with 0 errors and 0 duplicates; all Friendli; baselines match |
| cost | $0.78 |

With this run, **every one of the 977 test tiles has been masked or tried.**

## Primary, pre-registered: does the effect replicate on fresh tiles? **Yes.**

| tiles (Friendli, the model's own citations) | cited-more | control-more | tied | exact sign-test p | flips per call, cited vs control |
|---|---|---|---|---|---|
| **ext2: 427 never-masked tiles** | **130** | **79** | 218 | **5.1 × 10⁻⁴** | 28.6% vs 21.8% |
| extension: 319 tiles (discovery) | 102 | 49 | 168 | 1.9 × 10⁻⁵ | 32.3% vs 21.8% |

- ext2 vs extension: Fisher p = 0.32; they don't differ.
- Cited share on ext2: 0.62 (95% CI 0.55–0.69). Power at the extension's share was 0.999.
- The primary survives project-wide Bonferroni correction (×31 gives p = 0.016).
- By occlusion:

  | occlusion | cited-more | control-more |
  |---|---|---|
  | mean fill | 70 | 41 |
  | blur | 76 | 42 |
  | black | 85 | 61 (p = 0.057) |

- Mean-net permutation p = 0.0007.

**Round 3's conclusion that "the effect is significant on one tile set only" is withdrawn.**

## Secondary: tissue-type-matched control

Tissue and epithelial area matched within 5 pp. Feasible on 354 of 427 tiles:

- **Result:** 104 vs 58 (p = 3.8 × 10⁻⁴); flips 29.1% cited vs 20.6% control.
- **Not independent of the primary.** It reuses the same cited-arm calls: per-tile net
  correlation is 0.74, and the area-matched arm on the same 354 tiles gives 112/63.
- **Type matching did not change the control flip rate** (20.6% vs 21.0%).

## Full test set (descriptive)

| tile set | host, masks | cited-more | control-more | p |
|---|---|---|---|---|
| legacy 185 | Cerebras, its own citations | 54 | 47 | 0.55 |
| legacy 185 | Friendli, Cerebras-derived masks (crossover) | 45 | 53 | 0.48 |
| legacy 184 | Friendli, its own citations | 55 | 41 | 0.18 |
| extension 319 | Friendli, own | 102 | 49 | 1.9 × 10⁻⁵ |
| ext2 427 | Friendli, own | 130 | 79 | 5.1 × 10⁻⁴ |

**Pooled Friendli own-citation sets (930 tiles):** 287 vs 169, with heterogeneity
across the three sets p = 0.25 (Freeman–Halton exact test 0.26).

**This pooled p (3.6 × 10⁻⁸) is not confirmatory, and must not be a headline:**

- The plan labelled the pool descriptive.
- About two-thirds of it had already been seen.
- Its membership was chosen after round 3 had seen which sets were null.

**The confirmatory evidence is the ext2 p = 5 × 10⁻⁴.**

## The older 185-tile results: underpowered plus chance

**Neither "it's the host" nor "whose citations were masked" is supported.**

- **Same tiles, each host using its own citations:** Cerebras 54/47 vs Friendli 55/41.
  Fisher p = 0.67; the paired per-tile comparison is 59 vs 58.
- **Same tiles and host, own vs Cerebras-derived masks:** 55/41 vs 45/53. Fisher
  p = 0.12; paired p = 0.095.
- **Power:** the legacy sets had only about 0.5–0.7 power at the observed effect size.

The crossover (45/53) is the one clear outlier: against the pool, P(X ≤ 45) = 4.5 × 10⁻⁴.

## Exploratory, not a finding (needs its own pre-registration)

**In ext2 the whole excess is in tiles the model called HP:**

| ext2 tiles | cited-more | control-more | flips, cited vs control |
|---|---|---|---|
| called HP | 100 | 40 | 25.7% vs 14.1% |
| called SSA | 30 | 39 | 35.1% vs 39.2% |

The interaction has p = 1.3 × 10⁻⁴.

**Direction of the shift.** On a directional measure (does masking the cited cells yield
more SSA answers than masking the control?), the shift toward SSA appears in every set:

| set | cited more SSA | control more SSA | p |
|---|---|---|---|
| Cerebras legacy | 61 | 40 | 0.046 |
| crossover | 60 | 38 | 0.033 |
| Friendli-own legacy | 58 | 38 | 0.052 |
| ext2 | 139 | 70 | 2 × 10⁻⁶ |
| extension | 79 | 72 | 0.63 |

**Why this matters.** The pre-registered flip measure counts changes in both directions,
so it cancels a push toward SSA where many tiles were called SSA. Cerebras called 41% of
its tiles SSA, against 31% in ext2. This could reconcile the legacy nulls. It is a
hypothesis only.

## Effect size and meaning

- **Modest:** about 7 extra label flips per 100 occlusions; 30% of tiles cited-more,
  19% control-more, 51% tied.
- **Most of the label's sensitivity lies outside the 3 masked cells.**
  - Masking the control alone flips about 9 points above the resampling floor (about
    13%).
  - Masking the cited cells flips about 16 points above it.
- **No diagnostic meaning:**
  - Baseline accuracy is 59.6%, against 68.2% for always-HP on these tiles; balanced
    accuracy is 52.8%.
  - Masking lowers accuracy equally in both arms (55.8% vs 55.1%).
  - Confidence is unchanged (0.8 on 95% of calls).

## Suggested wording

> On the OpenRouter→Friendli deployment of gemma-4-31b-it, occluding the three most
> tissue-rich cells the model itself cited changed its HP/SSA label modestly more often
> than occluding disjoint tissue-area-matched cells. The test was pre-registered on 427
> never-masked test tiles and replicated the earlier 319-tile result (130 vs 79 tiles,
> exact sign test p = 5 × 10⁻⁴; 28.6% vs 21.8% of masked calls flipped; Fisher p = 0.32
> against the earlier set). The effect held against controls also matched on
> stain-estimated epithelial area where feasible (104 vs 58). The earlier Cerebras
> sweeps on 185 tiles did not show it; on the same tiles the two deployments were
> indistinguishable, and those sets were underpowered. The effect is small, has no
> diagnostic value (the model classifies below the majority-class rate), and does not
> show that the named features drive the diagnosis.

## Overclaims to avoid

- the 930-tile pooled p as a headline or confirmatory result
- "independent replication"; say "disjoint, never-masked tile set, same deployment and
  pipeline"
- "top-3 cited cells"; say "the three most tissue-rich cited cells"
- "controls are uncited cells". Area-matched controls hold on average 1.29 cells cited
  elsewhere.
- the HP-called / SSA-shift pattern presented as a finding
- a deployment or citation-source explanation of the legacy nulls
- "faithful explanations", or that the features drive the label
- any clinical meaning
- a model-level claim (gemma in general)
