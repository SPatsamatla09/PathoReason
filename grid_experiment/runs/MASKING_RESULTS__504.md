# Grid-masking extension: results

> **Update 2026-09-30 (round 3, verified).** The host explanation below has been
> tested and refuted. On the same 185 legacy tiles with the same masks, Friendli gives
> 45 vs 53 (p = 0.48), agreeing with Cerebras's 54 vs 47 (Fisher p = 0.32). The
> difference reflects the tile set and whose citations defined the masks, not the
> deployment. The Friendli extension effect also survives tissue-type-matched controls
> (99 vs 37, p = 1.0 × 10⁻⁷). See `runs/ROUND3_RESULTS.md`, which supersedes the
> "Not ruled out: a genuine deployment difference" item and the "decisive next test"
> section below.

## Setup

| | |
|---|---|
| model | gemma-4-31b, prompt cte_p1 (classify, then explain with grid-cell citations) |
| design | the one used in every earlier sweep, unchanged |
| run date | 2026-09-29/30 |
| verification | adversarial, 2026-09-30; every number here was recomputed from the raw records by independent code |

**Design.** For each tile, the model's own baseline answer is taken, with its cited cells.

- **Cited arm:** the 3 highest-tissue cited cells are occluded.
- **Control arm:** 3 other cells matched for tissue area are occluded.
- **Occlusions:** each arm is run with mean fill, blur and black, giving 6 calls per tile.
- **Flip:** the masked label differs from that tile's baseline label.
- **Unit of analysis:** the tile. Net = the sum over the three occlusions of (cited flip − control flip).
- **Test:** exact sign test over tiles with net ≠ 0.

**The sweeps:**

| sweep | host | tiles | sample |
|---|---|---|---|
| B | Cerebras (archived) | 100 | label-balanced |
| C | Cerebras (archived) | 100, of which 85 unique (15 are shared verbatim with B and counted once) | design-matched |
| **ext (new)** | OpenRouter → Friendli | 319 | never masked before, proportional by label × band |

For ext, 340 tiles were drawn; 21 had no feasible control. Citations and baselines were
regenerated on the new host. That gives **504 unique tiles**.

**Integrity** (all verified):

- 0 errors, 0 invalid labels, 0 duplicate calls.
- All 3,114 masked images regenerate pixel-exactly from the recorded cells, with no
  changed pixels outside them.
- Each baseline and masked call on the new host came from the same provider.
- The 15 tiles shared by B and C are byte-identical.

## Results

| | tiles | cited-more | control-more | tied | exact sign-test p | per-call flip rate, cited vs control |
|---|---|---|---|---|---|---|
| **Friendli (ext)** | 319 | **102** | **49** | 168 | **1.9 × 10⁻⁵** | 32.3% vs 21.8% |
| **Cerebras (B + C)** | 185 | 54 | 47 | 84 | 0.55 | 25.0% vs 23.4% |
| — sweep B | 100 | 23 | 29 | 48 | 0.49 | 22.7% vs 27.3% |
| — sweep C (85 unique tiles) | 85 | 31 | 18 | 36 | 0.085 | 27.8% vs 18.8% |
| pooled, **not reportable** | 504 | 156 | 96 | 252 | (0.0002) | 29.6% vs 22.4% |

### The pooled p is not a result

The rule, set before the extension ran, was: if the hosts disagree at p < .05, report
per host, not the pooled p. They do.

- Fisher's exact test, host × (cited-more, control-more): p = 0.025.
- Mean-net permutation: p = 0.020.
- Stratified by label × band: p = 0.020.

The pooled signal comes entirely from the extension.

### "Host" is confounded with sweep

**Cerebras sweep C points the same way as Friendli and is statistically
indistinguishable from it** (31/18 vs 102/49, Fisher p = 0.60). Sweep B is the
outlier (B vs ext p = 0.005; B vs C p = 0.07).

Across the three sweeps as independent studies (random effects, DerSimonian–Laird):

- cited share 0.59 [0.45, 0.72], p = 0.21
- I² = 0.77

**So there is no replicated model-level effect.** There is a strong effect on one
deployment.

## The Friendli result is robust within that deployment

**Every occlusion alone** (tiles where only the cited mask flipped vs only the control):

| occlusion | cited-only | control-only | p |
|---|---|---|---|
| mean fill | 61 | 26 | 0.0002 |
| blur | 63 | 34 | 0.004 |
| black | 68 | 32 | 0.0004 |

**Both baseline labels:** HP 63 vs 33 (p = 0.003); SSA 39 vs 16 (p = 0.003).

**Split halves:** 56/23 and 46/26. The effect is consistent across tile-letter
blocks and run order.

**Trimming the most extreme tiles:** p = 0.005 without the top 20, and p = 0.045
without the top 30.

**Multiplicity.** This is roughly the 17th masking contrast run in the project.
Bonferroni ×17 gives p = 3.3 × 10⁻⁴.

**Explanations ruled out:**

- **Per-tile control seed.** The random draw only decided the control on 10 of 319
  tiles. Without those tiles: 99/48, p = 3 × 10⁻⁵.
- **Regenerated citations.** On the 204 tiles where Cerebras had also cited all three
  masked cells: 66/30, p = 3 × 10⁻⁴.
- **Position, tissue-gap and darkness confounds.**
  - The geometry is the same in every sweep.
  - The effect persists in balanced subsets:
    - control as central as the cited set: 32/9
    - cited tissue ≤ control tissue: 23/7
    - control darker than the cited cells: 31/14
  - Adjusting for 10 structural covariates barely moves the host difference.
- **Noise.** The control-arm flip rate is the same on both hosts (21.8% vs 23.4%). The
  host difference is about 82% in the cited arm. Unmasked resampling noise is not
  lower on Friendli (13.5% replicate disagreement).

**Not ruled out:**

- A genuine deployment difference. The two hosts agree on only 79% of baseline
  labels, and Friendli leans toward HP (McNemar p = 0.009).
- Chance, at about the 1–2% level.

## What it does and does not support

**Supported, on the Friendli deployment only:** occluding the top-3-tissue cells the
model cited changes its label more often than occluding three other tissue-area-matched
cells.

**Not supported:**

- that the explanation text or the named features are faithful
- that the model reasons from the cited features
- that the effect holds for gemma-4-31b in general, or that it replicated
- any confidence effect: confidence is 0.8 on 1,822 of 1,914 calls

**The controls are not "uncited cells".** They are disjoint from the 3 masked cited
cells only.

- On average 1.16 of the 3 control cells were cited elsewhere in the model's answer on
  ext. It was 1.41 on B and 1.49 on C.
- Only 94 of the 319 ext tiles have fully uncited controls. The effect holds there:
  36/11, p = 3.5 × 10⁻⁴. Legacy is 7/7.

**Tissue type is not matched, only tissue area.** Cited cells are darker in 217 of
319 tiles and more central.

**Much of the label also depends on non-cited regions.** Occluding the control alone
flipped 21.8% of calls, against a resampling floor of about 13.5%.

## Consequence for the paper

The earlier framing, that citations are "causally inert", must be withdrawn. The
Friendli deployment contradicts it. It was also an overreading of an underpowered
null on Cerebras:

- The legacy test had only 51% power at a true cited share of 0.60.
- The legacy 95% CI for the cited share reaches 0.63.

Places in `paper/pathoreason_revised.md` that repeat it: lines 29, 39, 43, 294, 298
and 318, plus "tissue it never mentioned" on line 39.

Also flagged, but not edited:

- **§4.3 line 163.** "40 versus 54, p = 0.18" is the pseudo-replicated pair-level
  count. The tile-level figure for B is 23 vs 29, p = 0.49.
- **Sweep B's sampling.** B is label-balanced, not proportional.

## Suggested wording

> On gemma-4-31b-it served through OpenRouter by Friendli (319 tiles not previously
> masked; citations and baselines regenerated on that deployment), occluding the three
> highest-tissue cells among those the model cited changed its HP/SSA label more often
> than occluding three other cells matched for tissue area: 102 tiles vs 49, 168 tied
> (exact two-sided sign test p = 1.9 × 10⁻⁵; 32.3% vs 21.8% of masked calls flipped).
> On the archived Cerebras deployment the same design showed no detectable difference
> (54 vs 47 of 185 tiles, p = 0.55), and its two sweeps pointed in opposite directions
> (23 vs 29; 34 vs 22). Because the deployments differ (Fisher p = 0.025), we report
> them separately, as specified before the extension was run, and give no pooled p.
> On the Friendli deployment this shows that the cited regions influence the label
> more than tissue-area-matched other regions, not that the stated features drive the
> diagnosis: controls were matched on tissue area rather than tissue type, about 39% of
> control cells had also been cited, and occluding the control cells alone flipped
> 21.8% of labels.

## Overclaims to avoid

- the pooled 504-tile p = 0.0002 as a result
- "citations are causally inert"
- "explanations are faithful" or "the model uses serration/crypt features"
- any model-level or "replicated" claim
- "controls are uncited cells" or "tissue it never mentioned"
- "the host difference is quantisation" (untested)
- pair-level counts (192 vs 92; 284 vs 175) as the test; they are pseudo-replicated
- p rounded to 0.0 from JSON
- the post-hoc finding that the host disagreement sits in SSA-baseline tiles (18/25
  vs 39/16, uncorrected p = 0.007), presented as a finding
- any clinical claim. Baseline accuracy on the ext tiles is 55%, below the 63.2% that
  "always HP" would score.

## The decisive next test (not run)

**Re-run the 185 legacy tiles on Friendli, with the same masks, on the same host.**

- About 1,110 calls, about $0.25, about 20 minutes with parallel workers.
- If Friendli shows cited > control on those tiles too, the difference is the
  deployment. If not, it is the tile sample (sweep B).
- The reverse test, the ext tiles on Cerebras, is impossible now that the model is
  archived.

## Files

- `runs/masking_pooled_analysis.json`: all sweeps, per host, heterogeneity.
- `runs/masking_pooled_legacy_analysis.json`: Cerebras only.
- `runs/masking_ext__openrouter__google-gemma-4-31b-it__friendli_k3.jsonl` and
  `…_k3_analysis.json`: raw extension calls and the per-sweep analysis.
- `runs/ext_baseline_check__openrouter__google-gemma-4-31b-it__friendli.json`: the
  baseline agreement gate.
- `masked/ext__…_k3/<tile>/manifest.json`: the cells masked per tile (images local
  only).
