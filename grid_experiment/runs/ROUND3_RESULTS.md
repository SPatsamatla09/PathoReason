# Round 3 results: host investigation, tissue-type controls, competence pilot

## Setup

| | |
|---|---|
| date | 2026-09-30 |
| model | gemma-4-31b, via OpenRouter pinned to Friendli, unless stated |
| verification | every number recomputed from raw records by independent code; masks, pins, sampling fields and pixels audited; adversarial interpretation review |

Sources:

- `runs/HOST_INVESTIGATION.json` (parts A–D)
- `runs/masking_typematched_analysis.json`
- `runs/PILOT_RESULTS.json`

Cost of this round: about $1.33 of OpenRouter credit ($2.85 left).

## 1. Why the masking effect appeared on one host and not the other

**Short answer: the serving host does not explain it.** With tiles and masks held
fixed, the two deployments agree. The earlier contrast compared different tiles with
different masks.

### Main result

| same 185 legacy tiles | cited-more | control-more | tied | sign-test p |
|---|---|---|---|---|
| Cerebras, original masks (original run) | 54 | 47 | 84 | 0.55 |
| **Friendli, the same original masks (crossover)** | **45** | **53** | 87 | **0.48** |
| Friendli, masks rebuilt from Friendli's own citations (184 tiles) | 55 | 41 | 88 | 0.18 |
| *(Friendli, the 319 extension tiles, own citations)* | *102* | *49* | *168* | *1.9 × 10⁻⁵* |

Pairwise Fisher tests:

- **Crossover vs Cerebras:** p = 0.32. On identical tiles and masks, the hosts do not
  differ.
- **Crossover vs extension:** p = 0.0009.
- **Own-citation vs crossover:** p = 0.12.
- **Own-citation vs extension:** p = 0.11.

### Reading, as the pre-stated rule sets it

The rule was: *if Friendli shows no effect on the legacy tiles, the difference is the
tile sample.* It pointed to the tile set. The own-citation run adds a second factor:
whose citations define the masks.

- **Citation source.** On the 78 tiles where the masks are identical in both runs, both
  runs are null. The gain on the own-citation run comes from the 105 tiles whose cited
  cells changed (37/20; post hoc).
- **Neither factor is significant on its own**, and noise is large:
  - Per-tile net correlates only 0.08 across hosts on identical masks.
  - Re-running the identical cited images changes 18% of labels.

**The effect is significant on one tile set only**, the 319 extension tiles.

### Sweep-B outlier

The sweep-B "outlier" follows sweep B's Cerebras-built masks, not the host (post hoc):

| masks, host | sweep B | sweep C-unique |
|---|---|---|
| Cerebras masks, Cerebras | 23/29 | 31/18 |
| Cerebras masks, Friendli | 20/35 | 25/18 |
| Friendli's own masks | 33/27 | 22/14 |

### Do the deployments differ at all? Yes, but it isn't decoding and it isn't formatting

**What is the same.** Both hosts process identical inputs:

- 825 prompt tokens on every call, and 713 and 830 on the other prompts
- no system prompt
- the same image-first token layout
- identical parsing

The request bodies differ only in the model id and OpenRouter's routing field.

**What differs.** Their outputs differ beyond sampling noise:

- **Code fence.** Cerebras opened a code fence in 99% of answers at temperature 1. If
  Cerebras were just another draw from Friendli's distribution, 76 of 100 would be
  expected (P = 1.4 × 10⁻¹³).
- **Label disagreement.** Cross-host label disagreement is 77/300, against 57.5
  expected from within-host noise (P = 4 × 10⁻⁵).
- **Tile MHIST_ddn** is SSA in 11/11 Cerebras calls and HP in 9/9 Friendli calls,
  under every sampling setting.

**Sampling settings do not reproduce Cerebras on Friendli.** Code-fence rate:

| setting | code-fence rate |
|---|---|
| host default | 68% |
| Gemma's top_p 0.95 / top_k 64 | 74% (p = 0.19 vs default) |
| temperature 0.5 | 78% |
| temperature 0 | 75% |

Sampling settings reshape probabilities, but cannot reverse which answer a model
prefers. Cerebras was sampling normally: it produced no duplicate outputs and had its
own label flips.

**Defensible statement.** On identical inputs the two deployments computed different
next-token distributions. Whether the cause is different weights, numeric precision,
image preprocessing, or an undisclosed logit processor cannot be determined, and can no
longer be tested now that Cerebras is archived.

**This deployment difference does not account for the masking discrepancy** (see the
crossover result above).

**Caveats:**

- Friendli's temperature-0 determinism was not tested: each tile was called once.
- Whether Friendli actually applied top_p/top_k cannot be confirmed from the records.

## 2. Tissue-type-matched controls (Kiran's control-region concern)

**Design.** On the Friendli extension tiles, each control is 3 cells that:

- are disjoint from the cited cells
- are matched on tissue area and on stain-estimated **epithelial** area, within 5
  percentage points
- use as few cells the model cited anywhere as possible

Feasible on **262 of 319** tiles. The cited arm was re-run alongside, on the same
images.

**Why it mattered.** The old area-matched controls held less epithelium than the cited
cells in 204/319 tiles, and differed by more than 5 pp in 170/319.

| same 262 tiles | cited-more | control-more | tied | p | flip rate, cited vs control |
|---|---|---|---|---|---|
| **cited vs tissue-type-matched** | **99** | **37** | 126 | **1.0 × 10⁻⁷** | 35.2% vs 21.8% |
| cited vs area-matched (original run) | 82 | 40 | 140 | 1.8 × 10⁻⁴ | 32.6% vs 21.9% |
| type-matched, control holds **no** model-cited cell (169 tiles) | 62 | 23 | 84 | 2.8 × 10⁻⁵ | 37.9% vs 24.5% |

**Reading.** On these tiles the cited-cell effect is **not explained by the cited cells
being more epithelial**, nor by controls that contain cited cells. The type-matched
controls flip labels as often as the area-matched ones (21.8% vs 21.9%).

**The effect survives type matching. It is not strengthened by it.** The rise from
82/40 to 99/37 is mostly from re-running the cited arm:

- The fresh cited arm against the old area-matched controls gives 93/34.
- Test–retest agreement of the cited arm is 82%.

**Limits:**

- Extension (discovery) tiles only; not run on the legacy tiles.
- "Type" is a hematoxylin-stain proxy, not histological structure.
- 57 tiles (18%) had no feasible control.
- The control rule and the uncited subgroup depend on the model's own output.
- This is a robustness check, not an independent replication.

## 3. Competence pilot: is there a "competent VLM"?

**Design.** 100 test tiles, drawn in proportion to label × agreement band (63 HP / 37
SSA, so "always HP" = 63%). Same gridded classify-then-explain prompt. One call per
tile. Each model pinned to its first-party provider, with thinking off.

**Bar, set in advance:** a Wilson 95% lower bound above 63.2%, i.e. at least 73/100
correct.

| model | correct | 95% CI | balanced accuracy | SSA calls | cost |
|---|---|---|---|---|---|
| gemma-4-31b (the studied model) | 61 | 51–70% | 56.8% | 32 | $0.02 |
| Gemini 2.5 Flash | 57 | 47–66% | 59.2% | 56 | $0.11 |
| GPT-4.1 | 39 | 30–49% | 51.6% | 98 | $0.29 |
| Qwen3-VL-235B | 37 | 28–47% | 50.0% | 100 | $0.04 |

**Reading.** Under this prompt and input format, **no candidate, including the model
studied here, was shown to discriminate HP from SSA.** Label-vs-truth Fisher p ≥ 0.096
for all four, and confidence AUROC is 0.52–0.58. GPT-4.1 and Qwen are degenerate.

**This does *not* show that** "no VLM can do HP/SSA", and it gives no ranking of the
models.

**The all-SSA default may be caused by the prompt.** SSA is expanded as "sessile
serrated…", the prompt offers "serration" as a feature term, and all four models cite
serration on every tile. That is untested.

**Cheap diagnostic (about $1–1.5, not run).** GPT-4.1 and Qwen on the same 100 tiles:

- a 2×2 of grid on/off × full prompt vs a minimal "HP or SSA" prompt
- plus a neutral-label arm (class A/B)
- plus GPT-4.1 log-probabilities, for a threshold-free AUROC

The rule, fixed in advance: if the SSA rate falls well below 90% and AUROC exceeds
about 0.65 without the grid and vocabulary, then the pilot measured the prompt, not the
models.

## 3b. Prompt diagnostic (follow-up to the pilot)

Pre-registered, about $0.61, verified. Full write-up: `runs/prompt_diagnostic/RESULTS.md`.

**The grid and the feature vocabulary are not needed for the SSA default.** With a
clean tile and a one-line "HP or SSA?" prompt, the SSA rate was:

- GPT-4.1: 100%
- Qwen3-VL-235B: 100%
- gemma-4-31b: 98%
- Gemini 2.5 Flash: 89%

Relabelling the classes as neutral letters, with the order reversed, left GPT-4.1 at
97% and Qwen at 100%.

**No model reached the bar in any of the 14 model × prompt cells** (best 61/100).
GPT-4.1's log-probability AUROC was 0.60–0.61, with CIs reaching about 0.5.

**Controls (added later, also pre-registered):**

- Reversing the letters and order leaves the SSA rate at 97–100%, so the letter and
  position are not the driver.
- With no image at all, Qwen still answers SSA 100/100, so its default needs no image.
- GPT-4.1 mostly refuses without an image, but its answer tokens still favour SSA; the
  image makes that stronger, not correct.

**Not excluded:** the shared "sessile serrated" wording.

**Exploratory:**

- Removing only the grid moved GPT-4.1 from 98% to 73% SSA. That is mostly a threshold
  shift, and still 58/100.
- GPT-4.1 and Qwen cited grid cells on 100/100 clean, gridless tiles.

## 4. Multiplicity

The project has now run about 27 tile-level cited-vs-control masking tests.

**Surviving Bonferroni ×27:**

- the extension result (1.9 × 10⁻⁵, ×27 = 5.2 × 10⁻⁴)
- its area-matched re-analysis on the 262 tiles
- the type-matched test
- the type-matched test with fully uncited controls

**All survivors come from one tile set and one set of baselines,** so they are not
independent confirmations. No test on the 185 legacy tiles is significant on either
deployment, even uncorrected.

## Suggested wording (for the write-up)

**Host.**

> With tiles and masks held fixed, the two deployments agreed: the 185 Cerebras masking
> tiles re-run on Friendli with their original masks gave 45 cited-more vs 53
> control-more tiles (p = 0.48; Cerebras 54 vs 47; Fisher p = 0.32). The earlier
> difference between deployments therefore reflects the tile sets and whose citations
> defined the masks, not the serving deployment. With masks built from Friendli's own
> citations, the legacy tiles gave 55 vs 41 (p = 0.18): the same direction as the 319
> extension tiles (102 vs 49), but weaker and not significantly different (p = 0.11).
> The effect is significant on one tile set only.

**Type-matched.**

> On 262 of the 319 extension tiles where a control could be matched to the cited cells
> on both tissue area and stain-estimated epithelial area (within 5 points), occluding
> the cited cells changed the label more often than occluding the type-matched control
> (99 vs 37 tiles, p = 1.0 × 10⁻⁷; 35.2% vs 21.8% of calls), including 62 vs 23 on the
> 169 tiles whose control held no cell the model had cited. Type-matched controls
> changed the label as often as area-matched ones (21.8% vs 21.9%), so on these tiles
> the effect is not explained by the cited cells being more epithelial. This check was
> run only on the extension tiles, where the effect was first found.

**Pilot.**

> In a 100-tile pilot (63 HP / 37 SSA) using the same gridded classify-then-explain
> prompt, no model met the pre-set competence bar (95% lower bound above 63.2%):
> accuracy was 61% for gemma-4-31b (balanced 57%), 57% for Gemini 2.5 Flash (balanced
> 59%), 39% for GPT-4.1 and 37% for Qwen3-VL-235B, the last two answering SSA on 98% and
> 100% of tiles. Under this prompt and input format, no candidate, including the model
> studied here, was shown to discriminate HP from SSA; whether the prompt or grid causes
> the SSA default is untested.

## Overclaims to avoid

**Host and masking:**

- "the effect replicated on Friendli but not Cerebras", or any host or "Cerebras was
  faulty" explanation
- quantisation or sampling named as the cause
- "different model versions"; say "different next-token distributions"
- "replicated across tile sets" or "model-level"
- the pooled 504-tile p, or the post-hoc Friendli pool (157/90, p = 2.4 × 10⁻⁵), used
  as a headline
- post-hoc splits presented as findings: B/C, the 105 changed-citation tiles, the
  baseline-agreement tiles

**Type-matched controls:**

- "type-matching strengthened the effect"
- "tissue type controlled" meaning histology
- the type-matched run described as an independent replication
- "controls are uncited cells", except the 169-tile subset

**Pilot:**

- "no VLM can do HP/SSA"
- "GPT-4.1/Qwen cannot recognise SSA"
- "Gemini is best"
- "gemma is competent"
- accuracy against 63.2% as the only metric; report balanced accuracy too

## Note on scope

Kiran's original round-2 constraints were: no second VLM, and no tissue-type-matched
controls. On 2026-09-30 the user directed both the tissue-type controls and a
low-credit competence pilot, following Kiran's full feedback. The full second-VLM
pipeline has **not** been run.
