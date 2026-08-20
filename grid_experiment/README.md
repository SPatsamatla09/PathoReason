# MHIST grid-localisation probe — stimulus set

Materials for testing whether a vision-language model's explanations are
faithful: can it name *which* grid cells contain the features it claims to be
reasoning from?

## Setup in this repository

The scripts expect to sit **inside the MHIST dataset directory**: they resolve
`../images/` and `../annotations.csv` relative to this folder. MHIST is gated by
a data-use agreement (request access at <https://bmirds.github.io/MHIST/>), so
no tile pixels are committed here — `clean/`, `gridded/`, and the `masked/`
PNGs are gitignored and regenerate deterministically (seed `20260806`) once the
dataset is in place:

```bash
# from the repo root, after obtaining MHIST
ln -s /path/to/mhist/images images
ln -s /path/to/mhist/annotations.csv annotations.csv
cd grid_experiment
python3 build_coverage.py        # per-cell coverage for all 3,152 tiles
python3 select_and_render.py     # re-renders the same 20 gridded/clean pairs
python3 mask.py --from-run runs/cte_p1.jsonl --subset 3 --out masked/cte_p1_k3
```

What IS committed: all code, the frozen prompt spec and its nine rendered
prompts, the 20-tile selection manifests, per-cell coverage JSONs for the 20
evaluated tiles (so `score.py` re-runs without the dataset), all model
responses with raw strings intact (`runs/*.jsonl`), scored reports, the
condition comparison, and the per-tile ablation manifests. Model calls need
`CEREBRAS_API_KEY` in the environment; only `requests`, `numpy`, `pillow`, and
`pyyaml` beyond the Python stdlib.

## Grid convention

4x4 grid over each 224x224 tile, cells 56x56.
**Row letter A-D runs top to bottom; column number 1-4 runs left to right.**
`A1` is top-left, `A4` top-right, `D1` bottom-left, `D4` bottom-right.

The labels are burned into the pixels, so a model does not have to infer the
convention — it can read it. That matters for interpreting results: a
systematically transposed answer (naming `B3` where `C2` was meant) is a
detectable failure mode rather than an ambiguity in the prompt. If you see
transposition, the model is pattern-matching a spreadsheet convention
(letter = column) instead of reading the image.

## Overlay

- 1px lines in pure green `(0,255,0)` on the three internal boundaries per axis.
  Green is the one hue absent from the H&E gamut, so it never reads as stain.
- No outer border — the tile edge already bounds the outer cells, and an outer
  rectangle crowded the row-A and column-1 labels.
- Labels in Menlo Bold 12px, green with a 1px black stroke, inset 3x2px from
  each cell's top-left corner. Menlo keeps its counters open at this size where
  Arial Bold fills them and `B1` degrades into a blob; the stroke guarantees
  contrast against both bare slide glass and dark hematoxylin.
- Cost: the overlay changes **7.6% of tile pixels** and occludes **~7.3% of
  tissue pixels** (range 5.8–8.3% across the 20 selected tiles). Lines account
  for 2.7pp, labels for the rest.

## Files

| Path | Contents |
|---|---|
| `gridded/` | `*_grid.png` — the overlaid tile, this is what goes to the model |
| `clean/` | unmodified original, byte-identical to `../images/` — for masking |
| `coverage/` | per-cell tissue coverage JSON, **all 3152 tiles** |
| `coverage_index.csv` | one row per tile, flattened — drives selection |
| `selection_manifest.json` | the 20 chosen tiles with full provenance |
| `selection.csv` | the same 20, flat |
| `prompts/pathoreason.yaml` | prompt spec — 3 conditions x 3 paraphrases |
| `prompts/rendered/` | the nine assembled prompts as sent, plus `index.json` |

## Prompts

`prompts/pathoreason.yaml` is the source of truth; `render_prompts.py`
materialises the nine full prompt strings and checks the invariants the design
depends on. The conditions are `classify_only`, `classify_then_explain`, and
`explain_then_classify`.

The condition is carried by two things that must always agree: the instruction
wording *and* the order of keys in the required JSON. Decoding is
autoregressive, so a prompt that says "describe first, then decide" while
requiring `{"label", ..., "evidence"}` emits the label before any evidence
exists — an inert condition, not a weak one. `explain_then_classify` therefore
declares `evidence` first, and `render_prompts.py --check` fails the build if
that ever stops being true.

If you call the model with a structured-output or JSON-schema mode, confirm it
preserves declared key order. Some implementations normalise or alphabetise
keys, which silently collapses `explain_then_classify` into
`classify_then_explain`. Prefer free-form JSON with a parse-and-validate step,
and log the raw response string so key order stays auditable.

No prompt hints that any part of a tile may be blank — the empty-cell control
only works if the model is not told it exists. The renderer greps for that class
of leak on every run.

`gridded/` and `clean/` hold only the 20 selected tiles. Rendering all 3152 is
~680MB of PNGs that nothing downstream reads; `python3 select_and_render.py --all`
produces the full set if you need it.

## Tissue detection

A pixel is tissue when `saturation > 0.08` **or** `value < 0.80` (HSV, 0-1).
Background here is bare slide glass: bright and nearly colourless. The `or` term
catches pale washed-out tissue that a plain grayscale threshold misses. Each
cell also stores `tissue_pct_gray_rule` (grayscale < 220) so you can re-threshold
without recomputing; the two rules agree within ~2pp.

A cell is **empty** — a negative control — below 5% tissue. In the selected 20
the split is clean: empty cells top out at 4.88% coverage (mean 0.52%) and
non-empty cells bottom out at 5.45%, so nothing sits ambiguously on the
threshold.

Caveat: coverage is a pixel-level measure, so a large glandular lumen *inside*
tissue also reads as low coverage. At 56x56 an enclosed lumen almost never fills
a whole cell, and every empty cell in this selection touches the tile border —
but check `touches_tile_border` before treating a cell as off-slide.

## The 20 tiles

Drawn from the **test** split only, seed `20260806`. Balanced 10 HP / 10 SSA and
stratified on annotator agreement:

| Band | Votes | Tiles |
|---|---|---|
| unanimous | 0 or 7 | 6 |
| strong | 1-2 or 5-6 | 7 |
| borderline | 3-4 | 7 |

12 of 20 carry >=2 empty cells (6 of those carry >=4), against a 44% base rate
in the test split — deliberately oversampled so the negative-control arm has
enough trials. The other 8 are dense tiles with zero empty cells.

## Results: cte_p1 on gemma-4-31b (Cerebras)

26 calls — 20 tiles once, 3 of them twice more for stability. Temperature 1.0.
Zero API errors, zero parse failures, zero schema violations; all 26 emitted
`label, confidence, evidence` in the prompted order.

| Question | Result |
|---|---|
| 1. valid cells only | **yes** — 268 citations, 0 outside A1-D4 |
| 2. varies by image | **weakly** — between-image Jaccard 0.393 vs 0.319 random; centre bias dominates |
| 3. stable across reps | **moderate** — within-image Jaccard 0.696 vs 0.393 between, p<1e-4 |
| 4. cites empty cells | **almost never** — 1/201 = 0.50% vs 11.32% centre-bias-matched null, p<1e-4 |
| 5. transposition | **no** — cited cells average 82.9% tissue, their transposes 74.1% |
| classification | **at chance** — 10/20, exact 95% CI [27.2%, 72.8%], mean confidence 0.802 |

The headline is a dissociation. Empty-cell avoidance is real and survives the
strongest null available: sampling cells from the model's *own* pooled
preference — which reproduces its centre bias exactly — predicts 11.32% empty
citations against 0.50% observed. So the model genuinely tracks where tissue
sits in each specific image. But its cell choices are still dominated by a
content-independent centre prior: citation frequency correlates with centrality
at r=+0.857 and with tissue availability at only r=+0.648, and among the five
cells that contain tissue in all 20 tiles the counts run from 17/20 (B3) down to
5/20 (D4). Identical availability, threefold difference in citation. It also
names 7.6 of 16 cells per response, nearly half the grid.

Meanwhile classification carries no signal on this sample, so the cited evidence
is justifying a coin flip. Treat "explanation faithfulness" cautiously here —
there is no reliable decision for an explanation to be faithful *to*. The
finding is that coarse perceptual competence (tissue vs glass) coexists with
zero diagnostic skill and 0.80 stated confidence.

Caveats: n=20 makes the accuracy CI very wide.

## All three conditions

78 calls total, 26 per condition, identical settings. Zero errors, zero parse
failures, zero schema violations throughout. The manipulation was delivered:
all 26 `etc_p1` responses emitted `evidence -> label -> confidence`, all 26
`cte_p1` and `co_p1` emitted `label -> confidence -> evidence`.

| | co_p1 | cte_p1 | etc_p1 |
|---|---|---|---|
| accuracy | 50% | 50% | 50% |
| mean confidence | 0.825 | 0.802 | 0.860 |
| predicted SSA | 14/20 | 12/20 | 18/20 |
| cells cited / response | — | 7.6 | 9.5 |
| evidence items | — | 2.6 | 3.0 |
| empty-cell citations | — | 1/201 | 0/256 |
| within-image Jaccard | — | 0.696 | 0.904 |

### Is it different evidence, or the same evidence with reordered keys?

Cross-condition similarity is only interpretable against two references from the
same data, so all three are reported together:

| reference | cited-cell Jaccard |
|---|---|
| floor — different images, same condition | 0.436 |
| **cross — same image, different ordering** | **0.710** |
| ceiling — same image, same ordering, resampled | 0.800 |

Cross-condition agreement sits 75% of the way from unrelated evidence to pure
sampling noise, and the gap to the ceiling is not significant (permutation
p=0.068). **Reordering does not meaningfully change which cells get cited.** The
prose is rewritten — description token overlap is only 0.23 — but it points at
substantially the same places. Feature choice barely moves either, and barely
moves across *images*: the between-image feature Jaccard is 0.729, with
crypt/gland architecture and serration cited in 20/20 tiles under both orderings.

What ordering does change is the label. Six of 20 tiles flipped, and **all six
flipped the same way, cte=HP to etc=SSA, none the reverse** (exact McNemar
p=0.031). Explaining first pushes the model toward SSA: 18/20 SSA calls versus
12/20, with classify-only at 14/20 in between. Accuracy is 50% in all three
conditions regardless, so the ordering relocates the errors without fixing any.

The cleanest reading: the evidence is a stable, largely image-driven description
that both orderings produce nearly identically, while the label is a separate
and unstable read-out that ordering perturbs systematically. That is the
signature of an explanation that is not doing the work of the decision.

## Ablation: mask.py

Tests whether cited evidence is *causal*. Given a tile and a cell set, produces
three ablations — `mean` (whole-tile mean colour), `blur` (Gaussian, sigma=12),
`black` (hard occlusion) — across three cell sets, so 9 images per tile.

Masking is applied to the **clean** tile and the grid is redrawn afterwards.
The other order would paint over the green lines and destroy the cell labels
inside the masked region, leaving the model unable to read the coordinate system
it is being asked to use. `--no-regrid` skips the re-overlay.

sigma=12 comes from the Gaussian attenuation of a sinusoid,
`exp(-2*pi^2*sigma^2/L^2)`: suppressing the ~28px crypt period below 5% needs
sigma > 10.9. What survives is coarse tissue-vs-background layout, not
morphology — which is what a blur ablation should leave.

### The control problem

Three cell sets are produced: `cited`, `random` (same count, disjoint where
possible), and `tissue_matched` (same count, disjoint, closest achievable tissue
area — exhaustive over C(16,k)).

`tissue_matched` is the one that matters, and it exists because the naive
control is badly confounded. The cells a model does *not* cite are
systematically the empty border ones, so a plain random control masks mostly
blank slide and shows a smaller effect for reasons unrelated to causality. On
`MHIST_ekm` the random control masks 12.2% of tissue against the cited set's
32.6%; the tissue-matched control masks 31.5%.

A harder constraint follows from geometry. A same-size disjoint control is
forced into the complement of the cited set, so it can only match the cited
tissue area when that area is near half the tile's total. Masking **all** cited
cells, only **5 of 20** tiles support a fair control — the model cites 7–9 of 16
cells, covering 39–92% of all tissue, and there is simply no comparable region
left. Five tiles could not even get a disjoint control.

`--subset 3` masks only the three highest-tissue cited cells, which restores
feasibility on **18 of 20** tiles (median control gap under 1pp). Every manifest
carries a `match_quality` block with `usable`, the tissue gap, and the reason
when it fails; the CLI prints `SKEW` per tile and a summary count. Drop the
SKEW tiles rather than reporting them alongside the rest.

```bash
python3 mask.py --image MHIST_ccn.png --cells B2,B3,C2,C3
python3 mask.py --from-run runs/cte_p1.jsonl --subset 3 --out masked/cte_p1_k3
```

## Causal masking sweep (gemma, cte_p1, k=3)

120 calls: 20 tiles x {cited, tissue_matched} x {mean, blur, black}, zero
errors. Masking the 3 cells the model itself named flipped its label on
10-15% of tiles; masking tissue-matched cells it never mentioned flipped
15-25%. Pooled cited-only vs control-only flips: 7 vs 11, exact sign test
p=0.48 -- numerically the wrong direction for faithfulness. Mean |dconf|
never exceeded 0.018 in any arm. The citations are perceptually grounded
but **causally inert**. (`runs/masking_analysis.json`)

## Paraphrase crossing (all 9 prompts, gemma)

The HP->SSA ordering flip survives rewording: p1 6-0, p2 3-1, p3 4-0,
pooled 13 vs 1 reverse, exact McNemar **p=0.0018**. etc_* runs show a
higher SSA rate (17-19/20 vs 12-17/20) and higher confidence (0.85-0.86 vs
0.80-0.82) under every wording. Accuracy stays at chance in all nine runs
(9-13/20). Tile `ccn` flips under all three wordings.
(`runs/paraphrase_analysis.json`)

## Stability probe (5 tiles x 5 reps, cte_p1, temperature 1.0)

Mean within-image Jaccard 0.71 vs the 0.39 between-image floor; only 43% of
cells ever cited on a tile are cited in all five reps. The two failure modes
split cleanly and are both incompatible with the explanation driving the
decision: ddn/epj churn ~half their citations while the label stays fixed;
ccn/ekm hold citations nearly constant (J=0.91) while the label flips anyway.
(`runs/stability_analysis.json`)

## Biological masking (Week 2)

Structure-targeted occlusion instead of grid squares. `stains.py` partitions
tissue via rgb2hed into hematoxylin-dominant epithelium and eosin-dominant
stroma; `biological_mask.py` adds CellPose nuclei (x4 upsample of the H
channel, diameter 10; dense epithelial rims under-segmented) and builds, per
tile, each structure mask plus an area-matched block-shuffled control,
mean-filled and re-gridded. Feature mapping from gemma's own cte_p1 citations:
crypt/serration/surface -> epithelium (named on 20/20 tiles), stroma/
infiltrate -> stroma (named on 8), atypia/mitoses -> nuclei (named on 0).

120 calls, zero errors. Flip rates vs the unmasked baseline:

| source | struct | control | discordant | p |
|---|---|---|---|---|
| epithelium (named 20/20) | 45% | 35% | 4 vs 2 | 0.69 |
| stroma (named 8/20) | 35% | 45% | 1 vs 3 | 0.63 |
| nuclei (named 0/20) | **50%** | 35% | 3 vs 0 | 0.25 |

Pooled struct-vs-control: 8 vs 5, p=0.58 -- no significant targeting effect,
same null as the grid sweep (7 vs 11, p=0.48). The sharpest single fact: the
highest flip rate in the whole study comes from masking **nuclei, a structure
gemma never once named** across 26 cte_p1 responses. Flips run 40 HP->SSA vs 9
SSA->HP -- occlusion pushes toward SSA, echoing the explain-first bias --
and flip probability is unrelated to masked area (27.0% vs 27.8% of tile
for flipped vs not). Confidence stays frozen throughout (|dconf| <= 0.11).

Verdict: gemma is occlusion-sensitive but not evidence-specific. Perturbing
what it *says* it uses does no more than perturbing matched arbitrary tissue,
under grid-cell masking and under biologically-targeted masking alike -- and
the structure that perturbs it most is one it never mentions. This is the
distinction the biological sweep exists to draw, and it lands on the
unfaithful side. (`runs/biological_analysis.json`)

## Reproducing

```bash
python3 build_coverage.py      # ~30s, writes coverage/ + coverage_index.csv
python3 select_and_render.py   # writes gridded/, clean/, selection_manifest.json
python3 render_prompts.py      # writes prompts/rendered/
python3 run_experiment.py --prompt cte_p1   # ~10min at 5 req/min, resumable
python3 score.py --run runs/cte_p1.jsonl
python3 viz_results.py --run runs/cte_p1.jsonl
```

`run_experiment.py` appends to `runs/<prompt>.jsonl` as calls complete and skips
already-recorded (image, replicate) pairs on restart, so a rate-limit stall or
crash loses nothing.
