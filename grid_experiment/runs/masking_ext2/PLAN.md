# Masking: the remaining 450 test tiles ("ext2")

Written 2026-09-30, before any call.

## Why

Kiran item 4 asked to extend masking to the full set. After ext2, every one of the 977
test tiles has been masked or tried. Round 3 found the cited > control effect
significant on **one tile set only**: the 319-tile extension (102 vs 49). ext2 is a
fresh, never-masked tile set on the **same host** (OpenRouter → Friendli, gemma-4-31b-it,
pinned, host-default sampling), so it is also a direct replication test.

## Design

Identical to the extension sweep:

- **Tiles:** all 450 remaining test tiles (`runs/.mask_ext2_tiles.json`).
- **Baseline:** fresh cte_p1 answers on Friendli (replicate 1) give both the baseline
  labels and the cited cells.
- **Masks:** `mask.py --subset 3 --per-tile-seed`, i.e. the 3 highest-tissue cited cells
  vs a disjoint area-matched control. Tiles with no feasible control are set aside
  (`finalize_mask_ext.py`).
- **Occlusions:** mean fill, blur, black, giving 6 calls per usable tile.
- **Secondary arm:** a tissue-type-matched control (`mask_typematched.py`: tissue and
  epithelial area within 5 pp, fewest model-cited cells) on the tiles where it is
  feasible. It runs at the same time as the main arm.

## Pre-specified analysis

**Unit:** the tile. net = the sum over the 3 occlusions of (cited flip − control flip),
where a flip means the masked label differs from the baseline label. The test is an
exact two-sided sign test over tiles with net ≠ 0.

**1. PRIMARY: replication.** Cited vs area-matched on ext2 alone.

- **Replicates:** p < .05 in the cited > control direction.
- **Opposite:** p < .05 in the other direction.
- Otherwise **not replicated**. If so, the power at the extension's cited share of
  0.675 is also reported.

**2. Heterogeneity:** Fisher exact test, extension (102:49) vs ext2 (cited-more :
control-more).

**3. Secondary:** cited vs type-matched on the feasible ext2 tiles (descriptive,
outside the primary test).

**4. Full-set summary (descriptive):**

- **Per-tile-set table** covering all 977 test tiles: legacy Cerebras, the crossover
  and own-citation re-runs on Friendli, the extension, and ext2.
- **One Friendli pool**, fixed now: extension + ext2 + the legacy tiles with Friendli's
  own citations. These are all the test tiles, each measured once on one host with its
  own citations. It is reported with that tile-set heterogeneity alongside, and is not
  cited without it.
- **No pooled cross-host p.**

**Integrity gates:**

- Every baseline tile has a valid label; missing tiles are re-run.
- Every masked call is served by Friendli, and matches its baseline provider.
- No duplicate calls.
