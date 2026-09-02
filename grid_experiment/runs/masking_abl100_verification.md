# Design-matched masking sweep (abl100): adversarial verification record

Claim under test: in the design-matched 100-tile sweep, masking gemma's cited
cells flips its label more than masking tissue-matched controls (pair-level
58 vs 37 discordant, sign test p=0.040) — the opposite sign of the pilot
(7 vs 11) and the proportional 100-tile sweep (40 vs 54).

Verification (9 agents, 2026-09-02): two blind recomputations from raw JSONL,
three stratum audits, one heterogeneity/multiplicity analysis, three
independent refuters (statistician, design critic, pathology-informed).
**Verdict: refuted, 3/3, all high confidence.** The counts are real; the
inference is not.

1. **Pseudo-replication.** The three occlusions of a tile mask the same cells
   against one baseline sample. At the tile level: 34 tiles cited-more, 22
   control-more, 44 tied, p=0.14; arm-swap permutation p=0.13; tile-cluster
   bootstrap 95% CI on the discordant difference [-5, 46].
2. **The cited arm did not change; the control arm did.** On baseline-HP tiles
   (where all of the effect lives) the cited-arm flip rate is 27.3% in sweep B
   vs 28.4% in C (Fisher p=0.90); the control-arm rate fell from 25.5% to
   14.2% (p=0.010). Sweep C's controls are worse matched (mean tissue gap
   1.27pp vs 0.71pp; 26 vs 11 tiles >2pp) and cluster in the D3/D4/C3/C4
   corner.
3. **Stratum-driven.** Baseline-HP tiles 42 vs 16 (p=0.0009); baseline-SSA
   tiles 16 vs 21 (sweep A/B sign); unanimous-agreement tiles 10 vs 17 (A/B
   sign); blur occlusion exactly null (13 vs 12); the 15 tiles copied from
   sweep B show B's sign (6 vs 8). The excess sits in wrongly-HP-called
   true-SSA tiles (27 vs 5), where resampling already drifts toward SSA.
4. **Noise floor.** Unmasked temperature-1.0 resamples disagree 24% pairwise
   (stability probe); C's cited-arm flip rate is 26%. Masking the model's own
   evidence flips the label about as often as re-asking about the untouched
   image.
5. **Heterogeneity and multiplicity.** Pooled across sweeps with the overlap
   counted once: 99 vs 94, p=0.77. Chi-square heterogeneity across sweeps
   p=0.009 — the sweeps disagree the way tile-composition changes or noise
   would produce, not the way a uniform causal effect would. p=0.040 is the
   smallest of 12 grid-masking sign tests (16 with the biological family) and
   fails Holm at every family size (most lenient threshold 0.0056). Sweep C's
   stratification was chosen after A and B returned null/opposite results.
6. **Fragility.** Dropping the 3 highest-net tiles: 49 vs 37 (p=0.24); 5
   tiles: 43 vs 37 (p=0.58); 8 tiles: 36 vs 37 (p=1.0).
7. **Not faithful even if real.** Flip ratio 1.36 with confidence pinned at 0.8
   on 533/600 records and a 19% flip rate from masking cells never mentioned
   does not meet the project's own criterion (cited >> control).

Consequence for the study: the masking family's conclusion stands — no
consistent causal advantage for cited cells across three sweeps totalling 220
tiles and 1,320 calls. `analyze_masking.py` now reports the tile-level
(cluster-correct) contrast alongside the pair-level one; use the tile-level
p in any writeup.
