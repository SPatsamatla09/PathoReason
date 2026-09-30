# Ordering controls, host `openrouter__google-gemma-4-31b-it__friendli` (K = 3 replicates per tile)

**Reproduction gate: PASS.** Same-host classify-only vs explain-then-classify, replicate 1: 23 HP→SSA / 9 SSA→HP (Cerebras reference 28 / 3; classify-then-explain vs explain-then-classify reference 41 / 0).

| condition | rep-1 HP→SSA / SSA→HP vs classify-only | McNemar p | Δ SSA rate (K reps) [95% CI] | Holm p | reading |
|---|---|---|---|---|---|
| ins_filler | 9 / 8 | 1 | +0.010 [-0.050, +0.070] | 0.826 | equivalent to classify-only (within +/-10 pts) |
| ins_checklist | 18 / 4 | 0.00434 | +0.130 [+0.067, +0.193] | 0.0004 | shift toward SSA |
| ins_copied | 28 / 22 | 0.48 | +0.060 [-0.073, +0.190] | 0.803 | inconclusive |
| describe_first | 21 / 10 | 0.0708 | +0.107 [+0.047, +0.170] | 0.0033 | shift toward SSA |

Tiles analysed: 100 of 100.

Noise floor, classify-only replicate 1 vs 2: 5 / 11 flips (rate 0.16). Compare each condition's total rep-1 flip rate against this, descriptively:

- ins_filler: flip rate 0.17
- ins_checklist: flip rate 0.22
- ins_copied: flip rate 0.5
- describe_first: flip rate 0.31

Secondary (outside Holm):

- ins_self_vs_classify_only: +0.310 [+0.213, +0.403] (n=100)
- ins_self_minus_ins_copied: +0.250 [+0.120, +0.380] (n=100)
- etc_minus_ins_self: -0.137 [-0.210, -0.063] (n=100)
- etc_minus_ins_filler: +0.163 [+0.093, +0.233] (n=100)
- etc_minus_ins_checklist: +0.043 [-0.030, +0.117] (n=100)
- etc_minus_ins_copied: +0.113 [-0.017, +0.247] (n=100)
- etc_minus_describe_first: +0.067 [-0.010, +0.143] (n=100)
- ins_self_without_lexical_SSA_texts: +0.369 [+0.271, +0.463] (n=85)
- HP-direction inserted text, unpaired: ins_self -0.150 [-0.333, +0.033] (n=20); ins_copied -0.380 [-0.547, -0.213] (n=50)
- SSA-direction inserted text, unpaired: ins_self +0.425 [+0.333, +0.517] (n=80); ins_copied +0.500 [+0.387, +0.613] (n=50)
- ins_copied, donor cells land on recipient background: +0.187 [-0.013, +0.373] (n=50); all on tissue: -0.067 [-0.240, +0.107] (n=50)

> All secondary contrasts are descriptive. etc minus each control differs in instruction wording and output format as well as content. ins_self_minus_ins_copied is NOT a tile-specificity test: ins_self texts are 80% SSA-direction vs 50/50 for donors, 15 name SSA lexically, and both were generated on Cerebras; read the direction-matched and lexical-split rows instead.

Exploratory:

- donor-text direction (HP-text minus SSA-text recipient shift): -0.8868, Welch t -7.662, text-level permutation p 0.0001999600079984003 (9 HP / 50 SSA texts)
- checklist order (hp_first minus ssa_first): 0.0418, permutation p 0.5100979804039192

Counts per condition (records / valid / errors by kind):

- co: {'records': 300, 'valid': 300}
- cte: {'records': 300, 'valid': 300}
- etc: {'records': 300, 'valid': 300}
- ins_filler: {'records': 300, 'valid': 300}
- ins_checklist: {'records': 300, 'valid': 300}
- ins_copied: {'records': 300, 'valid': 300}
- describe_first: {'records': 300, 'valid': 300}
- ins_self: {'records': 300, 'valid': 300}

Manipulation checks: {"ins_self": {"order_fail": 0, "failed_manipulation_total": 0, "of": 300, "median_copy_similarity": 1.0, "preamble_words_median": 52}, "ins_checklist": {"order_fail": 0, "failed_manipulation_total": 0, "of": 300, "median_copy_similarity": 1.0, "preamble_words_median": 54}, "etc": {"order_fail": 0, "failed_manipulation_total": 0, "of": 300}, "ins_filler": {"order_fail": 0, "failed_manipulation_total": 0, "of": 300, "median_copy_similarity": 1.0, "preamble_words_median": 52}, "describe_first": {"order_fail": 0, "failed_manipulation_total": 130, "of": 300, "with_diagnostic_terms": 46, "with_criterion_paraphrases": 125, "most_common_terms": [["jagged", 66], ["\\btooth", 57], ["saw-tooth", 38], ["\\bbranch", 27], ["zig-?zag", 12], ["sawtooth", 7], ["\\bwiden", 4], ["toothed", 1], ["gland", 1]], "preamble_words_median": 39}, "ins_copied": {"order_fail": 0, "failed_manipulation_total": 0, "of": 300, "median_copy_similarity": 1.0, "preamble_words_median": 51}}
