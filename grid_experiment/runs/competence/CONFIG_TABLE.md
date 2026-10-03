| config | model | provider | control | set | n | acc | acc 95% CI | bal acc | recall HP | recall SSA | parse | SSA calls | cost | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| names_crit_fs | google/gemma-4-31b-it | friendli | none | screen | 100 | 0.600 | 0.502–0.691 | 0.599 | 0.60 | 0.59 | 1.00 | 0.47 | $0.042 | stops |
| neutral_crit_fs | google/gemma-4-31b-it | friendli | none | screen | 100 | 0.650 | 0.552–0.736 | 0.694 | 0.52 | 0.86 | 1.00 | 0.62 | $0.042 | advances |
| neutral_crit_fs | google/gemma-4-31b-it | friendli | none | dev | 300 | 0.573 | 0.517–0.628 | 0.626 | 0.42 | 0.83 | 1.00 | 0.67 | $0.126 | fail |
| neutral_cte | google/gemma-4-31b-it | friendli | none | screen | 100 | 0.470 | 0.375–0.567 | 0.563 | 0.21 | 0.92 | 1.00 | 0.84 | $0.019 | stops |
| neutral_cte | qwen/qwen3-vl-235b-a22b-instruct | alibaba | none | screen | 100 | 0.370 | 0.282–0.468 | 0.500 | 0.00 | 1.00 | 1.00 | 1.00 | $0.036 | stops |
