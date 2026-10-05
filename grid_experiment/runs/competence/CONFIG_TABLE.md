| config | model | provider | control | set | n | acc | acc 95% CI | bal acc | recall HP | recall SSA | parse | SSA calls | cost | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| co_p1 | medgemma-1.5-4b-it-Q8_0 | local [tier1-t1] | none | screen | 100 | 0.620 | 0.522–0.709 | 0.520 | 0.90 | 0.14 | 1.00 | 0.11 | $0.000 | stops |
| co_p1 | medgemma-1.5-4b-it-Q8_0 | local [tier2-t0] | none | screen | 100 | 0.630 | 0.532–0.718 | 0.500 | 1.00 | 0.00 | 1.00 | 0.00 | $0.000 | stops |
| cte_p1 | medgemma-1.5-4b-it-Q8_0 | local [tier1-t1] | none | screen | 100 | 0.570 | 0.472–0.663 | 0.452 | 0.90 | 0.00 | 0.92 | 0.00 | $0.000 | stops |
| cte_p1 | medgemma-1.5-4b-it-Q8_0 | local [tier2-t0] | none | screen | 5 | 0.600 | 0.231–0.882 | 0.500 | 1.00 | 0.00 | 0.80 | 0.00 | $0.000 | stops |
| names_crit_fs | google/gemma-4-31b-it | friendli | none | screen | 100 | 0.600 | 0.502–0.691 | 0.599 | 0.60 | 0.59 | 1.00 | 0.47 | $0.042 | stops |
| names_crit_fs | qwen/qwen3-vl-235b-a22b-instruct | alibaba | none | screen | 100 | 0.400 | 0.309–0.498 | 0.524 | 0.05 | 1.00 | 1.00 | 0.97 | $0.048 | stops |
| neutral_crit_fs | google/gemma-4-31b-it | friendli | none | screen | 100 | 0.650 | 0.552–0.736 | 0.694 | 0.52 | 0.86 | 1.00 | 0.62 | $0.042 | advances |
| neutral_crit_fs | google/gemma-4-31b-it | friendli | none | dev | 300 | 0.573 | 0.517–0.628 | 0.626 | 0.42 | 0.83 | 1.00 | 0.67 | $0.126 | fail |
| neutral_crit_fs | medgemma-1.5-4b-it-Q8_0 | local [tier1-t1] | none | screen | 100 | 0.450 | 0.356–0.548 | 0.441 | 0.48 | 0.41 | 0.97 | 0.48 | $0.000 | stops |
| neutral_crit_fs | qwen/qwen3-vl-235b-a22b-instruct | alibaba | none | screen | 100 | 0.400 | 0.309–0.498 | 0.524 | 0.05 | 1.00 | 1.00 | 0.97 | $0.047 | stops |
| neutral_cte | google/gemma-4-31b-it | friendli | none | screen | 100 | 0.470 | 0.375–0.567 | 0.563 | 0.21 | 0.92 | 1.00 | 0.84 | $0.019 | stops |
| neutral_cte | medgemma-1.5-4b-it-Q8_0 | local [tier1-t1] | none | screen | 100 | 0.490 | 0.394–0.587 | 0.495 | 0.48 | 0.51 | 0.99 | 0.53 | $0.000 | stops |
| neutral_cte | qwen/qwen3-vl-235b-a22b-instruct | alibaba | none | screen | 100 | 0.370 | 0.282–0.468 | 0.500 | 0.00 | 1.00 | 1.00 | 1.00 | $0.036 | stops |
