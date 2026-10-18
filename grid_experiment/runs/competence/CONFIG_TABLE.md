| config | model | provider | control | set | n | acc | acc 95% CI | bal acc | recall HP | recall SSA | parse | SSA calls | cost | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| co_p1 | medgemma-1.5-4b-it-Q8_0 | local [tier1-t1] | none | screen | 100 | 0.620 | 0.522–0.709 | 0.520 | 0.90 | 0.14 | 1.00 | 0.11 | $0.000 | stops |
| co_p1 | medgemma-1.5-4b-it-Q8_0 | local [tier2-t0] | none | screen | 100 | 0.630 | 0.532–0.718 | 0.500 | 1.00 | 0.00 | 1.00 | 0.00 | $0.000 | stops |
| cte_p1 | medgemma-1.5-4b-it-Q8_0 | local [tier1-t1] | none | screen | 100 | 0.570 | 0.472–0.663 | 0.452 | 0.90 | 0.00 | 0.92 | 0.00 | $0.000 | stops |
| cte_p1 | medgemma-1.5-4b-it-Q8_0 | local [tier2-t0] | none | screen | 100 | 0.490 | 0.394–0.587 | 0.389 | 0.78 | 0.00 | 0.75 | 0.00 | $0.000 | stops |
| cte_p1 | medgemma-lora-posthoc-cte-v1 | kaggle [tier1-t1] | mismatch | dev | 300 | 0.527 | 0.470–0.583 | 0.490 | 0.63 | 0.35 | 0.99 | 0.36 | $0.000 | control OK |
| cte_p1 | medgemma-lora-posthoc-cte-v1 | kaggle [tier1-t1] | noimage | dev | 300 | 0.580 | 0.523–0.634 | 0.570 | 0.61 | 0.53 | 1.00 | 0.44 | $0.000 | control FAILS (holds up without the right image) |
| cte_p1 | medgemma-lora-posthoc-cte-v1 | kaggle [tier1-t1] | none | screen | 100 | 0.830 | 0.745–0.891 | 0.815 | 0.87 | 0.76 | 1.00 | 0.36 | $0.000 | advances |
| cte_p1 | medgemma-lora-posthoc-cte-v1 | kaggle [tier1-t1] | none | dev | 300 | 0.817 | 0.769–0.856 | 0.808 | 0.84 | 0.77 | 1.00 | 0.39 | $0.000 | PASS |
| cte_p1 | medgemma-lora-posthoc-cte-v1 | kaggle [tier2-t0] | mismatch | dev | 300 | 0.513 | 0.457–0.569 | 0.480 | 0.61 | 0.35 | 1.00 | 0.37 | $0.000 | control OK |
| cte_p1 | medgemma-lora-posthoc-cte-v1 | kaggle [tier2-t0] | noimage | dev | 300 | 0.630 | 0.574–0.683 | 0.500 | 1.00 | 0.00 | 1.00 | 0.00 | $0.000 | control OK |
| cte_p1 | medgemma-lora-posthoc-cte-v1 | kaggle [tier2-t0] | none | screen | 100 | 0.860 | 0.779–0.915 | 0.850 | 0.89 | 0.81 | 1.00 | 0.37 | $0.000 | advances |
| cte_p1 | medgemma-lora-posthoc-cte-v1 | kaggle [tier2-t0] | none | dev | 300 | 0.820 | 0.773–0.859 | 0.809 | 0.85 | 0.77 | 1.00 | 0.37 | $0.000 | PASS |
| names_crit_fs | google/gemma-4-31b-it | friendli | none | screen | 100 | 0.600 | 0.502–0.691 | 0.599 | 0.60 | 0.59 | 1.00 | 0.47 | $0.042 | stops |
| names_crit_fs | qwen/qwen3-vl-235b-a22b-instruct | alibaba | none | screen | 100 | 0.400 | 0.309–0.498 | 0.524 | 0.05 | 1.00 | 1.00 | 0.97 | $0.048 | stops |
| neutral_crit_fs | google/gemma-4-31b-it | friendli | none | screen | 100 | 0.650 | 0.552–0.736 | 0.694 | 0.52 | 0.86 | 1.00 | 0.62 | $0.042 | advances |
| neutral_crit_fs | google/gemma-4-31b-it | friendli | none | dev | 300 | 0.573 | 0.517–0.628 | 0.626 | 0.42 | 0.83 | 1.00 | 0.67 | $0.126 | fail |
| neutral_crit_fs | medgemma-1.5-4b-it-Q8_0 | local [tier1-t1] | none | screen | 100 | 0.450 | 0.356–0.548 | 0.441 | 0.48 | 0.41 | 0.97 | 0.48 | $0.000 | stops |
| neutral_crit_fs | medgemma-1.5-4b-it-Q8_0 | local [tier2-t0] | none | screen | 100 | 0.490 | 0.394–0.587 | 0.495 | 0.48 | 0.51 | 1.00 | 0.52 | $0.000 | stops |
| neutral_crit_fs | qwen/qwen3-vl-235b-a22b-instruct | alibaba | none | screen | 100 | 0.400 | 0.309–0.498 | 0.524 | 0.05 | 1.00 | 1.00 | 0.97 | $0.047 | stops |
| neutral_cte | google/gemma-4-31b-it | friendli | none | screen | 100 | 0.470 | 0.375–0.567 | 0.563 | 0.21 | 0.92 | 1.00 | 0.84 | $0.019 | stops |
| neutral_cte | medgemma-1.5-4b-it-Q8_0 | local [tier1-t1] | none | screen | 100 | 0.490 | 0.394–0.587 | 0.495 | 0.48 | 0.51 | 0.99 | 0.53 | $0.000 | stops |
| neutral_cte | medgemma-1.5-4b-it-Q8_0 | local [tier2-t0] | none | screen | 100 | 0.490 | 0.394–0.587 | 0.495 | 0.48 | 0.51 | 1.00 | 0.52 | $0.000 | stops |
| neutral_cte | qwen/qwen3-vl-235b-a22b-instruct | alibaba | none | screen | 100 | 0.370 | 0.282–0.468 | 0.500 | 0.00 | 1.00 | 1.00 | 1.00 | $0.036 | stops |
