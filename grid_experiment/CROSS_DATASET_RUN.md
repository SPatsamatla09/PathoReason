# Cross-dataset analysis

Run from the PathoReason repository root after extracting the supplied bundle.

```bash
python3 grid_experiment/analyze_cross_dataset.py --mhist grid_experiment/runs/track_a_faithfulness_gpt4o_full_corrected/image_feature_results.csv --pcam-masking grid_experiment/pcam/runs/masking_cte_p1_k3_plip.jsonl --pcam-baseline grid_experiment/pcam/runs/plip_baseline.jsonl --out grid_experiment/runs/cross_dataset_comparison
```

The comparison uses the image as the inferential unit. Multiple MHIST feature
rows are averaged within each image before bootstrap confidence intervals and
Wilcoxon tests are calculated. The between-dataset test uses independent image
effects and a Mann-Whitney U test. Positive effects mean cited removal caused a
larger original-class probability drop than matched control removal; negative
effects mean the matched control caused the larger drop.

The current comparison is full MHIST (977 images) versus the existing
PatchCamelyon pilot (20 images), so the PatchCamelyon side must be labeled as a
pilot rather than a full-dataset validation.

