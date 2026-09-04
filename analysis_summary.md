# Blinded rating and error-taxonomy analysis

## Task 8 — inter-rater agreement

| Axis | Exact agreement | Cohen's κ | Cluster-bootstrap 95% CI |
|---|---:|---:|---:|
| Presence | 43.0% | 0.000 | [-0.004, 0.006] |
| Relevance | 65.0% | 0.161 | [-0.002, 0.323] |
| Support | 58.0% | 0.168 | [0.000, 0.305] |

Agreement was low on all three axes. Presence was especially discordant: rater A marked 99/100 items present, whereas rater B marked 43 present, 16 uncertain, and 41 absent. The two raters never jointly marked an item absent. Accordingly, a reliable consensus-presence variable could not be formed.

### Exploratory association with masking faithfulness

Only base-run Gemma ratings can be matched to the 20-image Gemma masking experiment; the random rating sample provides at least one rated evidence item for 17 of those images. Because strict presence consensus had no usable variation, correlations use the mean ordinal score across the two raters and are exploratory, not consensus correlations.

The primary exploratory comparison—mean presence score versus cited-minus-control label-flip advantage—gave Spearman ρ=0.033, p=0.899 (n=17 images), indicating no detectable association. No exploratory comparison remained significant after Holm correction.

## Task 10 — conservative error taxonomy

| Category | Count | Denominator | Interpretation |
|---|---:|---:|---|
| Visually unsupported in cited cells | 0 | 100 | not reliably resolved because presence kappa is approximately zero |
| Hallucinated feature | not estimable | 100 | not separately measured; requires confirming absence from the entire tile |
| Correct feature, wrong cited region | not estimable | 100 | not separately measured; form assessed cited cells but not whole-tile presence |
| Correct cited region, diagnostically irrelevant | 2 | 100 | confirmed only when both raters marked present, irrelevant, and neutral |
| Evidence contradicts stated label | 0 | 100 | two-rater conservative and one-rater screening counts |
| Label unstable across repeated samples | 3 | 5 | model output changed label across five repetitions |
| Citation localization unstable (mean Jaccard < 0.50) | 2 | 5 | descriptive threshold applied to five repetitions |
| Any sampled-output instability | 4 | 5 | label instability or citation Jaccard below 0.50 |

The current form cannot separate a hallucinated feature from a correct feature cited in the wrong region: both would appear as ‘not visible in the cited cells.’ That distinction requires a new whole-tile review of the disputed presence cases. Therefore, the taxonomy is complete for the categories actually measured (irrelevance and sampled-output instability), but the first two requested categories must remain combined/unresolved unless adjudication is added.

## Reporting recommendation

Report the low agreement directly. Do not convert disagreements into a two-rater majority, and do not describe rater-average scores as consensus. For definitive hallucination and wrong-region counts, use an independent expert adjudicator or a follow-up whole-tile rating axis.
