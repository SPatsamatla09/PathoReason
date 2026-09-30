# Prompt diagnostic: is the all-SSA behaviour the prompt, or the model?

Written 2026-09-30, before any diagnostic call.

## Question

In the competence pilot (100 representative test tiles, gridded image, cte_p1 prompt),
GPT-4.1 answered SSA on 98% of tiles and Qwen3-VL-235B on 100%. Two possible causes:

- **Prompt or input.** cte_p1 expands SSA as "sessile *serrated* adenoma", offers
  "serration" as a feature term, and gives no criteria. The 4 × 4 grid covers the small
  224-px tile.
- **Model.** The models simply cannot separate HP from SSA.

## Arms

All arms use the same 100 pilot tiles (`runs/.pilot100_tiles.json`, 63 HP / 37 SSA),
temperature 1.0, one call per tile, and the same pinned provider per model:

| arm | image | prompt |
|---|---|---|
| grid + cte (existing pilot) | 4×4 grid | cte_p1: classify, then explain with grid-cell citations |
| clean + cte | no grid | cte_p1 (grid instructions left as written) |
| grid + min | 4×4 grid | MIN (below) |
| **clean + min (key arm)** | no grid | MIN |
| clean + alias | no grid | ALIAS (below): neutral letters, order swapped |

**MIN:**

> This is a 224x224 pixel H&E-stained image of a colorectal polyp. Is it a
> hyperplastic polyp (HP) or a sessile serrated adenoma/lesion (SSA)? Answer with
> exactly one of: HP, SSA.

**ALIAS:**

> This is a 224x224 pixel H&E-stained image of a colorectal polyp. It belongs to one of
> two classes. Class A: sessile serrated adenoma/lesion. Class B: hyperplastic polyp.
> Answer with exactly one letter: A or B.

For ALIAS, A maps to SSA and B to HP.

**Which models run which arms:**

- **GPT-4.1 (OpenAI) and Qwen3-VL-235B (Alibaba):** all four new arms.
- **Gemini 2.5 Flash (Google AI Studio, thinking off) and gemma-4-31b (Friendli):**
  clean + min only, to see whether any model becomes competent without the grid and
  vocabulary.
- **Log-probabilities:** GPT-4.1 only, on the MIN and ALIAS arms (first answer token,
  top 10). The Alibaba endpoint rejects the logprobs parameter.

## Measures

**Per model and arm:**

- accuracy with Wilson 95% CI
- balanced accuracy
- sensitivity (SSA) and specificity (HP)
- SSA-call rate
- Fisher test of label vs truth
- unusable answers, counted and never re-coded

**For GPT-4.1:** AUROC of P(SSA) from log-probabilities, with a tile-bootstrap 95% CI.
P(SSA) is computed as the probability mass on first tokens starting with "S" (or "A"
in the ALIAS arm), relative to "H" (or "B").

## Decision rules (fixed now)

1. **Prompt/grid-induced default:** for a model, **yes** if in the clean + min arm its
   SSA-call rate is ≤ 75% and its balanced accuracy is ≥ 0.60. For GPT-4.1 the rule
   additionally needs a logprob AUROC ≥ 0.65 with a CI lower bound > 0.5.
2. **Competence (same bar as the pilot):** a Wilson 95% lower bound on accuracy
   > 0.632, i.e. ≥ 73/100 correct. It is evaluated in each arm. An arm-specific pass is
   reported as such; it is not generalised to the pipeline's prompt.
3. **Label-name bias:** if the SSA rate in clean + alias differs from clean + min by
   ≥ 20 points, the label names themselves shift the answer.

With 5 arms × 2 models plus 2 extra model-arm cells, and several measures, everything
beyond these three rules is descriptive.
