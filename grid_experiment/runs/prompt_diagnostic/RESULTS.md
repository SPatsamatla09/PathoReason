# Prompt diagnostic: results

## Setup

| | |
|---|---|
| date | 2026-09-30 |
| plan | written before any call (`PLAN.md`) |
| verification | every number recomputed independently; parser, providers, tiles and images audited |
| calls | 1,000: 0 errors, 0 unusable answers, every call served by its pinned provider |
| cost | about $0.61 |

## Results

Same 100 representative tiles (63 HP / 37 SSA; answering "always HP" = 63%),
temperature 1.0, one call per tile.

| model | arm | correct | 95% CI | balanced acc. | SSA calls | label vs truth p |
|---|---|---|---|---|---|---|
| GPT-4.1 | grid + cte (pilot) | 39 | 30–49% | 0.516 | 98% | 0.53 |
| GPT-4.1 | clean + cte | 58 | 48–67% | 0.650 | 73% | 0.001 |
| GPT-4.1 | grid + min | 37 | 28–47% | 0.500 | 100% | 1 |
| GPT-4.1 | clean + min | 37 | 28–47% | 0.500 | 100% | 1 |
| GPT-4.1 | clean + alias | 40 | 31–50% | 0.524 | 97% | 0.29 |
| Qwen3-VL-235B | all 5 arms | 37 | 28–47% | 0.500 | 100% | 1 |
| Gemini 2.5 Flash | grid + cte (pilot) | 57 | 47–66% | 0.592 | 56% | 0.096 |
| Gemini 2.5 Flash | clean + min | 44 | 35–54% | 0.544 | 89% | 0.21 |
| gemma-4-31b | grid + cte (pilot) | 61 | 51–70% | 0.568 | 32% | 0.19 |
| gemma-4-31b | clean + min | 39 | 30–49% | 0.516 | 98% | 0.53 |

**GPT-4.1 log-probability AUROC** (threshold-free):

| arm | AUROC | 95% CI |
|---|---|---|
| clean + min | 0.60 | 0.49–0.71 |
| grid + min | 0.61 | 0.50–0.72 |
| clean + alias | 0.61 | 0.50–0.72 |

## Rules set before the run

1. **Was the SSA default caused by the prompt or grid?** Answered in the clean + min
   arm: **No, for all four models.** The SSA rate was 89–100%, and balanced accuracy
   0.50–0.54.
2. **Is any model competent in any arm?** **No.** None of the 14 model × arm cells
   reaches 73/100; the best is 61/100.
3. **Do the label names shift the answer?** **No** for GPT-4.1 (−3 points) and Qwen (0).
   Not applicable to Gemini or gemma, which had no alias arm.

## What this licenses

**Narrow claim.** Neither the 4 × 4 grid nor the pipeline's feature vocabulary is
needed to produce the SSA default. No model met the bar under any tested prompt.

**Not licensed:**

- that the default is a model property independent of prompt wording. Every prompt
  still names SSA "sessile *serrated* adenoma/lesion" and gives no criteria.
- that answer position doesn't matter. The alias arm moved SSA to option A, listed
  first, which does not counterbalance position.

**The SSA rate depends strongly on the prompt:**

- **gemma:** 32% SSA under the gridded cte prompt, 98% under the one-line prompt. The
  prompt and the grid changed together.
- **GPT-4.1, grid only (exploratory):** removing only the grid moved GPT-4.1 from 98%
  to 73% SSA (25 of 25 changed tiles went toward HP; McNemar p = 6 × 10⁻⁸). The
  balanced-accuracy gain is mostly a threshold shift, and it is still 58/100, below
  baseline.
- **Caveat on that comparison:** the clean + cte prompt still tells the model a grid is
  drawn.

**Citations on clean tiles.** With no grid drawn, GPT-4.1 and Qwen still cited grid
cells in 100/100 responses, and never said the grid was missing. Their "citations" are
not grounded. This says nothing about gemma's measured grounding.

**Qwen is uninformative about the cause.** It gives the same answer on every tile in
every arm, and its provider returns no log-probabilities.

## Suggested wording

> In a pre-registered follow-up on the same 100 tiles, neither the grid nor the feature
> vocabulary explained the SSA default: with an ungridded tile and a one-sentence "HP or
> SSA?" prompt, GPT-4.1 and Qwen3-VL-235B answered SSA on 100% of tiles, gemma-4-31b on
> 98% and Gemini 2.5 Flash on 89%, and neutral letter labels with the class order
> reversed left GPT-4.1 at 97% and Qwen at 100%. No model met the competence bar in any
> of the 14 model-prompt combinations (best 61/100), and GPT-4.1's threshold-free
> log-probability AUROC was 0.60–0.61, with 95% CIs reaching 0.49–0.50. Every prompt
> tested named SSA as "sessile serrated adenoma/lesion" and gave no diagnostic criteria,
> so a prior attached to that wording, or to answer position, is not excluded. The SSA
> rate itself depends strongly on the prompt: the studied model answered SSA on 32% of
> these tiles under the gridded classify-then-explain prompt and 98% under the one-line
> prompt without the grid, and, in an exploratory comparison, removing only the grid
> moved GPT-4.1 from 98% to 73% SSA without reaching the bar (58/100). The pilot's
> conclusion therefore stands for these four models under zero-shot prompting: none
> was shown to discriminate HP from SSA.

## Overclaims to avoid

**About the prompt:**

- "the SSA default is a property of the model, not the prompt", or "prompt ruled out"
- "label names or order don't matter"
- "all four models become more SSA-biased under the minimal prompt". Only gemma and
  Gemini moved.
- "the cte prompt lowers the SSA rate" for gemma or Gemini. It is confounded with the
  grid.

**About GPT-4.1:**

- "removing the grid makes GPT-4.1 discriminate"
- "GPT-4.1 has latent HP/SSA knowledge"
- pooling the three AUROCs, or citing p = 0.001 as a pre-registered result

**About models in general:**

- "no VLM can separate HP from SSA"
- "Qwen has an SSA prior"

## Cheapest remaining controls (not run)

About $0.07 each:

- a counterbalanced alias arm with A = HP
- a no-image, text-only control for GPT-4.1 with log-probabilities

## Scope

Four general-purpose, non-reasoning VLMs, zero-shot, on 224-px tiles. Not tested:

- pathology-tuned models
- reasoning-enabled models
- prompts with diagnostic criteria or few-shot examples
