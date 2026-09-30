# Prompt diagnostic: results

## Setup

| | |
|---|---|
| date | 2026-09-30 |
| plan | written before any call (`PLAN.md`) |
| verification | every number recomputed independently; parser, providers, tiles and images audited |
| calls | 1,000 in the main run, plus 400 in the two controls (addendum below): 0 errors, every call served by its pinned provider |
| cost | about $0.61 main run, about $0.1 controls |

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
- ~~that answer position doesn't matter~~. **Resolved by the addendum controls below:**
  letter and position are excluded as the driver.

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
- "Qwen has an SSA prior" in general. The addendum supports it only under the one-line
  prompt, which still names SSA "sessile serrated".

## Addendum: the two controls (pre-registered, run 2026-09-30)

Rules 4 and 5 were written into `PLAN.md` before these arms ran. The results were
verified by independent recomputation.

| arm | GPT-4.1 | Qwen3-VL-235B |
|---|---|---|
| clean + alias (A = SSA, listed first) | 97% SSA | 100% SSA |
| **clean + alias_rev (A = HP, B = SSA listed second)** | **98% SSA** | **100% SSA** |
| **no image + min** | **96/100 refusals**; the 4 answers were all SSA | **100/100 SSA** |

**Rule 4: letter and position are excluded as the driver.** Swapping which letter and
which position SSA occupies changes nothing. Without a content preference the two arms
would sum to about 100% SSA; they sum to 195% (GPT-4.1) and 200% (Qwen). The "SSA"
acronym is also excluded, because the alias prompts spell out both classes and never
contain it.

**Rule 5: does the SSA preference need an image?**

- **Qwen: no.** It answers SSA on 100/100 with no image at all (Wilson lower bound
  0.96). Its default exists without an image, under this wording. Its answers are
  identical with and without an image, so no image influence is visible. Its provider
  gives no log-probabilities.
- **GPT-4.1, labels.** As the rule is written, 4/4 answers were SSA, which reads
  "default exists", but that rests on only 4 of 100 calls. I report it as not
  assessable from labels.
  - **Deviation:** the ≥ 50-answer threshold behind that call is not in `PLAN.md`; it
    was added after seeing 96 refusals.
  - GPT-4.1 refused mostly because the prompt still says an image is attached.
- **GPT-4.1, log-probabilities.** Among answer tokens it prefers SSA even with no image
  (median P(SSA | answer token) 0.99998). With the image the preference is about 2.5
  orders of magnitude *more* SSA-leaning. So the rule's "image adds no SSA evidence"
  reading is not triggered.
  - **Caveat:** that comparison mixes the registered prefix scoring (clean + min) with
    exact-token scoring (no image). The two agreed to 10⁻¹⁶ where both could be
    computed.
- **GPT-4.1, image and ranking (exploratory).** Per-tile scores are consistent across
  the clean prompts (Spearman 0.81–0.89), but the ranking does not track the truth
  (AUROC 0.57–0.61). The no-image AUROC is 0.51 [0.40, 0.63], a null that validates the
  scoring.

**Two fixes made during this run, both recorded:**

- The first GPT-4.1 attempt at these arms scored refusals such as "Sorry…" as SSA,
  because of the prefix rule. It was stopped and moved to `superseded/`, which no
  analysis or sync reads. The arms were re-run with exact-token scoring, and the raw
  top-10 tokens were saved.
- A loose parser labelled 32 no-image refusals as "HP". The stored labels are corrected
  to the strict bare-answer parse, with the old value kept as `label_lenient_v1`.

**What the controls license.** The SSA default is not produced by:

- the grid
- the feature vocabulary
- the answer letter
- the answer position
- the "SSA" acronym

For Qwen it needs no image at all. For GPT-4.1 the image makes it stronger, not
correct.

**Still untested:**

- every prompt names SSA "sessile **serrated** adenoma/lesion" and gives no criteria
- letter prompts without an image
- Gemini and gemma on the controls
- more than one sample per cell

## Scope

Four general-purpose, non-reasoning VLMs, zero-shot, on 224-px tiles. Not tested:

- pathology-tuned models
- reasoning-enabled models
- prompts with diagnostic criteria or few-shot examples
