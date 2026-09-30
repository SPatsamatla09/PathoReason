# Cost estimate: running the full pipeline on a second, competent VLM

Prepared 2026-09-30. No candidate has been run beyond a single probe call.

## Measured cost per call

One real classify-then-explain call per candidate: the actual cte_p1 prompt plus a
gridded MHIST tile, temperature 1.0, one pinned first-party provider, thinking off.
The per-call costs are OpenRouter's billed `usage.cost` for that call. Raw data is in
`runs/.candidate_probe.json`.

| model | pinned provider | input tok | output tok | cost / call | JSON valid |
|---|---|---|---|---|---|
| qwen/qwen3-vl-235b-a22b-instruct | Alibaba | 616 | 210 | $0.00038 | yes |
| google/gemini-2.5-flash (thinking off) | Google AI Studio | 813 | 234 | $0.00083 | yes |
| openai/gpt-4.1 | OpenAI | 794 | 120 | $0.00255 | yes |
| anthropic/claude-sonnet-5.5 | Azure (accepts temperature) | 824 | 340 | $0.00505 | yes |
| *(reference)* gemma-4-31b-it | Friendli | 825 | ~205 | ~$0.00019 | yes |

Two models were considered and set aside:

- **Excluded: google/gemini-3.8-flash.** OpenRouter returned "Reasoning is mandatory
  for this endpoint and cannot be disabled." Hidden reasoning before the label would
  confound the ordering experiment: the model would "explain first" even in
  classify-only mode. Frontier reasoning-only models are set aside for the same reason.
- **Not pinned: Claude on Anthropic direct.** That endpoint does not accept a
  temperature parameter through OpenRouter, so it is pinned to Azure instead.

## Calls in the full pipeline

The pipeline mirrors what gemma received:

| stage | calls |
|---|---|
| **Pilot (competence gate):** 100 abl100 tiles, cte_p1 | 100 |
| Classification, rest of the 977-tile test set | 877 |
| Masking, the same 504 tiles × (cited + area-matched + tissue-type-matched) × 3 occlusions, upper bound | 4,536 |
| Ordering, 8 conditions × 100 tiles × K = 3 (also gives the self-consistency data) | 2,400 |
| Stability probe, 5 tiles × 5 repeats | 25 |
| **Total** | **≈ 7,940** |

## Cost

The +15% column allows for retries and the longer outputs of the explain-first and
describe prompts.

| model | pilot (100 calls) | full pipeline | +15% |
|---|---|---|---|
| Qwen3-VL-235B | $0.04 | $3.01 | **$3.46** |
| Gemini 2.5 Flash | $0.08 | $6.58 | **$7.57** |
| GPT-4.1 | $0.25 | $20.23 | **$23.27** |
| Claude Sonnet 5.5 | $0.50 | $40.08 | **$46.09** |

## Is there enough credit?

- **Current balance.** At the time of the probe it was $4.17. Today's host-investigation
  and tissue-type runs use about $0.6, leaving **about $3.5**.
- **Pilots.** All four together come to about $0.9. **Affordable.**
- **Full pipeline:**
  - **Qwen3-VL-235B:** about $3.5 with margin, which is essentially all the remaining
    credit. **Not safely affordable** without a small top-up of about $2.
  - **Gemini 2.5 Flash:** needs about $5 more.
  - **GPT-4.1:** needs about $20 more.
  - **Claude Sonnet 5.5:** needs about $45 more.

## Recommendation

**1. Run the competence pilot first.** Candidates, on the same 100 tiles:

- Qwen3-VL-235B
- Gemini 2.5 Flash
- GPT-4.1

That costs about $0.4. **The unknown is not cost, it is whether any general VLM clears
63.2%.**

- No general VLM has published MHIST zero-shot results we could check.
- GPT-4o earlier returned SSA for every tile.
- On 100 tiles, clearing the baseline needs roughly ≥ 73% for the lower 95% bound to
  exceed 63.2%.

**2. Pick the cheapest candidate that clears the bar** and top up enough for its full
run.

**3. If none clears it,** report that plainly. The options would be few-shot prompting,
which changes the pipeline and raises tokens per call, or a pathology-specialised VLM,
which is not available on OpenRouter.

## Pinning

The runners support any OpenRouter model through `PATHO_MODEL`, `PATHO_PROVIDER` and
`PATHO_BASE_URL`. One pinned provider per model, with fallbacks off, keeps a second
host-dependence problem out of the comparison.

`run_experiment.py` does not yet pass a thinking-off flag. Gemini 2.5 Flash needs
`reasoning: {max_tokens: 0}`, which is a one-line addition before running it.
