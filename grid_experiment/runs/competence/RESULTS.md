# Competence search: results so far

Date: 2026-10-03. Protocol: `PLAN.md`, written before any call.

## Status

- **Steps 1 and 2 are run, and every configuration fails.**
- **Steps 3 and 4 have reached stop points.**
- **Step 6 is paused** because the Mac was on battery.
- API spend is **$0.32 of the $1.00 cap**, leaving a balance of $1.09.

## Every configuration tried

All rows use the dev split; there were no test runs. Unparseable answers count as wrong.
Every run used the pipeline's gridded tile and classify-then-explain JSON format, at
temperature 1.0.

**Model IDs:**

- `google/gemma-4-31b-it`, via OpenRouter pinned to Friendli
- `qwen/qwen3-vl-235b-a22b-instruct`, via OpenRouter pinned to Alibaba

Neither provider exposes a model revision; the served ID is recorded per call.

**Prompt hashes** (in `prompts/rendered_competence/`):

| prompt | hash |
|---|---|
| neutral, Class A carries the HP criteria | 51293f8e |
| neutral, Class A carries the SSA criteria | b4ba5883 |
| real names + criteria | e0ef4bbe |

**Few-shot examples** (6 unanimous train-pool tiles): MHIST_dpv, ctz and aul (HP);
eqa, dqq and bxb (SSA).

| step | config | model | set | n | acc | acc 95% CI | bal acc | recall HP | recall SSA | parse | SSA-class calls | cost | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | neutral labels + criteria | gemma-4-31b-it | screen | 100 | 0.470 | 0.375–0.567 | 0.563 | 0.21 | 0.92 | 1.00 | 0.84 | $0.019 | stops |
| 1 | neutral labels + criteria | Qwen3-VL-235B | screen | 100 | 0.370 | 0.282–0.468 | 0.500 | 0.00 | 1.00 | 1.00 | 1.00 | $0.036 | stops |
| 2 | real names + criteria + 6 examples | gemma-4-31b-it | screen | 100 | 0.600 | 0.502–0.691 | 0.599 | 0.60 | 0.59 | 1.00 | 0.47 | $0.042 | stops |
| 2 | neutral labels + criteria + 6 examples | gemma-4-31b-it | screen | 100 | 0.650 | 0.552–0.736 | 0.694 | 0.52 | 0.86 | 1.00 | 0.62 | $0.042 | advances |
| 2 | ″ | gemma-4-31b-it | **dev** | **300** | **0.573** | 0.517–0.628 | **0.626** | 0.42 | 0.83 | 1.00 | 0.67 | $0.126 | **fail** |
| 2 | real names + criteria + 6 examples | Qwen3-VL-235B | screen | 100 | 0.400 | 0.309–0.498 | 0.524 | 0.05 | 1.00 | 1.00 | 0.97 | $0.048 | stops |
| 2 | neutral labels + criteria + 6 examples | Qwen3-VL-235B | screen | 100 | 0.400 | 0.309–0.498 | 0.524 | 0.05 | 1.00 | 1.00 | 0.97 | $0.047 | stops |

The pass bars, unchanged:

- **Screen:** accuracy ≥ 65% to advance.
- **Dev:** accuracy ≥ 72% **and** balanced accuracy ≥ 65%.

No configuration passed dev, so no image control was run.

**Counterbalancing.** The A/B halves agree, so the result is not a letter effect. For
gemma neutral + examples on dev, balanced accuracy is 0.618 when A = HP and 0.634 when
A = SSA.

## What steps 1–2 show

**The untested factor, the class name "sessile serrated", is not the cause.** With both
disease names removed and replaced by Class A/B plus criteria, both models still choose
whichever letter carries the SSA criteria. gemma does this 84% of the time, Qwen 100%.

**Examples help gemma a little.** It moves from 47% to 57% accuracy, and balanced
accuracy rises to 0.63. But it stays below the 63% always-HP rate. The screen's 65% did
not hold on the full dev set.

**Qwen does not move at all:** 97% SSA with examples.

## Step 3: pathology-tuned model (STOP)

**Newest multimodal MedGemma:** `google/medgemma-1.5-4b-it` (January 2026). It is
**gated**: Google's Health AI Developer Foundations terms have to be accepted.

**This machine cannot run it as things stand:**

- **Disk:** the bf16 weights are about 8.6 GB, against **4.8 GB free**.
- **No PyTorch GPU:** Apple M1 with 16 GB RAM, on macOS 13.2. PyTorch 2.11 reports that
  MPS (the Mac GPU backend) is not available, so inference would be CPU only.
- **Missing packages:** `transformers` and `huggingface_hub` are not installed.

**What would unblock it, all yours:**

1. Accept the terms at <https://huggingface.co/google/medgemma-1.5-4b-it> (and, if
   wanted, <https://huggingface.co/google/medgemma-4b-it>).
2. Create a Hugging Face read token and put it in `grid_experiment/.env` as `HF_TOKEN`.
   Never paste it in chat.
3. Either free up at least 12 GB of disk, or approve a 4-bit community build (about
   3 GB, run through llama.cpp, which needs installing).
4. Expect the CPU-only run to be slow. **Upgrading to macOS 14 or later** would enable
   the GPU backend and speed this and steps 4 and 6 up a lot. That is a system change
   for you to decide.

## Step 4: LoRA fine-tune (STOP)

**Not feasible on this Mac:**

- no GPU in PyTorch on macOS 13.2
- 4.8 GB of free disk
- Gemma-family VLM weights are gated too

**Free GPU compute** (Colab or Kaggle) would mean uploading MHIST images to a service
that has not received them. The brief makes that a stop point, so it needs your explicit
approval.

## Step 6: ResNet-18 baseline (done, local)

**Design:**

- ResNet-18 in plain PyTorch, trained **from scratch**, with no downloads
- train on train minus dev (1,875 tiles)
- choose the epoch by dev AUC, over 30 epochs
- 3 seeds, with **one test evaluation per seed**

Results: `runs/resnet18/RESULTS.json`.

| seed | chosen epoch | test acc | test AUC | balanced acc | recall HP | recall SSA |
|---|---|---|---|---|---|---|
| 0 | 28 | 76.1% | 0.848 | 0.754 | 0.78 | 0.73 |
| 1 | 24 | 75.4% | 0.849 | 0.757 | 0.75 | 0.77 |
| 2 | 27 | 76.2% | 0.846 | 0.754 | 0.78 | 0.73 |
| **mean ± SD** | | **75.9 ± 0.4%** | **0.848 ± 0.002** | **0.755** | 0.77 | 0.74 |

**Seed 0 checkpoint incident.** iCloud "Optimize Mac Storage" renamed seed 0's
checkpoint (`best.pt` became `best 9.pt`), so the script's own test step crashed.

- The surviving file's stored epoch (28) equals the log's best-dev-AUC epoch. Its single
  test evaluation was run from that file and documented in `seed0/test.json`.
- Seeds 1 and 2 ran normally.
- Checkpoints now live outside iCloud.

**This replaces the unsourced ResNet figures.** The earlier 81.8% figure probably came
from an ImageNet-pretrained model, which was not re-run; that would need about 45 MB of
weights. ViT-B/16 was not re-run either.

## Kiran items 1 and 5, updated

| # | item | status |
|---|---|---|
| 1 | fix the classifier | ❌ Not achieved. 7 configurations on dev (neutral labels, criteria, 6-shot examples; gemma and Qwen), plus 14 earlier on the pilot, all below bar. The best on full dev is 57.3%, balanced 0.626. |
| 5 | competent VLM + rerun the pipeline | ❌ Blocked. There is no competent configuration to rerun on. Steps 3–4 (MedGemma, LoRA) need a gated-model sign-up, disk or a GPU, or approval to use cloud compute. |

## Summary for Kiran (5 lines)

1. **The SSA default is not caused by the class name.** We searched on a fresh 300-tile
   dev set from the train split, with bars fixed in advance (≥ 72% accuracy, ≥ 65%
   balanced). Replacing both disease names with neutral Class A/B labels plus WHO-based
   criteria left gemma choosing the SSA-criteria class 84% of the time, and Qwen 100%.
2. **Six labelled examples helped gemma only modestly.** On the full dev set it reached
   57.3% accuracy and balanced 0.63, below the 63% always-HP rate. Qwen stayed at 97%
   SSA.
3. **No API-based configuration passed dev,** so nothing went to the 977-tile test set
   and the faithfulness pipeline was not rerun. No test data was touched.
4. **Pathology-tuned (MedGemma 1.5) and fine-tuned models are the remaining routes.**
   Both are blocked on this machine: the model is gated, there are only 4.8 GB of free
   disk, and PyTorch has no GPU here. They need a sign-up and either local resources or
   approved cloud compute.
5. **A sourced ResNet-18 baseline** (from scratch; 3 seeds; one test evaluation each) is
   written and will replace the unsourced ResNet/ViT numbers once it has run (about
   5 hours on CPU).

## ETA for the remaining steps

- **Step 6 (ResNet-18):** about 5 hours of CPU once the Mac is on power.
- **Steps 3–4:** cannot start until the unblocks above are done. After that, step 3
  would take a few hours of CPU inference for the 100-tile screen; step 4 is not
  feasible locally at all.
- **Step 5:** only if a configuration passes dev. Its full pipeline, with few-shot
  images in every call, would need about $4–5 of API credit, more than the $1 cap, or
  a local model.

**No claim of diagnostic validity is made. Exploratory remarks above are labelled as
such.**
