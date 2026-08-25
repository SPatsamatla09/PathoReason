# Full-MHIST Track-A run with GPT-4o citations

Run all commands from the PathoReason repository root.

## 1. Put the full citation file in the repository

Copy `cte_p1_seed42_v2.jsonl` from Sid's handoff to:

```text
grid_experiment/runs/gpt4o_cte_p1_full.jsonl
```

The modified runner accepts both the original Gemma citation schema and Sid's
full GPT-4o schema. The supplied full file contains 2,498 image-feature jobs
across 977 unique MHIST test images.

## 2. Recover the local MHIST paths if needed

```bash
history | grep run_track_a_faithfulness | tail -5
```

Use the same `--images-dir` and `--annotations` paths that worked previously.

## 3. One-job smoke test

```bash
python3 grid_experiment/run_track_a_faithfulness.py \
  --images-dir "/actual/path/to/MHIST/images" \
  --annotations "/actual/path/to/MHIST/annotations.csv" \
  --citations grid_experiment/runs/gpt4o_cte_p1_full.jsonl \
  --limit 1 \
  --out grid_experiment/runs/track_a_gpt4o_full_smoke.jsonl
```

Successful output ends with `[1/1] ...` and:

```text
done -> grid_experiment/runs/track_a_gpt4o_full_smoke.jsonl
```

Check that records were written:

```bash
wc -l grid_experiment/runs/track_a_gpt4o_full_smoke.jsonl
```

## 4. Full resumable run

Keep the Mac connected to power and do not close the lid.

```bash
caffeinate -i python3 grid_experiment/run_track_a_faithfulness.py \
  --images-dir "/actual/path/to/MHIST/images" \
  --annotations "/actual/path/to/MHIST/annotations.csv" \
  --citations grid_experiment/runs/gpt4o_cte_p1_full.jsonl \
  --out grid_experiment/runs/track_a_faithfulness_gpt4o_full.jsonl
```

The output is append-only and resumable. Re-running the exact command skips
completed records. Do not use the old `track_a_faithfulness.jsonl` output name;
that file contains the earlier 20-image Gemma-conditioned run.

## 5. Analyze the completed full run

```bash
python3 grid_experiment/analyze_track_a_faithfulness.py \
  --run grid_experiment/runs/track_a_faithfulness_gpt4o_full.jsonl \
  --out grid_experiment/runs/track_a_faithfulness_gpt4o_full
```

