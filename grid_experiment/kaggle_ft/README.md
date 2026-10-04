# Step 4 on Kaggle, privately

How to run the step-4 LoRA fine-tune and its dev evaluation (`runs/competence/PLAN.md`, "Addendum: step 4")
on a private Kaggle GPU notebook, and how its privacy is enforced and checked.

Written 2026-10-04, revised the same day after review. Nothing has been uploaded yet: the bundle is built
locally and every upload command has only been run with `--dry-run` or against an offline stand-in.

## Rules this folder is built around

- **Private only.** Every dataset and notebook is private, on your account, with no collaborators.
  Nothing here can create a public resource.
- **Internet off.** Notebooks run with internet off. If Kaggle reports it on although off was asked for, the
  tool treats that like a privacy failure.
- **No test tile leaves this machine.** The upload bundle holds the 1,875 pool tiles and the 300 dev tiles.
- **No dev or test label leaves this machine.** Labels are uploaded for pool tiles only. Scoring is local.
- **No token goes to Kaggle.** The weights are uploaded as a private dataset, so the notebook needs no
  Hugging Face token and no Kaggle secret.
- **Step order (PLAN).** Training may run while step 3 is still screening. The step-4 dev evaluation
  (sections 8-10 below) is run only if step 3 ends with no passing configuration. Dev job files cannot be
  built without `--step3-closed`.
- **One adapter on dev.** Only the final, validation-selected adapter of the finished protocol run is
  evaluated on dev. The code refuses anything else.

## Before any training: two things in the PLAN

`PLAN_AMENDMENTS_PROPOSED.md` holds text for `runs/competence/PLAN.md` that this folder's code may not write
there itself:

1. Notes on what the dev result can show (pool and dev may share slides), where the format bar is applied,
   and the choices the addendum left open. Paste them into the PLAN before training.
2. **A decision on the training target.** The PLAN's target ends in a bare quote token that a full answer
   never uses. The code keeps the PLAN's target until you decide otherwise; the file explains both options.

## What is where

| file | runs | what it does |
|---|---|---|
| `build_bundle.py` | Mac | builds the data bundle in `~/mhist_local/kaggle_bundle_mhist/` and proves it is clean |
| `kaggle_push.py` | Mac | the only thing that talks to Kaggle: uploads, pushes, downloads, privacy checks |
| `train_lora.py` | Kaggle | the trainer; every hyperparameter of the addendum is a constant in it |
| `build_dev_jobs.py` | Mac | writes label-free job files and, separately, local-only sidecars with the labels |
| `infer_jobs.py` | Kaggle | runs a job file, with or without the adapter |
| `import_results.py` | Mac | joins the downloaded answers with the local sidecars into `runs/competence/*.jsonl`; applies the format gate |
| `test_train_data_local.py`, `test_jobs_local.py`, `test_push_local.py` | Mac | CPU tests; no weights are loaded, no network is used |
| `PLAN_AMENDMENTS_PROPOSED.md` | - | draft text for the PLAN (see above) |

Outside the project, never in iCloud-synced `Documents`, never in the tree that is copied to the public
repository:

| path | content |
|---|---|
| `~/mhist_local/kaggle_bundle_mhist/` | the bundle (239 MB) |
| `~/mhist_local/kaggle_jobs/` | job files (uploaded; no labels) |
| `~/mhist_local/kaggle_local_only/` | sidecars with the dev labels. Never uploaded, never synced |
| `~/mhist_local/kaggle_stage/` | `registry.json` (what was created, when it was last confirmed private), staging copies |
| `~/mhist_local/kaggle_out/<notebook>/` | downloaded notebook outputs: adapter, logs, answer files |
| `~/mhist_local/kaggle_wheels/` | wheels for the offline package install |
| `~/mhist_local/tmp/` | the tests' temporary folders (each run its own, removed at the end) |

`kaggle_ft/` itself holds code and documentation only. The files Kaggle needs that name your account
(`dataset-metadata.json`, `kernel-metadata.json`) are written under `~/mhist_local/`, not here. A
`.gitignore` in this folder keeps adapters, weights, wheels, sidecars and job files out of a repository
even if one is copied here by mistake.

**Open point outside this folder.** `sync_to_pathoreason.py` copies everything under `grid_experiment/` to
the public repository and skips neither `*.safetensors`, `*.bin`, `*.whl` nor `*.sidecar.jsonl`. Until it
does, never copy an adapter or a sidecar anywhere under `grid_experiment/`. The suggested change is in the
last section.

## What you must do once

1. **Create the API token file.** On <https://www.kaggle.com/settings>, section *API*, under
   *Legacy API Credentials*, click *Create Legacy API Key*. The browser downloads `kaggle.json`. Then:

   ```sh
   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json
   ```

   Never paste the key into a chat, a script or a notebook. `kaggle_push.py` reads the username from this
   file and never prints the key.

   The newer settings page gives an *API token* (a single `KGAT_...` string) and no file. That works too:
   copy it and run `mkdir -p ~/.kaggle && pbpaste > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token`.
   `kaggle_push.py` accepts either file; with a token it asks Kaggle for the username once and keeps only the
   name in `~/mhist_local/kaggle_stage/account.json`. (This is what the 2026-10-04 run used.)
2. **Verify your phone number** on the same settings page. Kaggle requires it before a notebook can use a
   GPU.
3. **Check:**

   ```sh
   python3 kaggle_ft/kaggle_push.py check
   ```

   It must end with `check passed` and show the weekly GPU hours left.

## The sequence

Add `--dry-run` to any `kaggle_push.py` command to see the exact `kaggle` commands and metadata files first.
Run the two uploads (2 and 3) in an ordinary Terminal window, not inside an agent session: a long upload
is killed when such a session ends. `caffeinate -i <command>` keeps the Mac awake.

**`kernel-status` saying `complete` is not success.** It only means the script exited with status 0, and
both scripts also exit 0 after a pause or a failure that left output worth keeping (Kaggle discards the
output of a run that exits non-zero). Success is `"status": "complete"` in `train_summary.json` (training)
or in `<answers>.status.json` (inference). `kernel-output` prints both.

### 1. Build the bundle (local, offline, about 45 s)

```sh
python3 kaggle_ft/build_bundle.py            # safe to re-run; unchanged files are not rewritten
python3 kaggle_ft/build_bundle.py --check    # verifies without writing
```

Use the same `python3` that runs `comp_run.py`: the tiles are rendered by
`run_experiment.b64_gridded_tile`, so they are byte-identical to what the other steps send.

### 2. Upload it as a private dataset

```sh
python3 kaggle_ft/kaggle_push.py dataset-create        # -> <you>/mhist-priv-bundle
```

### 3. Upload the weights as a private dataset (8.6 GB, once)

```sh
caffeinate -i python3 kaggle_ft/kaggle_push.py weights # -> <you>/mhist-priv-medgemma15-4b
```

No copy is made (hard links). `.cache/` and dot-files stay local. If the upload is interrupted, run the
same command again: the Kaggle CLI resumes.

### 4. Packages on Kaggle: a private wheels dataset

The trainer needs `peft >= 0.13.0`, and `bitsandbytes >= 0.43` on a T4 (no native bf16). Neither is in
Kaggle's own package list. A pushed script has no cell to run `pip` in, so `kernel-push` installs them
before the script starts, offline, from a private dataset of wheels:

```sh
~/mhist_local/venv/bin/pip download --no-deps --only-binary=:all: \
    --platform manylinux_2_24_x86_64 --platform manylinux_2_28_x86_64 --platform manylinux2014_x86_64 \
    --python-version 3.12 -d ~/mhist_local/kaggle_wheels \
    peft==0.21.2 bitsandbytes==0.50.2 accelerate==1.15.0
python3 kaggle_ft/kaggle_push.py dataset-create --kind wheels --dir ~/mhist_local/kaggle_wheels
```

Then add to every push below:
`--dataset wheels --pip-install 'peft==0.21.2 bitsandbytes==0.50.2 accelerate==1.15.0' --pip-find-links '{wheels}'`

- The versions are the newest on PyPI on 2026-10-04. The `bitsandbytes` wheel is tagged
  `manylinux_2_24_x86_64`; without that `--platform`, pip finds no `bitsandbytes` at all. The other two are
  pure Python. The download itself has not been run yet.
- The probe in 5a prints which versions the Kaggle image already has. If `pip` then reports a conflict with
  the image's `torch` or `transformers`, change the pins, download again and upload with `dataset-version`.
- Internet on is possible but not recommended: `--enable-internet true --i-accept-internet`. The second flag
  is required, because the script and everything pip installs can then reach the network while the tiles
  and the weights are mounted.

### 5. Train

**The three smoke runs (a, b, c) are required before the protocol run.** The local tests use a stand-in for
`peft` and have no `bitsandbytes` or GPU, so the 4-bit load, real PEFT on 238 layers, gradient checkpointing
on a GPU, the cached-feature probe on the real model and `generate()` on Kaggle's `transformers` version
have never run. Together the three should take under an hour of GPU quota (an estimate, not measured).

a. **Probe (about a minute).** `--verify-only` checks the bundle, the split, the packages and the
   tokenisation on Kaggle and loads no weights. The log also lists `/kaggle/input`.

   ```sh
   python3 kaggle_ft/kaggle_push.py kernel-push --script kaggle_ft/train_lora.py --slug mhist-priv-probe \
       --dataset bundle --dataset weights \
       --args='--verify-only --model-dir {weights} --data-dir {bundle}'
   python3 kaggle_ft/kaggle_push.py kernel-status --slug mhist-priv-probe --wait
   python3 kaggle_ft/kaggle_push.py kernel-output --slug mhist-priv-probe    # read mhist-priv-probe.log
   ```

   The first log line of the trainer is `arguments as parsed: verify_only=True ...`. If it says
   `verify_only=False`, stop the run at once: the arguments did not arrive.

b. **A tiny training run (32 tiles, 1 epoch).** It exercises the 4-bit load, PEFT, the feature cache, the
   loop, validation and the adapter files.

   ```sh
   python3 kaggle_ft/kaggle_push.py kernel-push --script kaggle_ft/train_lora.py --slug mhist-priv-smoke-train \
       --dataset bundle --dataset weights \
       --args='--model-dir {weights} --data-dir {bundle} --out-dir /kaggle/working/smoke --debug-limit-train 32 --debug-limit-val 16 --debug-epochs 1'
   python3 kaggle_ft/kaggle_push.py kernel-status --slug mhist-priv-smoke-train --wait
   python3 kaggle_ft/kaggle_push.py kernel-output --slug mhist-priv-smoke-train
   ```

   Read in the log and in `smoke/train_summary.json`: `sec_per_example` (the full run is 6,376 examples plus
   five validation passes; it must fit the session, see "A second training session"), `gpu_mem_gb`,
   `image_input.mode` (cached features, or the slower pixel path) and `precision.n_linear4bit_modules`
   (238). The run is marked `protocol_run: false`; its adapter can never be scored on dev.

c. **The 20 smoke tiles through the inference script, with that adapter.** Pool tiles; no label is used.

   ```sh
   ~/mhist_local/venv/bin/python kaggle_ft/build_dev_jobs.py --config cte_p1 --tiles smoke
   python3 kaggle_ft/kaggle_push.py dataset-create --kind jobs --dir ~/mhist_local/kaggle_jobs
   python3 kaggle_ft/kaggle_push.py kernel-push --script kaggle_ft/infer_jobs.py --slug mhist-priv-smoke-infer \
       --dataset bundle --dataset weights --dataset jobs --kernel-source mhist-priv-smoke-train \
       --args='--jobs {jobs}/smoke__cte_p1__none__tier1.jsonl --bundle-dir {bundle} --model-dir {weights} --adapter-dir {input:mhist-priv-smoke-train}/smoke/best_adapter'
   python3 kaggle_ft/kaggle_push.py kernel-status --slug mhist-priv-smoke-infer --wait
   python3 kaggle_ft/kaggle_push.py kernel-output --slug mhist-priv-smoke-infer
   python3 kaggle_ft/import_results.py --dry-run --model-tag smoke \
       --train-summary ~/mhist_local/kaggle_out/mhist-priv-smoke-train/smoke/train_summary.json \
       --results ~/mhist_local/kaggle_out/mhist-priv-smoke-infer/smoke__cte_p1__none__tier1__lora-*.jsonl
   ```

   This gives the real tokens per second and shows that generation, the thinking-token ban and the import
   work end to end. An adapter trained on 32 tiles may well fail the format check printed here; what matters
   at this point is that the chain runs.

d. **The protocol run.** `kernel-push` starts it; this spends GPU quota (`kaggle_push.py quota`).

   ```sh
   python3 kaggle_ft/kaggle_push.py kernel-push --script kaggle_ft/train_lora.py --slug mhist-priv-train \
       --dataset bundle --dataset weights \
       --args='--model-dir {weights} --data-dir {bundle} --out-dir /kaggle/working/step4_lora'
   python3 kaggle_ft/kaggle_push.py kernel-status --slug train --wait
   python3 kaggle_ft/kaggle_push.py kernel-output --slug train \
       --file-pattern 'best_adapter|train_summary|train_log|train_steps|split\.json|val_scores|FAILED'
   ```

   `{bundle}` and `{weights}` expand to `/kaggle/input/mhist-priv-bundle` and
   `/kaggle/input/mhist-priv-medgemma15-4b`. A script notebook is started without arguments, so
   `kernel-push` inserts a short header into the pushed copy that sets them; the dry run prints it.

**A second training session.** The trainer stops by itself after 11 hours with a checkpoint
(`train_summary.json` says `"status": "paused"`). One T4 session may not be enough. To continue, push to a
**new** slug with the earlier notebook's output attached (a new slug, so that the earlier output stays
available as an input and is not replaced by the new run):

```sh
python3 kaggle_ft/kaggle_push.py kernel-push --script kaggle_ft/train_lora.py --slug mhist-priv-train2 \
    --dataset bundle --dataset weights --kernel-source train \
    --args='--model-dir {weights} --data-dir {bundle} --out-dir /kaggle/working/step4_lora'
```

`--resume-from auto` (the default) finds `/kaggle/input/mhist-priv-train/step4_lora/checkpoints/last` and
continues from it; the result is identical to an uninterrupted run. Everything afterwards then uses
`mhist-priv-train2` where this page says `train`. A paused run's `best_adapter/` is the best so far, is
marked `"final": false`, and is refused for dev.

### 6. Fetch the adapter

After 5d, `~/mhist_local/kaggle_out/mhist-priv-train/step4_lora/` holds the run. Read `train_summary.json`:
`status` must be `complete`, `protocol_run` true. Then pin the chosen adapter as its own private dataset,
so the evaluation does not depend on a notebook version:

```sh
python3 kaggle_ft/kaggle_push.py dataset-create --kind adapter \
    --dir ~/mhist_local/kaggle_out/mhist-priv-train/step4_lora/best_adapter      # -> <you>/mhist-priv-adapter
```

(Shortcut without the upload: `--kernel-source train` and `--adapter-dir {train}/step4_lora/best_adapter`.)

`train_summary.json` also reports `expected_tier1_accuracy` per epoch. It is for information only and is
not used to choose the epoch: tier 1 samples the label, so its dev accuracy can sit below the argmax
accuracy that the validation log shows.

### 7. Format check before dev (label-free, required)

The final adapter writes the full `cte_p1` answer for the 20 smoke tiles (pool tiles; no label is used).
This is the check the step-3 addendum already permits. It must pass before any dev tile is shown to the
adapter, because the training target supervises nothing after the label.

```sh
python3 kaggle_ft/kaggle_push.py kernel-push --script kaggle_ft/infer_jobs.py --slug mhist-priv-format-check \
    --dataset bundle --dataset weights --dataset jobs --dataset adapter \
    --args='--jobs {jobs}/smoke__cte_p1__none__tier1.jsonl --bundle-dir {bundle} --model-dir {weights} --adapter-dir {adapter}'
python3 kaggle_ft/kaggle_push.py kernel-status --slug mhist-priv-format-check --wait
python3 kaggle_ft/kaggle_push.py kernel-output --slug mhist-priv-format-check
python3 kaggle_ft/import_results.py --dry-run --model-tag medgemma-1.5-4b-it-lora-r16 \
    --train-summary ~/mhist_local/kaggle_out/mhist-priv-train/step4_lora/train_summary.json \
    --results ~/mhist_local/kaggle_out/mhist-priv-format-check/smoke__cte_p1__none__tier1__lora-*.jsonl
```

It must print `SMOKE FORMAT CHECK ... -> ok`: every answer ends with a normal stop, and at least 90% have a
valid label and a valid grid cell. Otherwise the exit code is 1 and the dev jobs are not run with this
adapter (see `PLAN_AMENDMENTS_PROPOSED.md`, section C, for what a retrain would change).

### 8. Dev jobs (only if step 3 ended with no passing configuration)

```sh
~/mhist_local/venv/bin/python kaggle_ft/build_dev_jobs.py --config cte_p1 --control none noimage mismatch --tier 1 2 --step3-closed
python3 kaggle_ft/kaggle_push.py dataset-version --kind jobs --dir ~/mhist_local/kaggle_jobs -m 'dev jobs'
```

`--step3-closed` is your statement that step 3 has ended without a pass; without it no dev job file is
built. Job files carry prompts, tile names and seeds, and no label. The sidecars with the labels are written
to `~/mhist_local/kaggle_local_only/` and are never uploaded: a `jobs` upload may hold nothing but valid
job files. Building all six job files is not running them; the PLAN decides which run.

### 9. Inference notebook, one job file per notebook

```sh
python3 kaggle_ft/kaggle_push.py kernel-push --script kaggle_ft/infer_jobs.py --slug mhist-priv-infer-none-t1 \
    --dataset bundle --dataset weights --dataset jobs --dataset adapter \
    --args='--jobs {jobs}/dev__cte_p1__none__tier1.jsonl --bundle-dir {bundle} --model-dir {weights} --adapter-dir {adapter}'
python3 kaggle_ft/kaggle_push.py kernel-status --slug mhist-priv-infer-none-t1 --wait
python3 kaggle_ft/kaggle_push.py kernel-output --slug mhist-priv-infer-none-t1 --file-pattern '\.jsonl$|status'
```

- Use a new slug for each job file (`...-noimage-t1`, `...-mismatch-t1`, `...-none-t2`): `kernel-output`
  returns the latest run of a notebook, so re-using a slug before downloading loses the earlier answers.
- `infer_jobs.py` runs dev jobs only with an adapter whose `step4_meta.json` says `protocol_run: true` and
  `final: true`. Anything else stops before a dev tile is read.
- Add the same `--dataset wheels --pip-install ... --pip-find-links '{wheels}'` options as for training.
  Run one GPU notebook at a time.
- Order (PLAN): tier 1 first; the two image controls only for a configuration that passes dev; tier 2 only
  if tier 1 fails.

### 10. Import and score (local)

```sh
python3 kaggle_ft/import_results.py --model-tag medgemma-1.5-4b-it-lora-r16 \
    --train-summary ~/mhist_local/kaggle_out/mhist-priv-train/step4_lora/train_summary.json \
    --results ~/mhist_local/kaggle_out/mhist-priv-infer-none-t1/dev__cte_p1__none__tier1__lora-*.jsonl
python3 comp_analyze.py
```

`--train-summary` ties the answers to the finished training run: the summary must say `complete` and
`protocol_run: true`, and every answer must come from its final adapter. Each dev import is logged in
`runs/competence/step4_imports.ndjson`; a second, different adapter for the same run is refused.

**A step-4 configuration passes dev only if all three hold:**

1. `comp_analyze.py` prints `PASS` for it (accuracy at least 72% and balanced accuracy at least 65%);
2. `runs/competence/step4_gate__cte_p1__<model-tag>__<tier>.json` says `"format_ok": true` (at least 90% of
   the 300 answers have a valid label and a valid grid cell). `import_results.py` prints this as
   `STEP-4 FORMAT GATE`. `comp_analyze.py` does not apply it, so its `PASS` alone is not enough;
3. both image controls are `control OK`.

### 11. Confirm that everything is still private

```sh
python3 kaggle_ft/kaggle_push.py verify-private
```

Run it after the last push, and again whenever you have touched anything on kaggle.com. It also fails if a
notebook has internet on that was not pushed that way. Exit codes: 0 all private; 1 an upload that Kaggle is
still creating (`PENDING`: not confirmed, run it again in a few minutes); 3 the privacy alarm.

## Privacy: what is guaranteed, and how it is checked

| guarantee | how it is enforced | how it is checked |
|---|---|---|
| datasets are private | `kaggle datasets create` is private unless `-u/--public` is passed; `kaggle_push.py` never passes it and refuses any command line containing `-u`, `--public` or `--update` | the dataset is entered in the registry before the upload starts. Right after the upload, and again after Kaggle has processed it, `kaggle datasets metadata <ref>` must return `"isPrivate": true` and no collaborators, else exit code 3. A processing error is reported only after that check |
| notebooks are private | `kernel-metadata.json` always has `"is_private": "true"`; an existing notebook with the same slug must already be private, or nothing is pushed | after every push: `kaggle kernels pull <ref> -m` must return `"is_private": true`, else exit code 3 |
| notebooks have no internet | `"enable_internet": "false"` unless you pass both `--enable-internet true` and `--i-accept-internet` | after every push Kaggle must report `false`, else exit code 3 and the notebook is not recorded as confirmed (a missing value is not taken on trust); `verify-private` checks it again |
| nothing drifts later | - | `verify-private` re-checks everything in the registry and every dataset / notebook on the account whose slug starts with `mhist-priv-` (`--all`: everything on the account). An unanswered metadata request counts as a failure. It also requests each page with no credentials; a page that is served although a made-up slug is not counts as visible |
| a notebook only reads private data | a push first checks that every attached dataset / notebook of yours is private | same commands as above |
| no test tile is uploaded | the bundle is built from `fewshot_pool` + `dev` only, each asserted to be in the train partition | before every upload: no file, archive member or text may name a test-partition tile (exit code 2) |
| no dev / test label is uploaded | uploads are allow-listed by kind (`--kind`): the bundle must match `MANIFEST.json` file for file; `jobs` may hold only `*.jsonl` whose every line is a valid five-key job; `adapter` exactly the adapter files; `wheels` only `*.whl`. There is no kind for arbitrary files, and sidecars are refused by name | before every upload, and before a folder is copied to the staging area: no line may pair a dev tile with a label; no table may name a dev tile except the plain `image,subset` list; no JSON document may name a dev tile and hold a label anywhere; text files inside archives are read too (exit code 2) |
| the bundle is exactly what was built | `MANIFEST.json` lists sha256 and size of every file | before every upload the directory must match it file for file. `dataset-metadata.json` and `.DS_Store` are tolerated at the top level only, where the Kaggle CLI skips them |
| no credential is uploaded | the tool reads keys only to compare, and prints only the kind of key it found | scripts, arguments, `--env` values and text files are scanned for key formats (Hugging Face, Kaggle, OpenRouter, Anthropic, OpenAI, Google, Friendli, GitHub, AWS, Slack, private-key blocks) and for this machine's own keys: `kaggle.json`, `KAGGLE_KEY`, `KAGGLE_API_TOKEN`, `~/.kaggle/access_token` and the values in the project's `.env`. `--env` names that look like a credential are refused |
| nothing public can be run by this tool | allow-list of `kaggle` commands: no `delete`, no `metadata --update`, no `--public` | `Runner._guard` |
| dev is evaluated with one adapter | only `best_adapter/` of a finished protocol run is marked `final`; `infer_jobs.py` refuses anything else for dev jobs | `import_results.py` needs `--train-summary` (or `--base-model`), compares the adapter hash, and logs every dev import |

If a check fails, the tool prints `PRIVACY CHECK FAILED - ACT NOW` with the page to open and stops. It does
not delete or stop anything on its own: with internet reported on, stop the session on kaggle.com yourself.

Two limits of the checks. Kaggle's API does not report a notebook's collaborators, so "no collaborators" is
checked for datasets only; for notebooks, look at *Share* on the notebook page. And the anonymous page check
assumes that Kaggle answers 404 for a private page, as it does for a page that does not exist; public pages
were seen to answer 200 and made-up ones 404 on 2026-10-04, a private page has not been tried yet.

**Do not do these by hand:**

- `kaggle datasets create -u` / `--public`.
- `kaggle datasets metadata <ref> --update`: the CLI sets a dataset to **public** when the local metadata
  file has no `"isPrivate": true` (kaggle 2.2.4, `dataset_metadata_update`). The metadata files written by
  this tool carry that key for this reason.
- `kaggle kernels push` on a hand-written metadata file without `"enable_internet"`: the installed client
  defaults it to **on**, although the documentation says off.
- On kaggle.com: do not change *Visibility* or *Share*, do not add collaborators, do not switch *Internet*
  on, do not attach these datasets to any other notebook.

## Protocol notes

- **The adapter that goes to dev.** `train_lora.py` writes an adapter after every epoch, and `best_adapter/`
  as soon as one epoch is best so far. Only when all four epochs are validated is `best_adapter/` marked
  `"final": true` with its sha256, and only then does `train_summary.json` say `complete`. `infer_jobs.py`
  and `import_results.py` both insist on that mark for dev tiles.
- **4-bit scope.** On a T4 the language model's 238 linear layers are loaded 4-bit, because float32 they do
  not fit. The frozen SigLIP vision tower, the projector, the embeddings and `lm_head` are not quantised
  (an estimated 6.5 GB of GPU memory in total; 5b shows the real figure). The loaded model is checked for
  exactly that, in training and in evaluation, and an adapter trained with another scope is refused.
  With `transformers` 5.18 the obvious skip list (`vision_tower`) quantises the vision tower all the same.
  The scripts pass a list that works under both matching rules, and the check on the loaded model catches
  a version where it does not.
- **Training target.** `TARGET_VARIANT` in `train_lora.py` is `"plan"`: the addendum's text, 9 tokens, the
  last a bare quote. See `PLAN_AMENDMENTS_PROPOSED.md`, section C.
- **Choices the addendum left open** are constants in `train_lora.py` and are written into
  `train_summary.json` (`choices_not_fixed_by_the_addendum`): gradient clipping at 0.3, AdamW betas 0.9 /
  0.999, eps 1e-8, weight decay 0; loss = mean cross-entropy over the target tokens times the class weight;
  `lm_head` not adapted; ties go to the earlier epoch. `PLAN_AMENDMENTS_PROPOSED.md` has them as PLAN text.
- **Pool and dev may share slides** (the split is per tile; there is no slide id). The dev result is a gate;
  the test run is the evidence. No two tiles of pool, dev and test have identical pixels (checked 2026-10-04).

## Kaggle facts this relies on (checked 2026-10-04)

- **Metadata keys.** Datasets: `title` (6-50 characters), `id` (`owner/slug`), `licenses` (exactly one),
  optional `subtitle` (20-80), `description`, `keywords`, `resources`. There is no privacy key for
  `create`; privacy is the absence of `--public`. Notebooks: `id`, `title`, `code_file`, `language`,
  `kernel_type`, `is_private`, `enable_gpu`, `enable_internet`, `machine_shape`, `dataset_sources`,
  `competition_sources`, `kernel_sources`, `model_sources`; the client also reads `enable_tpu`,
  `docker_image`, `docker_image_pinning_type`.
  [datasets_metadata.md](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets_metadata.md),
  [kernels_metadata.md](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md),
  installed source `kaggle/api/kaggle_api_extended.py` (2.2.4).
- **Directories must be archived.** `--dir-mode` is `skip` by default: sub-directories are silently left
  out. `zip` uploads each as an archive that Kaggle unpacks. A dataset may have at most 50 top-level files,
  200 GB, and 200 GB of private data per account. So the 2,175 tiles go up as `gridded.zip`. Only the
  bundle is uploaded with `zip`; the flat kinds use `skip`.
  [datasets.md](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md),
  [Kaggle datasets documentation](https://www.kaggle.com/docs/datasets).
- **GPU.** `"enable_gpu": "true"` plus `"machine_shape": "NvidiaTeslaT4"` is *GPU T4 x2*, the default GPU.
  The P100 was retired on 2026-09-14: `NvidiaTeslaP100` now silently runs on the T4, and before that the
  image's torch build could not use it. `kaggle_push.py` refuses it.
  [kernels.md](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md),
  [PR 1192](https://github.com/Kaggle/kaggle-cli/pull/1192),
  [issue 1151](https://github.com/Kaggle/kaggle-cli/issues/1151),
  [issue 1196](https://github.com/Kaggle/kaggle-cli/issues/1196).
- **Limits.** 12 h per GPU session, 20 GB in `/kaggle/working`, about 30 GB RAM, about 30 GPU hours per
  week (the quota floats; `kaggle_push.py quota` shows yours). Phone verification is required for GPU and
  internet. [Kaggle notebooks documentation](https://www.kaggle.com/docs/notebooks).
- **A private notebook can attach your own private datasets** through `dataset_sources`, and another of
  your notebooks' output through `kernel_sources`. Kaggle reports rejected sources in the push response;
  `kernel-push` also compares the attached list afterwards.
- **Kaggle Secrets cannot be set or attached through the API or CLI.** They are defined in the notebook
  editor only ([kernels.md, "Using Secrets in Kernels"](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md),
  [issue 582](https://github.com/Kaggle/kaggle-cli/issues/582), open). So the PLAN's other option
  (download the weights inside the notebook with a token held as a secret) cannot be driven from here; the
  private weights dataset is used instead.
- **MedGemma on Kaggle Models.** Google lists Hugging Face and Vertex AI Model Garden as the sources;
  there is no `google/medgemma` model page on Kaggle. A Keras port exists (`keras/medgemma`, preset
  `medgemma_1.5_instruct_4b`): KerasHub format, not the pinned Hugging Face revision, so it cannot replace
  the upload for this protocol. Its consent flow could not be read (the page needs a browser).
  [MedGemma get started](https://developers.google.com/health-ai-developer-foundations/medgemma/get-started),
  [KerasHub presets](https://keras.io/keras_hub/presets/).
- **A new dataset is invisible to its owner while Kaggle creates it** (seen 2026-10-04). After
  `datasets create` returned "Your private Dataset is being created", `datasets status`, `metadata` and
  `files` answered **403 to the owner's own token** for 15 minutes (228 MB bundle), the dataset was not in
  `datasets list --mine`, and its page without credentials was 404. A made-up slug gives the owner the same
  403. Then status became `ready` and the metadata said `isPrivate: true`. A 45 MB dataset was ready at
  once. `kaggle_push.py` therefore waits: it treats "the owner cannot read it yet" as *pending* only when
  the page without credentials answers 404 **and** a made-up slug answers 404 (any other answer is the full
  alarm), checks privacy the moment Kaggle first answers, and never records a pending dataset as confirmed.
  If the wait runs out the command ends with exit code 1 and "privacy is NOT confirmed"; run the same
  command again (it waits, it does not upload twice) or `verify-private`, which reports such a dataset as
  `PENDING` (exit code 1, not the alarm).
- **Where inputs are mounted** (seen 2026-10-04). Attached datasets are at
  `/kaggle/input/datasets/<owner>/<slug>/`, not `/kaggle/input/<slug>/`. The header that `kernel-push`
  inserts therefore resolves every `/kaggle/input/<slug>[/...]` path at run time to the one attached
  directory of that name (at most 4 levels down; the flat layout still works) and stops the script if there
  is none or more than one. The image has Python 3.13, torch 2.11.0+cu128 and transformers 5.16.1; the three
  wheels install offline without changes.
- **Measured on the T4 (smoke run, 32 tiles):** 5.8 s per training example at micro-batch 2 (4-bit nf4,
  float32 compute, gradient checkpointing), 10.0 GB of GPU memory, 1.9 s per validation tile, 2.2 s per tile
  to cache image features. The protocol run (6,376 examples, five validation passes, 1,875 feature
  extractions) is therefore about 12.5 hours: one session pauses at 11 hours and a second one finishes it.
  `kaggle_ft/run_train_driver.sh` waits for the run and starts the next session by itself.
- **Status words.** `datasets status` prints `ready`, `failed`, `deleted`, or a processing state
  (`not_yet_persisted`, `blobs_received`, ...). `failed` and `deleted` end the wait as a processing error.
- **CLI version.** Tested against kaggle 2.2.4 (the newest release on PyPI). The documentation on `main`
  is ahead of it (`--no-run`, warnings for retired accelerators).
- **How Kaggle starts a script notebook** (plain `python script.py`, or inside a notebook kernel) is not
  known. Both scripts handle both: the pushed header replaces the whole of `sys.argv`, and neither script
  calls `sys.exit(0)`.

## Licence notes

- **MHIST.** The tiles are used under the MHIST research use agreement. Private Kaggle storage on your own
  account is the only copy outside this Mac. No test tile is uploaded.
- **MedGemma (HAI-DEF terms).** The weights are governed by the
  [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms).
  - The private dataset is your own unmodified copy for your own use. It is not shared with anyone, so it
    is not a distribution. A `NOTICE` file with the required sentence is uploaded with it anyway.
  - A LoRA adapter is a *Model Derivative* (section 1.1). It stays private: in `~/mhist_local/kaggle_out/`
    and in the private `mhist-priv-adapter` dataset. It is never pushed to GitHub, Hugging Face or a public
    Kaggle resource. Distributing it would require passing on the use restrictions of section 3.2, a copy
    of the terms and the notice (section 3.1).
  - Model answers are *Outputs*, not Model Derivatives (section 3.3). They are what `import_results.py`
    brings back.
  - Section 3.2 and the Prohibited Use Policy apply: no clinical use. No claim of diagnostic validity is made.
  - This is a reading of the terms, not legal advice.

## Resuming, and what to do when something stops

- `build_bundle.py`, `dataset-create`, `weights`: run the same command again. Finished work is detected
  (identical files; the registry's content fingerprint; the Kaggle CLI's own upload resume).
- `dataset-create` says the dataset already exists: use `dataset-version -m '...'`.
- `kernel-push` again creates a new version of the same notebook and starts a new run. A new Kaggle session
  starts with an empty `/kaggle/working`: training continues through "A second training session" above;
  inference through `--kernel-source <earlier notebook> ... --resume-from {input:<earlier slug>}`.
- Exit code 2 (`LEAK TRIPWIRE`): the directory or script contains something that must not be uploaded.
  The message names the file and line. Nothing was sent.
- Exit code 3 (`PRIVACY CHECK FAILED`): follow the printed steps, then run `verify-private`.
- `verify-private` reports `unknown` for an entry: Kaggle did not answer. Run it again. If the entry is an
  upload that failed before Kaggle created anything, and kaggle.com shows no such dataset, remove it from
  `~/mhist_local/kaggle_stage/registry.json` by hand.
- Paths under `/kaggle/input` are wrong: read the `[kaggle_push] input ...` lines at the top of the
  notebook log and correct `--args`.

## How this was tested (2026-10-04, nothing uploaded)

Run all three with `~/mhist_local/venv/bin/python`. Each writes to a folder of its own under
`~/mhist_local/tmp/`, so they can run at the same time.

- `test_train_data_local.py` (about 3 minutes): the trainer on a tiny random Gemma 3 with the real
  processor. Split, schedule, tokenisation of both target variants, LoRA targets, the 4-bit skip list and
  scope check, resume after a pause and after a crash, a second "Kaggle session", the final-adapter mark,
  out-of-memory handling in validation, and the argument header under a notebook-kernel start.
- `test_jobs_local.py` (about 4 minutes): job files against a captured `comp_run.py` run, token ids, the
  leak rules, the import with every refusal (adapter provenance, sidecar integrity, image verification, the
  ledger), the format gate, and `infer_jobs.py` end to end on a tiny model.
- `test_push_local.py` (about half a minute): `kaggle_push.py` against an offline stand-in for the Kaggle
  CLI. Datasets reported public or shared, a notebook reported public, internet reported on, a processing
  error, a client failure after the server created the dataset, refusal of `--public` / `--update` /
  `delete`, the secret scan, the upload kinds and ten ways a label could ride along.
- `build_bundle.py`: built twice and with a second interpreter, same `MANIFEST.json` hash
  (`dbf2197a...ecd4`), 0 files rewritten on the second run.
- `kaggle_push.py`: every subcommand run with `--dry-run`.
- **Not tested, because it needs the network or a GPU:** the real upload, the real metadata answers, the
  mount layout under `/kaggle/input`, which packages the Kaggle image has, real `peft` and `bitsandbytes`,
  and the 4-bit load. Sections 5a to 5c exist to find these out cheaply.

## Suggested change outside this folder (needs your OK)

In `sync_to_pathoreason.py`, so that nothing private can reach the public repository even by mistake:

```python
SKIP_DIRS |= {"local_only", "kaggle_local_only", "_test_tmp", "_local_test_tmp", "jobs"}
SKIP_PAT += ["*.safetensors", "*.bin", "*.gguf", "*.whl", "*.sidecar.jsonl", "*.meta.json"]
```

In `comp_analyze.py`, the format gate could be applied directly for provider `kaggle`, so that
`CONFIG_TABLE.md` cannot show `PASS` for a step-4 row that fails it. Until then, read the
`step4_gate__*.json` file as well (section 10).
