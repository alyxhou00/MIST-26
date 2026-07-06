# WMT26 MIST

Working notes for the WMT26 MIST shared task, using the sample dataset
[`pinzhenchen/wmt26-mist-sample`](https://huggingface.co/datasets/pinzhenchen/wmt26-mist-sample).

## Tasks

The shared task has three tasks:

**Sub-task 1.** **Context-based question answering** — given a document in language X, answer questions about it, also in language X. <br>
**Sub-task 2.** **Summarization** — given a document in language X, produce a summary in language Y.<br>
**Task 3.** **Open-ended generation** — given an open-ended question, produce a helpful, natural response.

## Columns: `task` and `source`

Two columns describe each example:

- **`task`** — has only **two** values in the data: `qa` and `sum`. It does *not* have a
  separate label for open-ended generation; those examples are tagged `qa`.
- **`source`** — the upstream dataset the example comes from.

## Source dataset → task mapping

Counts are from the `train` split (`df.groupby(["task", "source"]).size()`):

| Conceptual sub-task                | `task` label | `source`                              | rows |
|------------------------------------|--------------|---------------------------------------|------|
| **Sub-task 1 — Context-based QA**  | `qa`         | `facebook/belebele`                   | 5700 |
|                                    | `qa`         | `copenlu/answerable_tydiqa`           | 3112 |
|                                    | `qa`         | `FBK-MT/MCIF` (QA portion)            | 880  |
| **Sub-task 2 — Summarization**     | `sum`        | `csebuetnlp/CrossSum`                 | 7026 |
|                                    | `sum`        | `esdurmus/wiki_lingua`                | 1600 |
|                                    | `sum`        | `FBK-MT/MCIF` (summarization portion) | 400  |
| **Task 3 — Open-ended generation** | `qa`         | `CohereLabs/aya_dataset`              | 4741 |
|                                    | `qa`         | `wmt25-mist-oeg-gpt-4.1`              | 460  |

Totals: `qa` = 14,893 rows, `sum` = 9,026 rows.

## Load Dataset

```bash
pip install datasets
```

```python
from datasets import load_dataset

ds = load_dataset("pinzhenchen/wmt26-mist-sample")
print(ds["train"])   # the only split -- train
```

## Layout

| Path | Contents |
|------|----------|
| [`scripts/`](scripts) | `benchmark.py` (generation), `evaluate.py` (scoring), `error_analysis.py` (failure-mode breakdown) |
| [`slurm/`](slurm) | `setup.sh` (one-time login-node setup), `job.sbatch`, `evaluate.sbatch` |
| [`predictions/`](predictions) | predictions CSVs worth keeping long-term, committed deliberately |
| [`logs/`](logs) | every slurm `.out` log, always committed -- `$WORK` has no backup/retention guarantee, so logs are small and cheap enough to keep all of them |
| `runs/` | gitignored scratch dir for ad-hoc predictions CSVs (large, so only the ones worth keeping get promoted into `predictions/`) |

`$WORK/mist-out` (outside the repo) was the old location for predictions CSVs before this
layout existed. It's no longer used by `job.sbatch`/`evaluate.sbatch` -- everything now lives
under the repo (`logs/`, `runs/`, `predictions/`) so the same relative paths work locally and on
the cluster. Any old files left in `$WORK/mist-out` are safe to ignore or delete; nothing reads
from there anymore.

## 0. Zero-shot QA benchmark (`Qwen/Qwen3.5-2B`)

[`scripts/benchmark.py`](scripts/benchmark.py) is a minimal zero-shot benchmark of the `qa`
sub-task: it splits the examples **80/20 train/dev** (seed 42), runs the model on the dev half
via its chat template, and reports **chrF**. `Qwen/Qwen3.5-2B` is multimodal but used here
text-only (each `input` is one user turn); it needs a recent `transformers` (see
`requirements.txt`).

```bash
python scripts/benchmark.py --limit 50     # quick check
python scripts/benchmark.py                # full dev split (target-language hint ON by default)
python scripts/benchmark.py --no-lang-hint # raw zero-shot: no target-language instruction
```

**Language hint (on by default).** Each prompt gets a system turn `Respond in <language>.`,
where `<language>` is derived from `lang_code`. `lang_code` is the **output** language (verified:
e.g. `FBK-MT/MCIF` `zho_Hans` rows have an English input passage but Chinese golds). Most examples
don't strictly need the hint — the input's language already matches the expected output — but a
few (e.g. `aya_dataset` rows where the question is English but the answer is expected in another
language) are otherwise ambiguous: nothing in the input signals that the output should switch
languages. The same template lives in [`scripts/prompt_template.py`](scripts/prompt_template.py)
and is reused for SFT so training and inference match.

Pass **`--no-lang-hint`** to drop the system turn and reproduce the raw zero-shot baseline
(user turn only) — useful as an A/B control to measure how much the hint helps.

## Running on the Alex cluster (NHR@FAU)

The Alex login node has internet but no GPU; the compute nodes have GPUs but no internet. So we
download everything once on the login node ([`slurm/setup.sh`](slurm/setup.sh)) and run the GPU
job offline ([`slurm/job.sbatch`](slurm/job.sbatch)).

**1. Get the code onto the cluster.** Push from your laptop, then clone into `$WORK`:

```bash
# on your laptop
git push origin main

# on the cluster
ssh alex
cd $WORK
git clone https://github.com/alyxhou00/MIST-26.git
cd MIST-26
```

**2. One-time setup on the login node** — builds the venv and caches the model + dataset:

```bash
bash slurm/setup.sh
```

**3. (Recommended) Smoke test on a GPU** — grab an interactive node and run 20 examples:

```bash
srun --partition=a40 --gres=gpu:a40:1 --time=00:15:00 --pty bash -l
module load python/3.12-base cuda/12.8.1
source $WORK/mist-venv/bin/activate
export HF_HOME=$WORK/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/benchmark.py --limit 20
exit          # release the interactive allocation
```

**4. Submit the full dev-set run** and check the result:

```bash
sbatch slurm/job.sbatch
squeue --me                # PD = pending, R = running
cat logs/mist-qa-*.out     # output, ending in the chrF score
```

Each run produces two artifacts:

| File | Contents |
|------|----------|
| `logs/mist-qa-<jobid>.out` | log: progress, warnings, final chrF score (stdout + stderr) -- **committed**, so `git add logs/mist-qa-<jobid>.out && git commit && git push` from the cluster once the job finishes |
| `runs/predictions-<jobid>.csv` | per-example `source, lang_code, input, gold, prediction` -- gitignored scratch; promote it into `predictions/` (see below) only if it's worth keeping |

**5. Promote the predictions CSV into `predictions/`, if it's worth keeping.** `runs/` is
gitignored scratch, so this is what actually saves a run long-term. Two equivalent ways to do
it, replacing `<jobid>` with the actual job number:

- **Directly on the cluster** (no download needed):

  ```bash
  cp runs/predictions-<jobid>.csv predictions/
  git add predictions/predictions-<jobid>.csv
  git commit -m "add predictions-<jobid>"
  git push
  ```

- **Via your laptop** (e.g. to eyeball the CSV locally first): download it, then commit from
  there instead:

  ```bash
  # on your laptop
  scp alex:/home/atuin/b279bb/b279bb31/MIST-26/runs/predictions-<jobid>.csv predictions/
  git add predictions/predictions-<jobid>.csv
  git commit -m "add predictions-<jobid>"
  git push
  ```

**Re-scoring only (no re-generation).** `evaluate.py` now also computes BERTScore, which needs
a GPU forward pass over every row -- slow on a laptop CPU. Predictions worth keeping get
committed under `predictions/` (e.g. `predictions/predictions-3786727.csv`) so they're already
in the repo; if that same file is on the cluster too (after a `git pull`), re-score it there
instead of regenerating:

```bash
ssh alex
cd $WORK/MIST-26
git pull                       # get scripts/evaluate.py updates + any new predictions/*.csv
bash slurm/setup.sh            # re-run once: installs bert-score/rouge-score, caches mBERT
sbatch slurm/evaluate.sbatch   # scores the newest predictions/*.csv
squeue --me
cat logs/mist-qa-eval-*.out    # chrF + BERTScore + ROUGE-L summary
git add logs/mist-qa-eval-*.out && git commit -m "add eval log" && git push
```

Pass a specific file (`sbatch slurm/evaluate.sbatch predictions/predictions-<jobid>.csv`) if you
don't want the newest one picked automatically.

## Results so far

`predictions/predictions-3786727.csv` (2978 dev examples, zero-shot, no `--lang-hint`), scored
with `scripts/evaluate.py`:

overall chrF = 18.01, BERTScore = 62.21, ROUGE-L = 12.51

Best-scoring languages (chrF): Indonesian (25.7), English (25.1), French (25.0). Worst: Telugu
(6.2), Swahili (7.2), Thai (7.2). See
[`scripts/error_analysis.py`](scripts/error_analysis.py) for a script-mismatch/length-mismatch
breakdown of the low scorers.
