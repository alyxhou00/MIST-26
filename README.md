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
| [`scripts/`](scripts) | `benchmark.py` (generation, zero/few-shot, base or LoRA), `train_lora.py` (LoRA SFT), `evaluate.py` (scoring), `error_analysis.py` (failure-mode breakdown) |
| [`slurm/`](slurm) | one sbatch file per experiment, named after it: `0shot.sbatch` / `fewshot.sbatch` (Qwen3.5-2B full dev runs), `0shot-9b.sbatch` / `fewshot-9b.sbatch` (Qwen3.5-9B), `lora_sft.sbatch` / `lora_eval.sbatch` (LoRA SFT), `smoke-langhint.sbatch` / `smoke-fewshot.sbatch` / `smoke-9b.sbatch` / `smoke-lora.sbatch` (cheap A/Bs and pipeline checks); plus `setup.sh` (one-time login-node setup) and `evaluate.sbatch` (re-scoring) |
| [`predictions/`](predictions) | predictions CSVs worth keeping long-term, committed deliberately |
| [`logs/`](logs) | every slurm `.out` log, always committed -- `$WORK` has no backup/retention guarantee, so logs are small and cheap enough to keep all of them |
| `runs/` | gitignored scratch dir for ad-hoc predictions CSVs (large, so only the ones worth keeping get promoted into `predictions/`) |
| `adapters/` | gitignored scratch dir for trained LoRA adapters (`train_lora.py --out`), same reasoning as `runs/` |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | the experiment log -- one row per SLURM job ID, with its config and chrF/BERTScore/ROUGE-L |

`$WORK/mist-out` (outside the repo) was the old location for predictions CSVs before this
layout existed. It's no longer used by any sbatch job -- everything now lives
under the repo (`logs/`, `runs/`, `predictions/`) so the same relative paths work locally and on
the cluster. Any old files left in `$WORK/mist-out` are safe to ignore or delete; nothing reads
from there anymore.

## 0. Zero-shot QA benchmark (`Qwen/Qwen3.5-2B`)

[`scripts/benchmark.py`](scripts/benchmark.py) is a minimal zero-shot benchmark of the `qa`
sub-task: it splits the examples **80/20 train/dev** (seed 42), runs the model on the dev half
via its chat template, and writes a **predictions CSV**. Scoring is a separate step --
[`scripts/evaluate.py`](scripts/evaluate.py) reads the CSV and reports **chrF / BERTScore /
ROUGE-L** (single source of truth). `Qwen/Qwen3.5-2B` is multimodal but used here text-only
(each `input` is one user turn); it needs a recent `transformers` (see `requirements.txt`).

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

## 1. Few-shot prompting (`--shots N`)

`benchmark.py --shots 3` prepends N demonstration examples to each prompt as **completed
user/assistant chat turns** (real prior exchanges the model can imitate), before the actual
question. `--shots 0` (the default) is the zero-shot behavior above. How shots are chosen:

- **Pool:** only the **train 80%** of the split — a dev example can never appear among its own
  demonstrations, so the dev metric stays honest. Train rows whose input is byte-identical to
  the dev example's are also excluded: a few aya prompts repeat verbatim across the split
  (53 of 2978 dev rows, 4 with the same gold), and showing such a copy as a demonstration
  would hand the model its own answer.
- **Matching:** shots share the dev example's `(source, lang_code)` where possible, so they
  demonstrate both the task format (e.g. `belebele`'s multiple-choice answers) and the target
  language. If that stratum has fewer than N train rows, selection falls back to same `source`
  only, then to the whole train pool.
- **Determinism:** each dev example's shots are seeded from a hash of its input text, so they
  don't change with `--limit`/`--source`/`--lang` filters or row order — A/B runs stay
  comparable and any single prediction is reproducible in isolation.

The turn insertion lives in `build_messages()` in
[`scripts/prompt_template.py`](scripts/prompt_template.py) (shared with SFT); the selection
logic is `make_shot_picker()` in [`scripts/benchmark.py`](scripts/benchmark.py).

On the cluster: [`slurm/smoke-fewshot.sbatch`](slurm/smoke-fewshot.sbatch) is a ~1h A/B of
N-shot vs zero-shot on the cross-lingual `aya` subset (edit `SRC`/`LANG`/`SHOTS` at the top);
[`slurm/fewshot.sbatch`](slurm/fewshot.sbatch) is the full dev run
(`sbatch slurm/fewshot.sbatch` for 3 shots, or pass a count: `sbatch slurm/fewshot.sbatch 5`).
Its time limit is 18h rather than 12h because every prompt carries N extra demonstration
passages.

## 2. Qwen3.5-9B benchmark

The organizers allow any model/approach (including fine-tuning or distillation) as long as the
final model is under 10B parameters, so `Qwen/Qwen3.5-9B` -- the same model family as the 2B
above, just a bigger size in the same collection -- is benchmarked the same two ways (0-shot,
then few-shot) before building on it with LoRA SFT. `benchmark.py --model`/`--shots` already
generalize to it; the only real difference is sampling parameters, since Qwen's model cards give
per-size recommendations (9B's non-thinking-mode card: `temperature=0.7, top_p=0.8, top_k=20`,
vs. the 2B-specific `1.0/1.0/20` `benchmark.py` otherwise defaults to) -- hence the new
`--temperature`/`--top-p` flags.

On the cluster: `bash slurm/setup.sh` now also caches Qwen3.5-9B (re-run it if you set up before
this was added). [`slurm/smoke-9b.sbatch`](slurm/smoke-9b.sbatch) is a ~1h 0-shot/3-shot A/B on
the same `aya`/`hin_Deva` subset `smoke-fewshot.sbatch` uses, to sanity-check the bigger model
before committing to a full run; [`slurm/0shot-9b.sbatch`](slurm/0shot-9b.sbatch) (24h) and
[`slurm/fewshot-9b.sbatch`](slurm/fewshot-9b.sbatch) (30h, shot count as `$1`) are the full
dev-set runs -- longer time limits than their 2B counterparts since 9B is ~4.5x the weights.

## 3. LoRA SFT

[`scripts/train_lora.py`](scripts/train_lora.py) fine-tunes a LoRA adapter on the train 80% of
the `qa` split (the same seed-42 split `benchmark.py` evaluates against), using the identical
zero-shot prompt (`prompt_template.build_messages`, no demonstrations) so the adapter is trained
to be good at the same thing `benchmark.py --shots 0` measures -- keeping "did SFT help"
independent of "did few-shot help". LoRA attaches to the text decoder's attention (`q/k/v/o_proj`)
and every layer's MLP (`gate/up/down_proj`); Qwen3.5 is a hybrid architecture (some decoder
layers are full self-attention, others gated linear attention) but every layer has an MLP, so
that half of the adaptation still reaches the whole network. Defaults to `Qwen/Qwen3.5-9B`
(`--model` to use a different size, e.g. the already-benchmarked 2B).

`benchmark.py --lora <adapter dir>` loads the trained adapter on top of `--model` (which must
match whatever `train_lora.py` used) before generation -- everything downstream (predictions CSV
-> `evaluate.py`) is identical in shape to the prompting-only baselines.

On the cluster: `bash slurm/setup.sh` now also installs `peft` (added to `requirements.txt`).
[`slurm/smoke-lora.sbatch`](slurm/smoke-lora.sbatch) (~30 min) trains on 200 rows for 1 epoch
and evaluates on 50 dev rows, to check the pipeline end-to-end before the full run;
[`slurm/lora_sft.sbatch`](slurm/lora_sft.sbatch) (18h, checkpoints every 200 steps and
auto-resumes -- see the file for how to resubmit) trains the full adapter; then
[`slurm/lora_eval.sbatch <adapter dir>`](slurm/lora_eval.sbatch) (24h) scores it on the full
dev set.

## Running on the Alex cluster (NHR@FAU)

The Alex login node has internet but no GPU; the compute nodes have GPUs but no internet. So we
download everything once on the login node ([`slurm/setup.sh`](slurm/setup.sh)) and run the GPU
job offline. Each experiment has its own sbatch file named after it (`0shot.sbatch`,
`fewshot.sbatch`, ...), so the exact command that produced a run is recorded and its log is
identifiable by job name.

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

**3. (Recommended) Smoke test on a GPU.** Either grab an interactive node and run 20 examples:

```bash
srun --partition=a40 --gres=gpu:a40:1 --time=00:15:00 --pty bash -l
module load python/3.12-base cuda/12.8.1
source $WORK/mist-venv/bin/activate
export HF_HOME=$WORK/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/benchmark.py --limit 20
exit          # release the interactive allocation
```

...or submit [`slurm/smoke-langhint.sbatch`](slurm/smoke-langhint.sbatch)
(`sbatch slurm/smoke-langhint.sbatch`), a short (~40 min) job that A/Bs `--lang-hint` on/off
over just the cross-lingual `aya` rows — a cheap way to check a prompt change before spending
~10h on the full `0shot.sbatch` run. Edit the `SRC`/`LANG` vars at the top to target a
different subset. The `--source`/`--lang` filters it uses are on `benchmark.py` directly
(e.g. `--source aya --lang hin_Deva`).

**4. Submit the full dev-set run** and check the result:

```bash
sbatch slurm/0shot.sbatch
squeue --me                     # PD = pending, R = running
cat logs/mist-qa-0shot-*.out    # output, ending in the evaluate.py metric summary
```

Each run produces two artifacts:

| File | Contents |
|------|----------|
| `logs/mist-qa-0shot-<jobid>.out` | log: progress, warnings, final metric summary from `evaluate.py` (chrF / BERTScore / ROUGE-L, stdout + stderr) -- **committed**, so `git add logs/*.out && git commit && git push` from the cluster once the job finishes |
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

See [`EXPERIMENTS.md`](EXPERIMENTS.md) for the full experiment log (one row per SLURM job ID,
with its config and chrF/BERTScore/ROUGE-L). Headline so far: Qwen3.5-2B 0-shot chrF=18.01 ->
3-shot chrF=21.84 (BERTScore/ROUGE-L improve alongside it). See
[`scripts/error_analysis.py`](scripts/error_analysis.py) for a script-mismatch/length-mismatch
breakdown of the low-scoring languages.
