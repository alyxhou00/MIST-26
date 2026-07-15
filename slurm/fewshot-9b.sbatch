#!/bin/bash -l
#SBATCH --job-name=mist-qa-fewshot-9b
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jenchien.hou@tum.de
#
# Full few-shot dev run on Qwen3.5-9B -- same as slurm/0shot-9b.sbatch but with --shots
# (default 3), mirroring slurm/fewshot.sbatch's (2B) shape.
# Run from the repo root ($WORK/MIST-26) on Alex (NHR@FAU):
#     sbatch slurm/fewshot-9b.sbatch        # 3 shots
#     sbatch slurm/fewshot-9b.sbatch 5      # any other shot count
#     sbatch slurm/fewshot-9b.sbatch 3 --no-lang-hint   # extra args go to benchmark.py
# Smoke-test first with slurm/smoke-9b.sbatch. Assumes slurm/setup.sh has been (re-)run since
# it now also caches Qwen3.5-9B -- see README.
#
# Time limit is 8h: the first real run of this file (job 3822329, submitted with a 30h->24h
# budget before any 9B throughput was known) was measured mid-run at ~389/2978 rows in 33
# minutes -- projects to ~4.3h total (faster than 0shot-9b.sbatch's ~6.7h estimate, consistently
# with the 2B baselines too: few-shot answers imitate the shown demonstrations' brevity and hit
# EOS sooner, where zero-shot tends to ramble -- see IMPLEMENTATION_NOTES.md's length-mismatch
# finding). 8h leaves solid margin. The hard per-job max on every partition here is 24h
# regardless (`sinfo` shows TIMELIMIT=1-00:00:00 on a40/a100/rtxpro6k/a100mig) -- if a future
# run ever needs more, ask NHR@FAU support about a longer QOS, or resume the remainder with
# --limit/--source/--lang and concatenate CSVs (benchmark.py flushes each row as it's written,
# so a run that hits the wall clock still leaves a valid, if incomplete, predictions CSV).
#
# Sampling params (--temperature 0.7 --top-p 0.8) come from Qwen3.5-9B's own card.

module load python/3.12-base cuda/12.8.1
source "$WORK/mist-venv/bin/activate"

# Compute nodes have no internet -> use the cache populated on the login node.
export HF_HOME="$WORK/hf_cache"
# Prepared-dataset cache: prefer the canonical atuin copy, and only retreat to the $HOME
# copy while atuin writes fail (group file quota exceeded -- README "Temporary layout").
# Offline `datasets` writes a lock file on every load, so an unwritable cache dir kills the
# job in seconds (jobs 3857583, 3859591). Probed at runtime so the scripts heal themselves
# the moment the quota is fixed; baked in here (not left to the submitter to set).
if touch "$WORK/hf_cache/datasets/.write-probe" 2>/dev/null; then
  rm -f "$WORK/hf_cache/datasets/.write-probe"
  export HF_DATASETS_CACHE="$WORK/hf_cache/datasets"
else
  export HF_DATASETS_CACHE="$HOME/hf_datasets_cache"
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SHOTS="${1:-3}"
[ $# -ge 1 ] && shift  # anything after the shot count is passed through to benchmark.py

# Write per-example results into the repo's (gitignored) runs/ dir, keyed by shot count and
# job ID so runs don't overwrite each other -- same relative path locally and on the cluster.
mkdir -p runs
CSV="runs/predictions-fewshot$SHOTS-9b-$SLURM_JOB_ID.csv"
python scripts/benchmark.py --model Qwen/Qwen3.5-9B --temperature 0.7 --top-p 0.8 \
  --shots "$SHOTS" --out "$CSV" "$@"
python scripts/evaluate.py "$CSV"          # prints chrF/BERTScore/ROUGE-L (overall + per-source + per-language)
# Both scripts' stdout+stderr are captured by SLURM into logs/%x-%j.out (see --output above).
