#!/bin/bash -l
#SBATCH --job-name=mist-qa-0shot-9b
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jenchien.hou@tum.de
#
# Full zero-shot dev run on Qwen3.5-9B -- same shape as slurm/0shot.sbatch (2B) but a bigger
# base model, under the organizers' 10B-parameter cap. Lang-hint stays ON (default), unlike the
# original 3786727 2B run which predates the lang-hint feature -- see README/EXPERIMENTS.md.
# Run from the repo root ($WORK/MIST-26) on Alex (NHR@FAU):  sbatch slurm/0shot-9b.sbatch
# Smoke-test first with slurm/smoke-9b.sbatch. Assumes slurm/setup.sh has been (re-)run since
# it now also caches Qwen3.5-9B -- see README.
#
# Time limit is 10h: the first real run of this file (job 3822324, submitted with a 24h budget
# before any 9B throughput was known) was measured mid-run at ~292/2978 rows in 39 minutes --
# projects to ~6.7h total, close to the 2B baseline's 5h36m despite 9B being ~4.5x the weights
# (Qwen3.5's hybrid attention/linear-attention layers apparently keep generation cost from
# scaling with raw parameter count). 10h leaves real margin without reserving most of a 24h
# slot idle. The hard per-job max on every partition here is 24h regardless (`sinfo` shows
# TIMELIMIT=1-00:00:00 on a40/a100/rtxpro6k/a100mig) -- if a future run ever needs more, ask
# NHR@FAU support about a longer QOS, or resume the remainder with --limit/--source/--lang and
# concatenate CSVs (benchmark.py flushes each row as it's written, so a run that hits the wall
# clock still leaves a valid, if incomplete, predictions CSV).
#
# Sampling params (--temperature 0.7 --top-p 0.8) come from Qwen3.5-9B's own card -- different
# from the 2B-specific defaults benchmark.py otherwise uses.

module load python/3.12-base cuda/12.8.1
source "$WORK/mist-venv/bin/activate"

# Compute nodes have no internet -> use the cache populated on the login node.
export HF_HOME="$WORK/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Write per-example results into the repo's (gitignored) runs/ dir, keyed by job ID so
# runs don't overwrite each other -- same relative path locally and on the cluster.
mkdir -p runs
CSV="runs/predictions-0shot-9b-$SLURM_JOB_ID.csv"
python scripts/benchmark.py --model Qwen/Qwen3.5-9B --temperature 0.7 --top-p 0.8 --out "$CSV"
python scripts/evaluate.py "$CSV"          # prints chrF/BERTScore/ROUGE-L (overall + per-source + per-language)
# Both scripts' stdout+stderr are captured by SLURM into logs/%x-%j.out (see --output above).
