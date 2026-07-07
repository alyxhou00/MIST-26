#!/bin/bash -l
#SBATCH --job-name=mist-qa-0shot-9b
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
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
# Time limit is 24h (2x the 2B 0shot.sbatch's 12h): 9B is ~4.5x the weights of 2B and
# generation is memory-bandwidth-bound, so per-token cost scales roughly with active weight
# bytes; the 2B run only used 5h36m of its 12h budget, so 24h leaves real margin. 24h also
# happens to be the hard per-job max on every partition here (`sinfo` shows
# TIMELIMIT=1-00:00:00 on a40/a100/rtxpro6k/a100mig alike) -- if this run needs more, ask
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
