#!/bin/bash -l
#SBATCH --job-name=mist-qa-fewshot-9b
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=30:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jenchien.hou@tum.de
#
# Full few-shot dev run on Qwen3.5-9B -- same as slurm/0shot-9b.sbatch but with --shots
# (default 3), mirroring slurm/fewshot.sbatch's (2B) shape.
# Run from the repo root ($WORK/MIST-26) on Alex (NHR@FAU):
#     sbatch slurm/fewshot-9b.sbatch        # 3 shots
#     sbatch slurm/fewshot-9b.sbatch 5      # any other shot count
# Smoke-test first with slurm/smoke-9b.sbatch. Assumes slurm/setup.sh has been (re-)run since
# it now also caches Qwen3.5-9B -- see README.
#
# Time limit is 30h (1.25x the 24h 0shot-9b.sbatch budget -- the same ratio the 2B
# 0shot->fewshot step used, 12h->18h -- since every prompt carries N extra demonstration
# passages on top of the already-larger 9B model). If SLURM rejects this as exceeding the
# partition's max walltime, check `sinfo -p a40` / NHR@FAU docs and either request a longer
# QOS or split the run with --limit.
#
# Sampling params (--temperature 0.7 --top-p 0.8) come from Qwen3.5-9B's own card.

module load python/3.12-base cuda/12.8.1
source "$WORK/mist-venv/bin/activate"

# Compute nodes have no internet -> use the cache populated on the login node.
export HF_HOME="$WORK/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SHOTS="${1:-3}"

# Write per-example results into the repo's (gitignored) runs/ dir, keyed by shot count and
# job ID so runs don't overwrite each other -- same relative path locally and on the cluster.
mkdir -p runs
CSV="runs/predictions-fewshot$SHOTS-9b-$SLURM_JOB_ID.csv"
python scripts/benchmark.py --model Qwen/Qwen3.5-9B --temperature 0.7 --top-p 0.8 \
  --shots "$SHOTS" --out "$CSV"
python scripts/evaluate.py "$CSV"          # prints chrF/BERTScore/ROUGE-L (overall + per-source + per-language)
# Both scripts' stdout+stderr are captured by SLURM into logs/%x-%j.out (see --output above).
