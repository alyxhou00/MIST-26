#!/bin/bash -l
#SBATCH --job-name=mist-qa-smoke-teacher
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jenchien.hou@tum.de
#
# Smoke test for scripts/teacher_generate.py before the full train-split run: 15 rows
# (the shuffled split mixes sources, so this samples several datasets/languages), enough
# to verify the 35B-A3B teacher loads on one a100_80, generation works, throughput is
# measurable from the log timestamps, and the answers look teacher-grade next to the
# golds. Run from the repo root on Alex (NHR@FAU) after caching the teacher weights (see
# slurm/teacher_gen.sbatch's header):
#     sbatch slurm/smoke-teacher.sbatch
#     sbatch slurm/smoke-teacher.sbatch --model Qwen/Qwen3.5-27B   # extra args pass through
# The default 15 rows are deterministic (seed-42 split order), so smokes of different
# teacher models are directly comparable row-by-row.

module load python/3.12-base cuda/12.8.1
# venv: prefer $WORK (documented home for regenerable software); fall back
# to $HOME when it has been archived off atuin to relieve the group inode quota.
MIST_VENV="$WORK/mist-venv"; [ -f "$MIST_VENV/bin/activate" ] || MIST_VENV="$HOME/mist-venv"
source "$MIST_VENV/bin/activate"

# Compute nodes have no internet -> caches populated on the login node (see header).
# Teacher weights live on $HPCVAULT (1TB quota) -- $HOME's 100GB soft quota is exceeded by
# double-counted NFS accounting with even one big model cached (README "Temporary layout").
export HF_HOME="/home/vault/b279bb/b279bb31/hf_cache_teacher"
# Prepared-dataset cache: prefer atuin, retreat to $HOME while atuin is unwritable
# (README "Temporary layout"); probed at runtime so this heals when the quota is fixed.
if touch "$WORK/hf_cache/datasets/.write-probe" 2>/dev/null; then
  rm -f "$WORK/hf_cache/datasets/.write-probe"
  export HF_DATASETS_CACHE="$WORK/hf_cache/datasets"
else
  export HF_DATASETS_CACHE="$HOME/hf_datasets_cache"
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p runs
python scripts/teacher_generate.py --limit 15 \
  --out "runs/smoke-teacher-$SLURM_JOB_ID.jsonl" "$@"
# stdout+stderr are captured by SLURM into logs/%x-%j.out (see --output above).
