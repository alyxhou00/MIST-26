#!/bin/bash -l
#SBATCH --job-name=mist-qa-smoke-9b
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jenchien.hou@tum.de
#
# Fast Qwen3.5-9B smoke test -- sanity-check the bigger model (download cached, memory fits,
# output not garbage) on the same small aya/hin_Deva subset slurm/smoke-fewshot.sbatch used,
# before committing to the ~24-30h full dev runs (0shot-9b.sbatch / fewshot-9b.sbatch). A/Bs
# zero-shot vs 3-shot, same as smoke-fewshot.sbatch, but on the 9B model.
# Run from the repo root ($WORK/MIST-26):  sbatch slurm/smoke-9b.sbatch
# Assumes slurm/setup.sh has been (re-)run since it now also caches Qwen3.5-9B -- see README.
#
# Sampling params (--temperature 0.7 --top-p 0.8) come from Qwen3.5-9B's own card, which
# differs from the 2B-specific defaults benchmark.py otherwise uses -- see benchmark.py's
# --temperature/--top-p help text.

module load python/3.12-base cuda/12.8.1
source "$WORK/mist-venv/bin/activate"

# Compute nodes have no internet -> use the cache populated on the login node.
export HF_HOME="$WORK/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p runs

MODEL="Qwen/Qwen3.5-9B"
SRC=aya
LANG=hin_Deva

ZERO="runs/smoke-9b-0shot-$SLURM_JOB_ID.csv"
FEW="runs/smoke-9b-fewshot3-$SLURM_JOB_ID.csv"

echo "=== 9B zero-shot (source=$SRC lang=$LANG) ==="
python scripts/benchmark.py --model "$MODEL" --temperature 0.7 --top-p 0.8 \
  --source "$SRC" --lang "$LANG" --shots 0 --out "$ZERO"
python scripts/evaluate.py "$ZERO"

echo "=== 9B few-shot k=3 (source=$SRC lang=$LANG) ==="
python scripts/benchmark.py --model "$MODEL" --temperature 0.7 --top-p 0.8 \
  --source "$SRC" --lang "$LANG" --shots 3 --out "$FEW"
python scripts/evaluate.py "$FEW"
