#!/bin/bash -l
#SBATCH --job-name=mist-qa-smoke-fewshot
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jenchien.hou@tum.de
#
# Fast few-shot smoke test -- A/B `--shots N` vs zero-shot on a small subset before
# committing to the full dev run (slurm/fewshot.sbatch). Submit from the repo root
# ($WORK/MIST-26):  sbatch slurm/smoke-fewshot.sbatch
# Assumes the one-time login-node setup (venv + cached model/dataset) is done; see README.
#
# Time limit is 1h (vs 40min for the lang-hint smoke): few-shot prompts carry N extra
# demonstration passages, so prefill is a few times longer per example. Same log capture
# as the other jobs: both scripts' stdout+stderr go to logs/%x-%j.out (see --output above).

module load python/3.12-base cuda/12.8.1
source "$WORK/mist-venv/bin/activate"

# Compute nodes have no internet -> use the cache populated on the login node.
export HF_HOME="$WORK/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p runs

# Subset + shot count to run. aya/hin_Deva matches the lang-hint smoke so results are
# comparable; edit SRC/LANG to smoke-test elsewhere (e.g. SRC=belebele, where few-shot
# should help most by demonstrating the multiple-choice answer format).
SRC=aya
LANG=hin_Deva
SHOTS=3

FEWSHOT="runs/smoke-fewshot$SHOTS-$SLURM_JOB_ID.csv"
ZEROSHOT="runs/smoke-0shot-$SLURM_JOB_ID.csv"

echo "=== $SHOTS-shot (source=$SRC lang=$LANG) ==="
python scripts/benchmark.py --source "$SRC" --lang "$LANG" --shots "$SHOTS" --out "$FEWSHOT"
python scripts/evaluate.py "$FEWSHOT"

echo "=== zero-shot control (source=$SRC lang=$LANG) ==="
python scripts/benchmark.py --source "$SRC" --lang "$LANG" --out "$ZEROSHOT"
python scripts/evaluate.py "$ZEROSHOT"
