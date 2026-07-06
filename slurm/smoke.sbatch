#!/bin/bash -l
#SBATCH --job-name=mist-qa-smoke
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jenchien.hou@tum.de
#
# Fast smoke test -- sanity-check a change before committing to the full ~10h dev run
# (slurm/job.sbatch). Submit from the repo root ($WORK/MIST-26):  sbatch slurm/smoke.sbatch
# Assumes the one-time login-node setup (venv + cached model/dataset) is done; see README.
#
# What it does: A/B the language hint on the cross-lingual rows it actually affects
# (aya_dataset, English question -> non-English answer). Same log-capture as job.sbatch:
# both scripts' stdout+stderr go to logs/%x-%j.out (see --output above).

module load python/3.12-base cuda/12.8.1
source "$WORK/mist-venv/bin/activate"

# Compute nodes have no internet -> use the cache populated on the login node.
export HF_HOME="$WORK/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p runs

# Subset to run. Edit these to smoke-test something else (e.g. SRC=belebele, LANG=zho_Hans,
# or leave SRC/LANG empty and set LIMIT=50 for a generic quick run).
SRC=aya
LANG=hin_Deva

HINT="runs/smoke-hint-$SLURM_JOB_ID.csv"
NOHINT="runs/smoke-nohint-$SLURM_JOB_ID.csv"

echo "=== lang-hint ON  (source=$SRC lang=$LANG) ==="
python scripts/benchmark.py --source "$SRC" --lang "$LANG" --lang-hint --out "$HINT"
python scripts/evaluate.py "$HINT"

echo "=== lang-hint OFF (source=$SRC lang=$LANG) ==="
python scripts/benchmark.py --source "$SRC" --lang "$LANG" --no-lang-hint --out "$NOHINT"
python scripts/evaluate.py "$NOHINT"
