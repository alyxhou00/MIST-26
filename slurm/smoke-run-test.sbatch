#!/bin/bash -l
#SBATCH --job-name=mist-qa-smoke-test
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jenchien.hou@tum.de
#
# Smoke test for scripts/run_test.py before any full test-set run: a handful of rows from
# each qa task, including the surprise language Bhojpuri (bho -- zero training rows; the
# point is to eyeball whether outputs stay in Bhojpuri or slide into Hindi, and whether
# embedded word-count instructions are respected). Outputs land in runs/ for inspection.
# Run from the repo root on Alex (NHR@FAU) after downloading data/tests.jsonl (see
# slurm/run_test.sbatch's header):
#     sbatch slurm/smoke-run-test.sbatch

module load python/3.12-base cuda/12.8.1
source "$WORK/mist-venv/bin/activate"

# Compute nodes have no internet -> use the cache populated on the login node.
export HF_HOME="$WORK/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p runs
python scripts/run_test.py --task qa-context --limit 10 \
  --out "runs/smoke-test-context-$SLURM_JOB_ID.jsonl"
python scripts/run_test.py --task qa-oeg --lang bho --limit 10 \
  --out "runs/smoke-test-oeg-bho-$SLURM_JOB_ID.jsonl"
# stdout+stderr are captured by SLURM into logs/%x-%j.out (see --output above).
