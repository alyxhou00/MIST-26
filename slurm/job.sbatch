#!/bin/bash -l
#SBATCH --job-name=mist-qa
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jenchien.hou@tum.de
#
# Run from the repo root ($WORK/MIST-26) on Alex (NHR@FAU):  sbatch slurm/job.sbatch
# Assumes the one-time login-node setup (venv + cached model/dataset) is done; see README.
#
# Log goes to logs/ (committed -- $WORK has no backup/retention guarantee, so `git add
# logs/*.out && git commit && git push` from the cluster once the job finishes). Predictions
# CSV goes to runs/ (gitignored scratch -- promote it to predictions/ only if worth keeping).

module load python/3.12-base cuda/12.8.1
source "$WORK/mist-venv/bin/activate"

# Compute nodes have no internet -> use the cache populated on the login node.
export HF_HOME="$WORK/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Write per-example results into the repo's (gitignored) runs/ dir, keyed by job ID so
# runs don't overwrite each other -- same relative path locally and on the cluster.
mkdir -p runs
CSV="runs/predictions-$SLURM_JOB_ID.csv"
python scripts/benchmark.py --out "$CSV"   # writes predictions to $CSV; add --limit 50 for a quick run
python scripts/evaluate.py "$CSV"          # reads $CSV, prints chrF/BERTScore/ROUGE-L (overall + per-source + per-language)
# Both scripts' stdout+stderr are captured by SLURM into logs/%x-%j.out (see --output above).
