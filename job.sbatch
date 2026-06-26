#!/bin/bash -l
#SBATCH --job-name=mist-qa
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#
# Run on Alex (NHR@FAU):  sbatch job.sbatch
# Assumes the one-time login-node setup (venv + cached model/dataset) is done; see README.

module load python/3.12-base cuda/12.8.1
source "$WORK/mist-venv/bin/activate"

# Compute nodes have no internet -> use the cache populated on the login node.
export HF_HOME="$WORK/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Write per-example results to a per-job CSV so runs don't overwrite each other.
mkdir -p "$WORK/mist-out"
CSV="$WORK/mist-out/predictions-$SLURM_JOB_ID.csv"
python benchmark.py --out "$CSV"   # add --limit 50 for a quick run
python evaluate.py "$CSV"          # chrF summary (overall + per-source + per-language) into the .out log
