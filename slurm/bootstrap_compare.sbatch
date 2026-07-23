#!/bin/bash -l
#SBATCH --job-name=mist-qa-bootstrap
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jenchien.hou@tum.de
#
# Paired bootstrap on one dev source column (scripts/bootstrap_compare.py) -- how much of a
# COMBINED gap between two systems is the column's sample size. Needs a GPU for the same
# reason evaluate.sbatch does: BERTScore is a forward pass per row.
#
#   sbatch slurm/bootstrap_compare.sbatch runs/predictions-lora-REF.csv runs/predictions-lora-B.csv ...
#   sbatch slurm/bootstrap_compare.sbatch --source CohereLabs/aya_dataset REF.csv B.csv
#
# First CSV is the reference; the rest are reported as deltas against it. All arguments are
# passed straight through, so --source / --n-boot / --seed work here too.
module load python/3.12-base cuda/12.8.1
MIST_VENV="$WORK/mist-venv"; [ -f "$MIST_VENV/bin/activate" ] || MIST_VENV="$HOME/mist-venv"
source "$MIST_VENV/bin/activate"
export HF_HOME="$WORK/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python scripts/bootstrap_compare.py "$@"
