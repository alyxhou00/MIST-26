#!/bin/bash -l
#SBATCH --job-name=mist-qa
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#
# Run on Alex (NHR@FAU):  sbatch job.sbatch
# One-time setup on the login node first:
#   module load python/3.12-base cuda/12.8.1
#   python -m venv $WORK/mist-venv && source $WORK/mist-venv/bin/activate
#   pip install -r requirements.txt

module load python/3.12-base cuda/12.8.1
source "$WORK/mist-venv/bin/activate"

python benchmark.py --limit 50
