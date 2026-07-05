#!/bin/bash -l
#SBATCH --job-name=mist-qa-eval
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jenchien.hou@tum.de
#
# Re-score an *existing* predictions CSV (no re-generation) -- chrF + BERTScore + ROUGE-L.
# Needs a GPU because BERTScore does a transformer forward pass over every row.
#
# Run from the repo root ($WORK/MIST-26) on Alex (NHR@FAU):
#   sbatch evaluate.sbatch                                       # scores the newest predictions/*.csv
#   sbatch evaluate.sbatch predictions/predictions-3786727.csv   # or a specific one
#
# Assumes setup.sh has already been (re-)run so the venv has bert-score/rouge-score installed
# and bert-base-multilingual-cased is cached -- see README.

module load python/3.12-base cuda/12.8.1
source "$WORK/mist-venv/bin/activate"

export HF_HOME="$WORK/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CSV="${1:-$(ls -t predictions/*.csv | head -n 1)}"
echo "scoring: $CSV"
python evaluate.py "$CSV"
