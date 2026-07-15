#!/bin/bash -l
#SBATCH --job-name=mist-filter-teacher
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jenchien.hou@tum.de
#
# scripts/filter_teacher.py on a compute node -- BERTScorer() defaults to CPU with no thread
# cap, which is exactly what got the login-node (`alex1`) run killed by the HPC admin (it
# pegged ~67 cores on a frontend shared by hundreds of users). A GPU node gives BERTScore a
# device to run on instead of fanning out over every core it can see, and keeps the frontend
# free for compiling/submitting per NHR@FAU's actual-use policy.
#
# Run from the repo root ($HOME/MIST-26, per README's "Temporary layout" atuin-quota
# workaround) on Alex (NHR@FAU):
#   sbatch slurm/filter_teacher.sbatch runs/teacher122b-aya-oeg.jsonl --report
#   sbatch slurm/filter_teacher.sbatch runs/teacher122b-aya-oeg.jsonl --chrf-min 30 --bertscore-min 70 --out data/sft-distilled.jsonl
# All args are passed straight through to filter_teacher.py.

module load python/3.12-base cuda/12.8.1
source "/home/atuin/b279bb/b279bb31/mist-venv/bin/activate"

export HF_HOME="/home/atuin/b279bb/b279bb31/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python scripts/filter_teacher.py "$@"
