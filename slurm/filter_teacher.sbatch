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
#
# FILE ORDER MATTERS when combining teachers. The 35B whole-corpus shards and the 122B
# aya+oeg run cover the same qa_idx for every aya/OEG row, and load_shards() drops
# duplicate qa_idx keep-first -- so the file listed FIRST wins those rows. The 122B is the
# better teacher on exactly these two sources (see teacher_gen_vllm.sbatch), so it goes
# first:
#   sbatch slurm/filter_teacher.sbatch runs/teacher122b-aya-oeg.jsonl runs/teacher-s1of3.jsonl runs/teacher-s2of3.jsonl runs/teacher-s3of3.jsonl --report
# Reversing that silently trains on the 35B's aya/OEG answers instead. Sanity-check the
# "WARNING: dropping N duplicate qa_idx rows" line against the aya+OEG row count (4,126).
#
# Caveat of keep-first: the losing file's answer is discarded BEFORE scoring, so a row
# where the 122B hallucinated falls back to gold even when the 35B answered it well. Fixing
# that means scoring both and preferring the passing one -- a filter_teacher.py change, not
# an argument-order one. Note that ranking two *passing* answers by gold similarity is not
# the same fix: it selects the more gold-like (terser) answer, which is what distillation is
# meant to improve on.

module load python/3.12-base cuda/12.8.1
source "/home/atuin/b279bb/b279bb31/mist-venv/bin/activate"

export HF_HOME="/home/atuin/b279bb/b279bb31/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python scripts/filter_teacher.py "$@"
