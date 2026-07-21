#!/bin/bash -l
#SBATCH --job-name=mist-qa-test
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jenchien.hou@tum.de
#
# Official test-set inference (scripts/run_test.py) -> submission-format JSONL.
# Run from the repo root on Alex (NHR@FAU); all arguments are passed through:
#     sbatch slurm/run_test.sbatch                              # all 10,999 qa rows
#     sbatch slurm/run_test.sbatch --shard 1/3 --out runs/test-qa-s1of3.jsonl
#     sbatch slurm/run_test.sbatch --lora <adapter dir> --lang-hint
# Needs data/tests.jsonl (47MB, gitignored) -- download it on the login node first:
#     curl -sL "https://huggingface.co/datasets/pinzhenchen/wmt26-mist-test/resolve/main/tests.jsonl" \
#       -o data/tests.jsonl
#
# Time limit is the 24h partition max. MEASURED rates on this workload (9B + LoRA, one a40,
# max-new-tokens 512) -- NOT benchmark.py's dev rates, which this header used to quote and
# which understate qa-oeg by ~3x:
#     qa-context  ~5.9 s/row   (jobs 3876525/3876526 -- the prompt asks for ONE SENTENCE,
#                               so generation stops early and 512 is never approached)
#     qa-oeg     ~20.5 s/row   (jobs 3875151/3875152 -- answers run to the full budget)
# The full qa set is 8,640 qa-context + 2,359 qa-oeg = 8,640*5.9 + 2,359*20.5 ~= 27.5h, i.e.
# over the wall but only just: --shard i/2 is ~13.8h per shard and is enough. Use a separate
# --out per shard and concatenate, or just resubmit with the SAME --out -- run_test.py skips
# ids already present in it and resumes.
# (Caveat: the qa-context rate was measured on bho only; the one-sentence instruction is
# attested in every language, so it should carry, but re-check before the final schedule.)
#
# Smoke-test first with slurm/smoke-run-test.sbatch.

module load python/3.12-base cuda/12.8.1
# venv: prefer $WORK (documented home for regenerable software); fall back
# to $HOME when it has been archived off atuin to relieve the group inode quota.
MIST_VENV="$WORK/mist-venv"; [ -f "$MIST_VENV/bin/activate" ] || MIST_VENV="$HOME/mist-venv"
source "$MIST_VENV/bin/activate"

# Compute nodes have no internet -> use the cache populated on the login node.
export HF_HOME="$WORK/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p runs
# Default the output path to one keyed by job ID unless the caller chose one (resume runs
# MUST pass --out explicitly, otherwise each resubmission starts a fresh file).
for arg in "$@"; do
  case "$arg" in --out|--out=*) HAS_OUT=1 ;; esac
done
if [ -z "$HAS_OUT" ]; then
  set -- "$@" --out "runs/test-qa-$SLURM_JOB_ID.jsonl"
fi

python scripts/run_test.py "$@"
# stdout+stderr are captured by SLURM into logs/%x-%j.out (see --output above).
