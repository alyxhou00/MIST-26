#!/bin/bash -l
#SBATCH --job-name=mist-qa-teacher122b
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2
#SBATCH --constraint=a100_80
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jenchien.hou@tum.de
#
# Full teacher generation via vLLM (scripts/teacher_generate_vllm.py) for the
# aya_dataset + oeg subset -- see that script's docstring and EXPERIMENTS.md's 3-way
# teacher comparison (job 3859578) for why: the 122B teacher is measurably more
# knowledge-grounded than 35B-A3B/27B, which matters for these two sources specifically
# (open-ended generation with real headroom) and not for belebele-style MC (filter always
# falls back to gold there regardless of teacher).
# Run from the repo root on Alex (NHR@FAU):
#     sbatch slurm/teacher_gen_vllm.sbatch --source aya,oeg --out runs/teacher122b-aya-oeg.jsonl
#
# Time limit 6h: ~4,300 rows (aya_dataset + oeg share of the train split) at the smoke's
# observed batched throughput (190 tok/s output, 15 concurrent requests, not yet
# GPU-saturated) projects well under this; generous margin since throughput at larger
# batch sizes is unmeasured. Resubmit with the same --out to resume (skips qa_idx already
# present) if it runs long.

module load python/3.12-base cuda/12.8.1
source "$HOME/vllm-venv/bin/activate"

# Compute nodes have no internet -> caches populated on the login node.
export HF_HOME="/home/vault/b279bb/b279bb31/hf_cache_teacher"
export HF_DATASETS_CACHE="$HOME/hf_datasets_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p runs
# Default the output path unless the caller chose one (resume runs MUST pass --out
# explicitly, otherwise each resubmission starts a fresh file).
for arg in "$@"; do
  case "$arg" in --out|--out=*) HAS_OUT=1 ;; esac
done
if [ -z "$HAS_OUT" ]; then
  set -- "$@" --out "runs/teacher122b-$SLURM_JOB_ID.jsonl"
fi

python scripts/teacher_generate_vllm.py "$@"
# stdout+stderr are captured by SLURM into logs/%x-%j.out (see --output above).
