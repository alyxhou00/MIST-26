#!/bin/bash -l
#SBATCH --job-name=mist-qa-teacher
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jenchien.hou@tum.de
#
# Teacher generation over the qa train split (scripts/teacher_generate.py) for the
# distillation recipe -- see EXPERIMENTS.md "LoRA SFT on distilled data".
# Run from the repo root on Alex (NHR@FAU); all arguments are passed through:
#     sbatch slurm/teacher_gen.sbatch
#     sbatch slurm/teacher_gen.sbatch --shard 1/2 --out runs/teacher-s1of2.jsonl
# Smoke-test first with slurm/smoke-teacher.sbatch.
#
# Hardware: Qwen3.5-35B-A3B in bf16 is ~70GB of weights -> needs the a100_80 nodes
# (a100_40 would need gres=gpu:a100:2 + the script's device_map=auto). MoE with 3B active
# params, so decoding is fast despite the size; if throughput projects past the 24h cap
# anyway, resubmit with the same --out (rows already generated are skipped) or split with
# --shard i/n across two jobs.
#
# Teacher weights are cached under $HOME/hf_cache_teacher, NOT $WORK/hf_cache: the atuin
# group file quota is exceeded (README "Temporary layout"), so new downloads can only land
# in $HOME. Populate on the login node first (internet):
#     HF_HOME=$HOME/hf_cache_teacher hf download Qwen/Qwen3.5-35B-A3B
# The prepared datasets cache must also be the $HOME copy (offline `datasets` will not
# re-prepare from the hub cache -- job 3857583's failure mode).

module load python/3.12-base cuda/12.8.1
source "$WORK/mist-venv/bin/activate"

# Compute nodes have no internet -> caches populated on the login node (see header).
export HF_HOME="$HOME/hf_cache_teacher"
export HF_DATASETS_CACHE="$HOME/hf_datasets_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p runs
# Default the output path to one keyed by job ID unless the caller chose one (resume runs
# MUST pass --out explicitly, otherwise each resubmission starts a fresh file).
for arg in "$@"; do
  case "$arg" in --out|--out=*) HAS_OUT=1 ;; esac
done
if [ -z "$HAS_OUT" ]; then
  set -- "$@" --out "runs/teacher-$SLURM_JOB_ID.jsonl"
fi

python scripts/teacher_generate.py "$@"
# stdout+stderr are captured by SLURM into logs/%x-%j.out (see --output above).
