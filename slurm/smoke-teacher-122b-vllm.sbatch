#!/bin/bash -l
#SBATCH --job-name=mist-qa-smoke-teacher122b-vllm
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2
#SBATCH --constraint=a100_80
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jenchien.hou@tum.de
#
# Smoke test for the 122B teacher via vLLM (scripts/teacher_generate_vllm.py) -- same 15
# deterministic rows as the other teacher smokes. transformers+gptqmodel could not load
# this checkpoint (jobs 3859341/3859345/3859381/3859398: missing deps, then Marlin shape
# rejection, then CUDA illegal memory access); vLLM is the standard serving path for
# Qwen MoE GPTQ. Uses $HOME/vllm-venv (vllm pins its own torch, kept isolated).
# Run from the repo root on Alex (NHR@FAU):
#     sbatch slurm/smoke-teacher-122b-vllm.sbatch

module load python/3.12-base cuda/12.8.1
source "$HOME/vllm-venv/bin/activate"

# Compute nodes have no internet -> caches populated on the login node.
export HF_HOME="/home/vault/b279bb/b279bb31/hf_cache_teacher"
export HF_DATASETS_CACHE="$HOME/hf_datasets_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p runs
python scripts/teacher_generate_vllm.py --limit 15 \
  --out "runs/smoke-teacher122b-vllm-$SLURM_JOB_ID.jsonl" "$@"
# stdout+stderr are captured by SLURM into logs/%x-%j.out (see --output above).
