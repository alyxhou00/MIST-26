#!/bin/bash -l
#SBATCH --job-name=mist-qa-smoke-teacher122b
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2
#SBATCH --constraint=a100_80
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jenchien.hou@tum.de
#
# Smoke test for the BIG teacher candidate: Qwen3.5-122B-A10B-GPTQ-Int4 on the same
# deterministic 15 rows as smoke-teacher.sbatch (35B-A3B) and its --model Qwen/Qwen3.5-27B
# variant -- three teachers, row-by-row comparable.
#
# Differences from smoke-teacher.sbatch, all forced by the GPTQ quantization:
# - 2x a100_80: the int4 checkpoint is 79GB of weights -- fits neither one 80GB GPU (with
#   runtime overhead) nor any a40; teacher_generate.py uses device_map=auto, which shards
#   it across both GPUs.
# - $HOME/mist-venv2 instead of $WORK/mist-venv: GPTQ loading needs the `gptqmodel`
#   package, which cannot be installed into the atuin venv while the group quota blocks
#   writes (README "Temporary layout"). venv2 = requirements.txt + gptqmodel.
# Run from the repo root on Alex (NHR@FAU):
#     sbatch slurm/smoke-teacher-122b.sbatch

module load python/3.12-base cuda/12.8.1
source "$HOME/mist-venv2/bin/activate"

# Compute nodes have no internet -> caches populated on the login node.
export HF_HOME="/home/vault/b279bb/b279bb31/hf_cache_teacher"
export HF_DATASETS_CACHE="$HOME/hf_datasets_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p runs
python scripts/teacher_generate.py --limit 15 \
  --model Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
  --out "runs/smoke-teacher122b-$SLURM_JOB_ID.jsonl" "$@"
# stdout+stderr are captured by SLURM into logs/%x-%j.out (see --output above).
