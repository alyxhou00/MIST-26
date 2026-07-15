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
# (a100_40 would need gres=gpu:a100:2 + the script's device_map=auto).
#
# Time: MEASURED throughput is ~14 s/row (smoke 3859176 and the real shards agree) -- the
# transformers loop is serial and overhead-bound, so the MoE's 3B active params do NOT make
# it fast. The full 11,915-row train split projects to ~46h >> the 24h partition cap:
# ALWAYS split (`--shard i/3` => ~16h/shard, what jobs 3859277-79 ran) or resubmit with the
# same --out to resume (rows already generated are skipped). For large batches prefer
# scripts/teacher_generate_vllm.py -- its batched decode measured ~250x faster per row
# (4,126 rows in 17min, job 3859682).
#
# Teacher weights are cached on $HPCVAULT, NOT $WORK/hf_cache (atuin group file quota
# exceeded) and NOT $HOME (its 100GB soft quota is blown by double-counted NFS accounting
# with even one big model cached) -- README "Temporary layout". Populate on the login node
# first (internet):
#     HF_HOME=/home/vault/b279bb/b279bb31/hf_cache_teacher hf download Qwen/Qwen3.5-35B-A3B
# The prepared datasets cache is probed below (offline `datasets` will not re-prepare from
# the hub cache -- job 3857583's failure mode).

module load python/3.12-base cuda/12.8.1
source "$WORK/mist-venv/bin/activate"

# Compute nodes have no internet -> caches populated on the login node (see header).
export HF_HOME="/home/vault/b279bb/b279bb31/hf_cache_teacher"
# Prepared-dataset cache: prefer atuin, retreat to $HOME while atuin is unwritable
# (README "Temporary layout"); probed at runtime so this heals when the quota is fixed.
if touch "$WORK/hf_cache/datasets/.write-probe" 2>/dev/null; then
  rm -f "$WORK/hf_cache/datasets/.write-probe"
  export HF_DATASETS_CACHE="$WORK/hf_cache/datasets"
else
  export HF_DATASETS_CACHE="$HOME/hf_datasets_cache"
fi
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
