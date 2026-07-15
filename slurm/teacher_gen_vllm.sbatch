#!/bin/bash -l
#SBATCH --job-name=mist-qa-teacher122b
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2
#SBATCH --constraint=a100_80
#SBATCH --time=02:00:00
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
# Time limit 2h: the real aya+oeg run (job 3859682, 4,126 rows) took 17m10s end to end --
# ~13min of that was one-time engine startup (load 79GB of weights, CUDA-graph capture) and
# only ~4min was decode. Even the full 11,915-row split projects to well under 1h; 2h is
# pure margin. Resubmit with the same --out to resume (skips qa_idx already present).

module load python/3.12-base cuda/12.8.1
source "$HOME/vllm-venv/bin/activate"

# Compute nodes have no internet -> caches populated on the login node.
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
