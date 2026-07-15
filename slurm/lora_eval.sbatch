#!/bin/bash -l
#SBATCH --job-name=mist-qa-lora-eval
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jenchien.hou@tum.de
#
# Full dev-set inference + scoring for a trained LoRA adapter (from slurm/lora_sft.sbatch) --
# kept separate from training so one adapter can be re-evaluated (e.g. combined with --shots
# later) without retraining. Zero-shot by default (--shots 0, benchmark.py's own default),
# matching how the adapter was trained (see scripts/train_lora.py) -- directly comparable to
# the 0-shot/few-shot baselines.
# Run from the repo root ($WORK/MIST-26) on Alex (NHR@FAU):
#     sbatch slurm/lora_eval.sbatch adapters/qwen3.5-9b-qa-lora-<jobid> [extra benchmark.py flags]
# Everything after the adapter dir is passed through to benchmark.py, e.g. `--shots 3` to
# evaluate the adapter combined with few-shot demonstrations.
# Assumes slurm/setup.sh has been (re-)run since it now also installs peft (requirements.txt)
# and caches Qwen3.5-9B -- see README.
#
# Time limit 10h, same as slurm/0shot-9b.sbatch (this is the same generation workload, just
# with a LoRA adapter attached -- adapter forward-pass overhead is negligible next to the base
# 9B model's). 0shot-9b's real run (job 3822324) measured ~6.7h projected total -- see that
# file's comment for the math.
#
# Sampling params (--temperature 0.7 --top-p 0.8) come from Qwen3.5-9B's own card. --model
# must match whatever train_lora.py used for this adapter (default Qwen3.5-9B) -- the adapter
# was trained against that base's module names/shapes.

module load python/3.12-base cuda/12.8.1
source "$WORK/mist-venv/bin/activate"

# Compute nodes have no internet -> use the cache populated on the login node.
export HF_HOME="$WORK/hf_cache"
# Prepared-dataset cache must be somewhere writable: in offline mode `datasets` does NOT
# re-prepare from the hub cache, and it also takes a lock file on every load -- $WORK/hf_cache
# is unwritable while the atuin group quota is exceeded (README "Temporary layout"; this is
# what killed jobs 3857583 and 3859591). Baked in here (not left to the submitter to set) so
# it can't be forgotten again.
export HF_DATASETS_CACHE="$HOME/hf_datasets_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ADAPTER="$1"
if [ -z "$ADAPTER" ]; then
  echo "usage: sbatch slurm/lora_eval.sbatch <adapter dir> [extra benchmark.py flags]" >&2
  exit 1
fi
shift

mkdir -p runs
CSV="runs/predictions-lora-$SLURM_JOB_ID.csv"
python scripts/benchmark.py --model Qwen/Qwen3.5-9B --lora "$ADAPTER" \
  --temperature 0.7 --top-p 0.8 --out "$CSV" "$@"
python scripts/evaluate.py "$CSV"          # prints chrF/BERTScore/ROUGE-L (overall + per-source + per-language)
# Both scripts' stdout+stderr are captured by SLURM into logs/%x-%j.out (see --output above).
