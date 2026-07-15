#!/bin/bash -l
#SBATCH --job-name=mist-qa-smoke-lora
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jenchien.hou@tum.de
#
# Cheap end-to-end LoRA SFT pipeline check -- train on 200 rows for 1 epoch, then evaluate on
# 50 dev rows, before committing to the full slurm/lora_sft.sbatch (18h) + slurm/lora_eval.sbatch
# run. Confirms: the model/adapter fit in memory, tokenization+masking doesn't crash on real
# data, save/reload round-trips through PeftModel correctly.
# Run from the repo root ($WORK/MIST-26):  sbatch slurm/smoke-lora.sbatch
# Assumes slurm/setup.sh has been (re-)run since it now also installs peft (requirements.txt)
# and caches Qwen3.5-9B -- see README.

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

mkdir -p runs adapters
ADAPTER="adapters/smoke-lora-$SLURM_JOB_ID"
CSV="runs/smoke-lora-$SLURM_JOB_ID.csv"

echo "=== training (200 rows, 1 epoch) ==="
python scripts/train_lora.py --limit 200 --epochs 1 --out "$ADAPTER"

echo "=== evaluating (50 dev rows) ==="
# --model must match train_lora.py's (default Qwen3.5-9B) -- the adapter was trained against
# that base's module names/shapes, so benchmark.py needs to load the same base before attaching it.
python scripts/benchmark.py --model Qwen/Qwen3.5-9B --lora "$ADAPTER" \
  --temperature 0.7 --top-p 0.8 --limit 50 --out "$CSV"
python scripts/evaluate.py "$CSV"
