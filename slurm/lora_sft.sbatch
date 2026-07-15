#!/bin/bash -l
#SBATCH --job-name=mist-qa-lora-sft
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jenchien.hou@tum.de
#
# Full LoRA SFT run on Qwen3.5-9B: trains scripts/train_lora.py on the train 80% of the qa
# split (same seed-42 split benchmark.py evaluates against), 2 epochs by default. Smoke-test
# first with slurm/smoke-lora.sbatch. Run from the repo root ($WORK/MIST-26):
#     sbatch slurm/lora_sft.sbatch                              # first submission
#     sbatch slurm/lora_sft.sbatch adapters/qwen3.5-9b-qa-lora-<first job's ID>   # resume it
# Assumes slurm/setup.sh has been (re-)run since it now also installs peft (requirements.txt)
# and caches Qwen3.5-9B -- see README.
#
# 10h budget: job 3822375 (the first real run of this file, 18h budget at the time) logged
# Trainer's own progress bar at step 21/1490 after 5:33 elapsed, projecting ~6.5h total -- 10h
# leaves solid margin without the 18h of mostly-idle reserved allocation. If this isn't enough:
# the script checkpoints every --save-steps (200) into ADAPTER and auto-resumes from the latest
# checkpoint there -- but only if the *next* submission points at that same directory, hence
# taking it as $1 above (a fresh job ID would otherwise pick a fresh, empty adapter dir and
# restart from scratch).
#
# Output: adapter-only weights (small) to ADAPTER (default keyed by this job's ID; gitignored
# scratch, like runs/ -- promote it elsewhere if you want to keep it beyond the cluster). Then
# score it with slurm/lora_eval.sbatch <that path>.

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

mkdir -p adapters
ADAPTER="${1:-adapters/qwen3.5-9b-qa-lora-$SLURM_JOB_ID}"
python scripts/train_lora.py --out "$ADAPTER"
echo "adapter: $ADAPTER"
echo "next:  sbatch slurm/lora_eval.sbatch $ADAPTER"
