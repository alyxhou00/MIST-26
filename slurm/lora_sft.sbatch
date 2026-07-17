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
# Args after the adapter dir (or all args, if the first is a flag) go straight to
# train_lora.py -- that is how the distilled run is submitted:
#     sbatch slurm/lora_sft.sbatch --data data/sft-distilled.jsonl --no-lang-hint
# --no-lang-hint trains in run_test.py's test format; it must match how the adapter will be
# evaluated (see train_lora.py's --no-lang-hint help and EXPERIMENTS.md job 3858987).
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
# venv: prefer $WORK (documented home for regenerable software); fall back
# to $HOME when it has been archived off atuin to relieve the group inode quota.
MIST_VENV="$WORK/mist-venv"; [ -f "$MIST_VENV/bin/activate" ] || MIST_VENV="$HOME/mist-venv"
source "$MIST_VENV/bin/activate"

# Compute nodes have no internet -> use the cache populated on the login node.
export HF_HOME="$WORK/hf_cache"
# Prepared-dataset cache: prefer the canonical atuin copy, and only retreat to the $HOME
# copy while atuin writes fail (group file quota exceeded -- README "Temporary layout").
# Offline `datasets` writes a lock file on every load, so an unwritable cache dir kills the
# job in seconds (jobs 3857583, 3859591). Probed at runtime so the scripts heal themselves
# the moment the quota is fixed; baked in here (not left to the submitter to set).
if touch "$WORK/hf_cache/datasets/.write-probe" 2>/dev/null; then
  rm -f "$WORK/hf_cache/datasets/.write-probe"
  export HF_DATASETS_CACHE="$WORK/hf_cache/datasets"
else
  export HF_DATASETS_CACHE="$HOME/hf_datasets_cache"
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p adapters
# $1 is the adapter dir only when it isn't a flag -- otherwise every arg is passed straight
# through and the dir keeps its job-ID default (so `sbatch lora_sft.sbatch --data x.jsonl`
# can't silently train into an adapter directory literally named "--data").
case "${1:-}" in
  ""|-*) ADAPTER="adapters/qwen3.5-9b-qa-lora-$SLURM_JOB_ID" ;;
  *)     ADAPTER="$1"; shift ;;
esac
python scripts/train_lora.py --out "$ADAPTER" "$@"
# Hand on an *absolute* path: the adapter is written next to whichever clone ran this job, and
# the eval is not always submitted from that same clone (job 3865036 was submitted from atuin
# against this job's $HOME-trained adapter). A relative path in a copy-pasted hint resolves
# against the submitting clone and finds nothing -- or, worse, a stale adapter of the same name.
ADAPTER_ABS="$(readlink -f "$ADAPTER")"
echo "adapter: $ADAPTER_ABS"
echo "next:  sbatch slurm/lora_eval.sbatch $ADAPTER_ABS"
