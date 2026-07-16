#!/bin/bash -l
#SBATCH --job-name=mist-qa-0shot
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jenchien.hou@tum.de
#
# Full zero-shot dev run (the baseline experiment). One sbatch file per experiment, named
# after it -- this one was slurm/job.sbatch (job name mist-qa) before that convention
# existed; its committed log was renamed retroactively to match (mist-qa-3786727.out ->
# mist-qa-0shot-3786727.out), so SLURM job IDs in old logs still line up with git history.
# Run from the repo root ($WORK/MIST-26) on Alex (NHR@FAU):  sbatch slurm/0shot.sbatch
# Assumes the one-time login-node setup (venv + cached model/dataset) is done; see README.
#
# Log goes to logs/ (committed -- $WORK has no backup/retention guarantee, so `git add
# logs/*.out && git commit && git push` from the cluster once the job finishes). Predictions
# CSV goes to runs/ (gitignored scratch -- promote it to predictions/ only if worth keeping).
#
# Time limit 8h: the real run (job 3786727) took 5h36m; 8h keeps ~40% margin. (Was 12h,
# set before any 2B throughput was known.)

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

# Write per-example results into the repo's (gitignored) runs/ dir, keyed by job ID so
# runs don't overwrite each other -- same relative path locally and on the cluster.
mkdir -p runs
CSV="runs/predictions-$SLURM_JOB_ID.csv"
python scripts/benchmark.py --out "$CSV"   # writes predictions to $CSV; add --limit 50 for a quick run
python scripts/evaluate.py "$CSV"          # reads $CSV, prints chrF/BERTScore/ROUGE-L (overall + per-source + per-language)
# Both scripts' stdout+stderr are captured by SLURM into logs/%x-%j.out (see --output above).
