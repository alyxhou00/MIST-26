#!/bin/bash -l
#SBATCH --job-name=mist-qa-lora-eval
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jenchien.hou@tum.de
#
# Full dev-set inference + scoring for a trained LoRA adapter (from slurm/lora_sft.sbatch) --
# kept separate from training so one adapter can be re-evaluated (e.g. combined with --shots
# later) without retraining. Zero-shot by default (--shots 0, benchmark.py's own default),
# matching how the adapter was trained (see scripts/train_lora.py) -- directly comparable to
# the 0-shot/few-shot baselines.
# Run from the repo root ($WORK/MIST-26) on Alex (NHR@FAU), and give the adapter as an ABSOLUTE
# path -- lora_sft.sbatch prints one ready to paste:
#     sbatch slurm/lora_eval.sbatch /home/hpc/.../MIST-26/adapters/qwen3.5-9b-qa-lora-<jobid> [flags]
# A relative path resolves against *this* clone, but the adapter lives next to the clone that
# trained it, which is not always the same one (job 3865036, submitted from atuin against a
# $HOME-trained adapter, was only correct because the path was absolute). benchmark.py now
# rejects a --lora dir with no adapter_config.json rather than let a mismatch reach the GPU.
# Everything after the adapter dir is passed through to benchmark.py, e.g. `--shots 3` to
# evaluate the adapter combined with few-shot demonstrations, or `--no-lang-hint` to match an
# adapter trained in the test format.
# Assumes slurm/setup.sh has been (re-)run since it now also installs peft (requirements.txt)
# and caches Qwen3.5-9B -- see README.
#
# Time limit 16h. It was 10h (copied from slurm/0shot-9b.sbatch on the theory that the adapter
# forward-pass overhead is negligible), and job 3858987 -- adapter + `--shots 3` -- died at
# 10:00:05 having generated all 2,978 rows but not yet scored them. Measured generation times
# on this dev set: 3-shot no adapter 3h42 (3822329), adapter 0-shot 6h52 (3857589), adapter
# 3-shot ~9h59 (3858987, scoring adds only ~1min -- see 3861569). Few-shot demos lengthen every
# prompt, and that cost lands on top of the adapter's, so this config is ~2.7x the plain 3-shot
# run and 10h had no margin at all.
#
# If a run does time out after generating, do NOT rerun it: benchmark.py has already written
# runs/predictions-lora-<jobid>.csv, so re-score that instead --
#     sbatch slurm/evaluate.sbatch runs/predictions-lora-<jobid>.csv
# which is how 3858987's numbers were recovered in 59s rather than in another 10h.
#
# Sampling params (--temperature 0.7 --top-p 0.8) come from Qwen3.5-9B's own card. --model
# must match whatever train_lora.py used for this adapter (default Qwen3.5-9B) -- the adapter
# was trained against that base's module names/shapes.

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

ADAPTER="$1"
if [ -z "$ADAPTER" ]; then
  echo "usage: sbatch slurm/lora_eval.sbatch <adapter dir> [extra benchmark.py flags]" >&2
  exit 1
fi
shift

# --data is passed through to benchmark.py, which now *requires* it (there is no default dev
# set any more) and rejects the run in seconds if it is absent -- so there is deliberately no
# guard here to drift out of sync with it. Pass `--data data/dev_v2.jsonl --no-lang-hint` for a
# v2 adapter, or `--data hf-sample` to reproduce a pre-v2 run, which warns loudly because that
# split leaks into train_v2 (DATA_AUDIT.md #2).

mkdir -p runs
CSV="runs/predictions-lora-$SLURM_JOB_ID.csv"
python scripts/benchmark.py --model Qwen/Qwen3.5-9B --lora "$ADAPTER" \
  --temperature 0.7 --top-p 0.8 --out "$CSV" "$@"
python scripts/evaluate.py "$CSV"          # prints chrF/BERTScore/ROUGE-L (overall + per-source + per-language)
# Both scripts' stdout+stderr are captured by SLURM into logs/%x-%j.out (see --output above).
