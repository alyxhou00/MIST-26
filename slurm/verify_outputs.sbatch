#!/bin/bash -l
#SBATCH --job-name=mist-qa-verify
#SBATCH --partition=a40
#SBATCH --gres=gpu:a40:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jenchien.hou@tum.de
#
# Roadmap C/D acceptance checks on run_test.py outputs (scripts/verify_outputs.py):
# word-budget compliance, bho_lid on qa-oeg bho rows, and the qa-context script/refusal/
# one-sentence/contrastive-function-word checks. Pure CPU and a few minutes, but it still
# goes through SLURM -- nothing computes on the login node.
#
#     sbatch slurm/verify_outputs.sbatch runs/test-qaoeg-a.jsonl runs/test-qaoeg-b.jsonl ...
#
# Pass several outputs in one call: the report is printed per file in argument order, which
# is how the C+D vs plain vs C-only comparison tables are read off a single log.
module load python/3.12-base cuda/12.8.1
MIST_VENV="$WORK/mist-venv"; [ -f "$MIST_VENV/bin/activate" ] || MIST_VENV="$HOME/mist-venv"
source "$MIST_VENV/bin/activate"
export HF_HOME="$WORK/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python scripts/verify_outputs.py "$@"
