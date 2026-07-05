#!/bin/bash -l
# One-time setup on the Alex (NHR@FAU) *login* node.
#
# Why this script exists: the compute nodes have no internet, so everything that needs to be
# downloaded -- the Python deps, the model, and the dataset -- must be fetched here on the login
# node first. job.sbatch then runs fully offline against this venv and cache.
#
#   ssh alex && cd $WORK/MIST-26 && bash setup.sh
set -euo pipefail

module load python/3.12-base cuda/12.8.1

# venv + deps
python -m venv "$WORK/mist-venv"
source "$WORK/mist-venv/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt

# Pre-cache the dataset and models so the offline compute node finds them.
export HF_HOME="$WORK/hf_cache"
python -c "from datasets import load_dataset; load_dataset('pinzhenchen/wmt26-mist-sample')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-2B')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('bert-base-multilingual-cased')"  # for evaluate.py's BERTScore

# Qwen3.5 is new; if this released transformers doesn't know "qwen3_5", install from source.
if ! python -c "from transformers import AutoConfig; AutoConfig.from_pretrained('Qwen/Qwen3.5-2B')" 2>/dev/null; then
  echo "[setup] transformers doesn't recognise qwen3_5 yet -- installing from source"
  pip install --upgrade "git+https://github.com/huggingface/transformers.git"
fi

echo "[setup] done -- now submit:  sbatch job.sbatch"
