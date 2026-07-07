# Experiment log

One row per SLURM job on the `qa` sub-task dev split (n=2978 unless narrowed by `--source`/
`--lang`/`--limit`). The job ID is the join key across `logs/<jobname>-<jobid>.out`,
`runs/predictions-*-<jobid>.csv` (or `predictions/predictions-<jobid>.csv` once promoted), and
this table -- see [README.md](README.md) for what each script/sbatch file does.

**Adding a row:** after a job finishes and its log is committed, add one row below with the job
ID, date, model/config, and the `overall chrF/BERTScore/ROUGE-L` line from
`logs/<jobname>-<jobid>.out`.

## Qwen3.5-2B

| Job ID | Date | Experiment | Model / config | n | chrF | BERTScore | ROUGE-L | Notes |
|---|---|---|---|---|---|---|---|---|
| 3786727 | 2026-06-26/27 | 0-shot baseline | Qwen3.5-2B, shots=0, **no lang-hint** | 2978 | 18.01 | -- | -- | Ran before `--lang-hint` existed; chrF only at the time. `job.sbatch` (pre-rename). |
| 3814759 | 2026-07-06 | re-score of 3786727 | (rescoring only, no generation) | 2978 | 18.01 | 62.21 | 12.51 | `evaluate.sbatch`, after `evaluate.py` gained BERTScore/ROUGE-L. Same predictions as 3786727. |
| 3817971 | 2026-07-06 | few-shot full dev run (k=3) | Qwen3.5-2B, shots=3, lang-hint ON | 2978 | 21.84 | 71.89 | 25.67 | `fewshot.sbatch`. Net improvement over the 18.01/62.21/12.51 baseline across all 27 languages. |

## Qwen3.5-9B (planned)

Under the organizers' 10B-parameter cap; benchmarked before LoRA SFT to establish whether the
bigger base is worth building on.

| Job ID | Date | Experiment | Model / config | n | chrF | BERTScore | ROUGE-L | Notes |
|---|---|---|---|---|---|---|---|---|
| _pending_ | | 0-shot full dev run | Qwen3.5-9B, shots=0, lang-hint ON | 2978 | | | | `0shot-9b.sbatch` |
| _pending_ | | few-shot full dev run (k=3) | Qwen3.5-9B, shots=3, lang-hint ON | 2978 | | | | `fewshot-9b.sbatch` |

## LoRA SFT (planned)

Distillation/fine-tuning is allowed by the organizers as long as the final model stays under
10B parameters -- see [scripts/train_lora.py](scripts/train_lora.py) for the training setup
(same train/dev split and prompt format as the prompting experiments above, evaluated
zero-shot so it's directly comparable).

| Job ID | Date | Experiment | Model / config | n | chrF | BERTScore | ROUGE-L | Notes |
|---|---|---|---|---|---|---|---|---|
| _pending_ | | full LoRA SFT training | Qwen3.5-9B LoRA, r=16/alpha=32, 2 epochs | n/a (train run) | n/a | n/a | n/a | `lora_sft.sbatch`; produces an adapter, scored separately below |
| _pending_ | | full LoRA SFT dev-set eval | Qwen3.5-9B + LoRA adapter, shots=0 | 2978 | | | | `lora_eval.sbatch <adapter path>` |
