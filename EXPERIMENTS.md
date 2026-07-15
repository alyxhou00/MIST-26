# Experiment log

One row per SLURM job on the `qa` sub-task dev split (n=2978 unless narrowed by `--source`/
`--lang`/`--limit`). The job ID is the join key across `logs/<jobname>-<jobid>.out`,
`runs/predictions-*-<jobid>.csv` (or `predictions/predictions-<jobid>.csv` once promoted), and
this table -- see [README.md](README.md) for what each script/sbatch file does.

**What gets a row:** full runs that support a meaningful comparison (baselines, ablations,
full generation/training passes). Smoke tests, pipeline checks and failed/aborted jobs are
NOT logged here -- their logs are still committed under `logs/`, infra failure post-mortems
live in [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) §6, and the teacher-selection
smoke comparison lives in IMPLEMENTATION_NOTES §5.1.

**Adding a row:** after a job finishes and its log is committed, add one row below with the job
ID, date, model/config, and the `overall chrF/BERTScore/ROUGE-L` line from
`logs/<jobname>-<jobid>.out`.

## Qwen3.5-2B

| Job ID | Date | Experiment | Model / config | n | chrF | BERTScore | ROUGE-L | Notes |
|---|---|---|---|---|---|---|---|---|
| 3786727 | 2026-06-26/27 | 0-shot baseline | Qwen3.5-2B, shots=0, **no lang-hint** | 2978 | 18.01 | 62.21 | 12.51 | 5h36. Ran before `--lang-hint` existed; chrF only at the time -- BERTScore/ROUGE-L added by re-scoring the same predictions after `evaluate.py` gained those metrics (job 3814759, `evaluate.sbatch`, 1m11s). |
| 3817971 | 2026-07-06 | few-shot full dev run (k=3) | Qwen3.5-2B, shots=3, lang-hint ON | 2978 | 21.84 | 71.89 | 25.67 | `fewshot.sbatch`, 3h29. Net improvement over the 18.01/62.21/12.51 baseline across all 27 languages. |

## Qwen3.5-9B

Under the organizers' 10B-parameter cap; benchmarked before LoRA SFT to establish whether the
bigger base is worth building on.

| Job ID | Date | Experiment | Model / config | n | chrF | BERTScore | ROUGE-L | Notes |
|---|---|---|---|---|---|---|---|---|
| 3822324 | 2026-07-08 | 0-shot full dev run | Qwen3.5-9B, shots=0, lang-hint ON | 2978 | 23.12 | 66.04 | 22.75 | `0shot-9b.sbatch`, 6h01. Beats 2B 0-shot (18.01/62.21/12.51) across the board. |
| 3822329 | 2026-07-08 | few-shot full dev run (k=3) | Qwen3.5-9B, shots=3, lang-hint ON | 2978 | 27.64 | 77.79 | 43.79 | `fewshot-9b.sbatch`, 3h42. Best chrF so far. Gains concentrated in belebele (chrF 17.69→52.70) and tydiqa (21.88→38.94); aya essentially flat (24.03→24.19) — few-shot teaches answer *format*, not open-ended generation. |
| 3859645 | 2026-07-15 | 3-shot dev run WITHOUT lang-hint | Qwen3.5-9B, shots=3, lang-hint OFF | 2978 | _running_ | | | `fewshot-9b.sbatch 3 --no-lang-hint`. A/B against 3822329 (27.64/77.79/43.79, hint ON): measures how much of our best config leans on the lang-hint system turn — matters because the official test prompts embed their own language instruction, so the hint is redundant-at-best there (TEST_SET_ANALYSIS.md §4). |

## LoRA SFT on gold answers

Fine-tuning is allowed by the organizers as long as the final model stays under 10B
parameters -- see [scripts/train_lora.py](scripts/train_lora.py) for the training setup
(same train/dev split and prompt format as the prompting experiments above, evaluated
zero-shot so it's directly comparable). Training targets here are the dataset's **gold
answers** -- the contrast is the distilled variant below, which is the same recipe with
teacher outputs as targets instead.

| Job ID | Date | Experiment | Model / config | n | chrF | BERTScore | ROUGE-L | Notes |
|---|---|---|---|---|---|---|---|---|
| 3822375 | 2026-07-08 | full LoRA SFT training | Qwen3.5-9B LoRA, r=16/alpha=32, 2 epochs | n/a (train run) | n/a | n/a | n/a | `lora_sft.sbatch`, 6h35, train_loss 0.664, 29,097,984 trainable params (0.31% of the 9,438,911,728-param base). Adapter: `adapters/qwen3.5-9b-qa-lora-3822375` (on the cluster; gitignored). |
| 3857589 | 2026-07-14/15 | full LoRA SFT dev-set eval | Qwen3.5-9B + LoRA adapter 3822375, shots=0 | 2978 | 26.56 | 79.15 | 48.00 | `lora_eval.sbatch`, 6h52. Below 9B 3-shot on chrF (27.64) but above it on BERTScore (77.79) and ROUGE-L (43.79). Strongly complementary per-source vs 3-shot: belebele 52.70→**85.82**, MCIF 34.61→**49.26**, OEG 25.55→**29.06** (ROUGE-L 10.96→37.38 — gold-SFT *does* move OEG, unlike prompting), but tydiqa **38.94→19.53** (below even the 0-shot base's 21.88) and aya 24.19→21.95. |
| 3858987 | 2026-07-15 | LoRA adapter + few-shot dev eval (k=3) | Qwen3.5-9B + LoRA adapter 3822375, shots=3 | 2978 | _running_ | | | `lora_eval.sbatch ... --shots 3`. Tests whether few-shot demos stack with the adapter: hoping demos recover tydiqa (38.94 with plain 3-shot vs 19.53 adapter-only) while keeping the adapter's belebele/MCIF/OEG gains. |

## LoRA SFT on distilled data — teacher outputs + gold mix (in progress)

Key finding from the 9B runs above: few-shot's gain is almost entirely *answer format*
(belebele chrF 17.69→52.70, tydiqa 21.88→38.94) while open-ended generation (aya) stays flat
(24.03→24.19) — prompting lacks a lever for answer *quality* there. The organizers allow
distillation as long as the final model is <10B, so the plan is sequence-level KD:
(1) generate teacher answers on the qa train split (11,915 rows; same seed-42 80/20 split,
dev untouched), (2) quality-filter against the golds ([scripts/filter_teacher.py](scripts/filter_teacher.py),
per-row sentence chrF OR BERTScore, thresholds calibrated per source via `--report`),
(3) LoRA SFT the 9B on the filtered teacher+gold mix (`train_lora.py --data`) as a *fresh*
adapter (not continued from 3822375) — same recipe, one variable (the data), directly
comparable to the gold-only adapter above.

**Teacher selection** (full 3-way smoke comparison and the transformers-GPTQ dead end:
IMPLEMENTATION_NOTES §5.1): Qwen3.5-35B-A3B bf16 (1× a100_80, transformers, ~14 s/row) for
the whole corpus, plus **Qwen3.5-122B-A10B-GPTQ-Int4 via vLLM** (2× a100_80, ~250× faster
per row batched) for the aya+oeg subset — the only teacher that got both knowledge probes
right, and knowledge-grounded open-ended rows are exactly where a better teacher raises the
filter pass rate. Teacher weights live on `$HPCVAULT` (README "Temporary layout").

| Job ID | Date | Experiment | Model / config | n | chrF | BERTScore | ROUGE-L | Notes |
|---|---|---|---|---|---|---|---|---|
| 3859277-79 | 2026-07-15 | full teacher generation, 3 shards | Qwen3.5-35B-A3B bf16, `--shard {1,2,3}/3`, lang-hint ON | 11,915 | _running_ | | | `teacher_gen.sbatch --shard i/3 --out runs/teacher-s{i}of3.jsonl` (stable names → resumable on resubmit). ~16h/shard projected from observed rate. |
| 3859682 | 2026-07-15 | full teacher generation: aya+oeg via 122B/vLLM | Qwen3.5-122B-A10B-GPTQ-Int4, `teacher_gen_vllm.sbatch --source aya,oeg`, 2× a100_80 | 4,126 | n/a (generation only) | | | **COMPLETED in 17m10s** (vs ~16h/shard for the 35B transformers loop) — vLLM's batched throughput advantage holds at scale. All 4,126 rows written, no generation failures. Output: `runs/teacher122b-aya-oeg.jsonl` (gitignored, feeds `filter_teacher.py`). |
| 3860144 | 2026-07-15 | filter report on 122B aya+oeg output | `filter_teacher.sbatch runs/teacher122b-aya-oeg.jsonl --report` (a40) | 4,126 | n/a (report) | | | 1m16s. Distributions: aya (n=3763) chrF p25/p50/p75 = 11.4/22.1/33.4, BERT p50 = 66.9; **oeg (n=363) scores much higher — chrF p50 = 34.5, BERT p50 = 72.2** (plausible: the oeg golds are themselves GPT-4.1 outputs, so a strong teacher style-matches them). Threshold grid (keep = chrF≥C OR BERT≥B): 30/70 → 44.3%, 20/70 → 60.9%, 30/75 → 36.3%. Candidate default 30/70; consider a deliberately looser OEG-only threshold per the human-eval argument (IMPLEMENTATION_NOTES §5.3). Final choice after the 35B full-corpus shards land. |

Scope notes: the `sum` sub-task is handled by a teammate, this repo's experiments stay on `qa`
(incl. the OEG rows folded into it). The official test set is out (as of 2026-07-15), so once a
recipe wins on dev, retrain it on 100% of the sample data and run the test set with it.
