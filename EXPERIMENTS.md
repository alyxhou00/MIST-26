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

## Qwen3.5-9B

Under the organizers' 10B-parameter cap; benchmarked before LoRA SFT to establish whether the
bigger base is worth building on.

| Job ID | Date | Experiment | Model / config | n | chrF | BERTScore | ROUGE-L | Notes |
|---|---|---|---|---|---|---|---|---|
| 3822324 | 2026-07-08 | 0-shot full dev run | Qwen3.5-9B, shots=0, lang-hint ON | 2978 | 23.12 | 66.04 | 22.75 | `0shot-9b.sbatch`, 6h01. Beats 2B 0-shot (18.01/62.21/12.51) across the board. |
| 3822329 | 2026-07-08 | few-shot full dev run (k=3) | Qwen3.5-9B, shots=3, lang-hint ON | 2978 | 27.64 | 77.79 | 43.79 | `fewshot-9b.sbatch`, 3h42. Best run so far. Gains concentrated in belebele (chrF 17.69→52.70) and tydiqa (21.88→38.94); aya essentially flat (24.03→24.19) — few-shot teaches answer *format*, not open-ended generation. Matches smoke 3822323 (aya/hin only: 0-shot 24.34 vs 3-shot 22.03, i.e. no aya gain). |

## LoRA SFT on gold answers

Fine-tuning is allowed by the organizers as long as the final model stays under 10B
parameters -- see [scripts/train_lora.py](scripts/train_lora.py) for the training setup
(same train/dev split and prompt format as the prompting experiments above, evaluated
zero-shot so it's directly comparable). Training targets here are the dataset's **gold
answers** -- the contrast is the distilled variant below, which is the same recipe with
teacher outputs as targets instead.

| Job ID | Date | Experiment | Model / config | n | chrF | BERTScore | ROUGE-L | Notes |
|---|---|---|---|---|---|---|---|---|
| 3822375 | 2026-07-08 | full LoRA SFT training | Qwen3.5-9B LoRA, r=16/alpha=32, 2 epochs | n/a (train run) | n/a | n/a | n/a | `lora_sft.sbatch`, 6h35, train_loss 0.664. Adapter: `adapters/qwen3.5-9b-qa-lora-3822375` (on the cluster; gitignored). Smoke pipeline check was 3822331. |
| 3858987 | 2026-07-15 | LoRA adapter + few-shot dev eval (k=3) | Qwen3.5-9B + LoRA adapter 3822375, shots=3 | 2978 | _running_ | | | `lora_eval.sbatch` with the new extra-args passthrough (`--shots 3`). Tests whether few-shot demos stack with the adapter: hoping demos recover tydiqa (38.94 with plain 3-shot vs 19.53 adapter-only) while keeping the adapter's belebele/MCIF/OEG format gains. Same `$HOME` clone + `HF_DATASETS_CACHE` workaround as 3857589. |
| 3857589 | 2026-07-14/15 | full LoRA SFT dev-set eval | Qwen3.5-9B + LoRA adapter 3822375, shots=0 | 2978 | 26.56 | 79.15 | 48.00 | `lora_eval.sbatch`, 6h52. Below 9B 3-shot on chrF (27.64) but above it on BERTScore (77.79) and ROUGE-L (43.79). Strongly complementary per-source vs 3-shot: belebele 52.70→**85.82**, MCIF 34.61→**49.26**, OEG 25.55→**29.06** (ROUGE-L 10.96→37.38 — gold-SFT *does* move OEG, unlike prompting), but tydiqa **38.94→19.53** (below even the 0-shot base's 21.88) and aya 24.19→21.95. Next: adapter+3-shot to see if the demos recover tydiqa without losing the format gains. Submitted from the `$HOME/MIST-26` clone with `HF_DATASETS_CACHE=$HOME/hf_datasets_cache` (atuin group file quota exceeded — see README). First attempt 3857583 died in 38s: offline `datasets` does **not** re-prepare from the hub cache, it needs the prepared cache under `HF_DATASETS_CACHE`, so the 79MB prepared cache was copied to `$HOME` first. |

## Test-format alignment (official test set)

The official test prompts (README "The official test set") are self-contained conversational
prompts with embedded format/length/language instructions — a different distribution from our
templated dev prompts. [`scripts/run_test.py`](scripts/run_test.py) feeds them verbatim (no
template, no lang-hint by default) and writes submission-format `{id, output}` JSONL.

| Job ID | Date | Experiment | Model / config | n | chrF | BERTScore | ROUGE-L | Notes |
|---|---|---|---|---|---|---|---|---|
| 3859059 | 2026-07-15 | test-set smoke (`smoke-run-test.sbatch`) | Qwen3.5-9B base, verbatim prompt, 10× qa-context arb + 10× qa-oeg bho | 20 | n\a (no golds) | | | Pipeline works end-to-end, outputs eyeballed. Findings: (1) test prompts *themselves* instruct 'if the answer is not in the passage, write only "no answer"' and end with an explicit "answer in <language>" — so an unanswerable-detection behavior and the output language are part of the task; (2) the base model **false-refuses**: 4/10 arb qa-context rows got "لا توجد إجابة" including a definitional question whose answer is verbatim in the passage; (3) **Bhojpuri OEG output drifts into Hindi (most rows), Nepali and Maithili** — the feared bho→hin slide is real and worse (roadmap D confirmed); (4) word budgets are mostly violated (150→235w, 120–150→282w, 100→60w; only 1/10 in range — roadmap C confirmed); (5) OEG outputs come out in heavy markdown (###, **bold**, emoji) — fine or not for human eval, TBD. |
| 3859058 | 2026-07-15 | 3-shot dev run WITHOUT lang-hint | Qwen3.5-9B, shots=3, lang-hint OFF | 2978 | _running_ | | | `fewshot-9b.sbatch 3 --no-lang-hint` (new passthrough). A/B against 3822329 (27.64/77.79/43.79, hint ON): measures how much of our best config leans on the lang-hint system turn. Matters because test prompts embed their own language instruction, so the hint is redundant-at-best there; a big drop would mean the model depends on our scaffolding. |

## LoRA SFT on distilled data — teacher outputs + gold mix (planned next)

Key finding from the 9B runs above: few-shot's gain is almost entirely *answer format*
(belebele chrF 17.69→52.70, tydiqa 21.88→38.94) while open-ended generation (aya) stays flat
(24.03→24.19) — prompting and gold-SFT both lack a lever for answer *quality* there. The
organizers allow distillation as long as the final model is <10B, so the plan is sequence-level
KD: (1) generate teacher answers on the qa train split (~11.9K rows) with a strong teacher
(Qwen3.5-32B on one A40, or a quantized 72B), (2) quality-filter against the golds
(chrF/BERTScore threshold), (3) LoRA SFT the 9B on the filtered teacher outputs **mixed with
golds**, as a *fresh* adapter (not continued from 3822375) — same recipe as `train_lora.py`,
only the training targets change. That keeps it directly comparable to the gold-only adapter
(3822375) above: same starting point, same recipe, one variable (the data).

Scope notes: the `sum` sub-task is handled by a teammate, this repo's experiments stay on `qa`
(incl. the OEG rows folded into it). The official test set is out (as of 2026-07-15), so once a
recipe wins on dev, retrain it on 100% of the sample data and run the test set with it.
