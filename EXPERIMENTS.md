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

> ## ⚠️ READ THIS BEFORE COMPARING ANY TWO NUMBERS IN THIS FILE
>
> **The `chrF` / `BERTScore` / `ROUGE-L` columns are dev *overall* scores, and dev overall is
> NOT a faithful predictor of test performance. 71% of the dev set is noise for our purposes.**
> Never pick a system by the overall column. Verified against the official test file
> 2026-07-15 (TEST_SET_ANALYSIS §5b):
>
> | dev source | rows | represents the test set? |
> |---|---|---|
> | `facebook/belebele` | 1,123 | ❌ **multiple choice — the test set has none at all** |
> | `CohereLabs/aya_dataset` | 978 | ❌ **golds are 24 words (p50); test `qa-oeg` asks for 120–180. No passage either, so it isn't `qa-context`. It resembles neither test task.** |
> | `copenlu/answerable_tydiqa` | 615 | ✅ the proxy for **`qa-context`** |
> | `FBK-MT/MCIF` | 165 | ✅ `qa-context` |
> | `wmt25-mist-oeg-gpt-4.1` | 97 | ✅ the proxy for **`qa-oeg`** (golds 175 words p50 — matches) |
>
> **Rules:** judge `qa-context` on **tydiqa**, judge `qa-oeg` on **OEG**, and treat belebele and
> aya as unscored. Only ~877/2,978 rows (29%) carry signal.
>
> Why this matters concretely — job 3859645 lost only 1.67 overall chrF, which reads as a mild
> regression, but the entire loss was belebele collapsing 20 points while every source that
> matters barely moved. The overall column hid the shape completely. It cuts the other way too:
> a system can win the overall column purely on belebele and be worse where it counts.
>
> ⚠️ The trap that produced this warning: README's sub-task table groups aya with OEG under
> "open-ended generation". That is a *task taxonomy*, not a claim that the two behave alike —
> do not read it as a dev→test proxy mapping.

## System comparison — read this one to choose a system

The tables further down are a chronological log, one row per job, scored on dev *overall*.
This one is the decision view: each candidate on the metric its test sub-task actually
deserves, with the 71% of dev that predicts nothing already excluded. Produced by re-scoring
the stored predictions CSVs with the current `evaluate.py` (no regeneration).

`qa-context` (8,640 test rows) is scored with **Exact Match + token F1** — the golds are
2-word extractions and chrF cannot resolve them. `qa-oeg` (2,359 test rows) keeps chrF /
BERTScore (175-word golds) and adds **word-budget compliance**, which is scored at test time
and which nothing else here can see.

| System | job | qa-context EM | qa-context F1 | qa-oeg chrF | qa-oeg BERT | budget in-band | (legacy overall chrF) |
|---|---|---|---|---|---|---|---|
| 9B 3-shot | 3822329 | **6.54** | **32.12** | 25.55 | 69.38 | 0% of 4 (−53% short) | 27.64 |
| 9B + gold-LoRA, 0-shot | 3857589 | _pending_ | | | | | 26.56 |
| 9B + gold-LoRA + 3-shot | 3858987 | _pending_ | | | | | 21.64 |
| 9B 3-shot, no lang-hint | 3859645 | _pending_ | | | | | 25.97 |
| 9B + distilled LoRA, 0-shot | — | _not built_ | | | | | |

**What the first row already changes:** 3-shot's tydiqa chrF of 38.94 was the strongest
number in this whole file and the reason `qa-context` looked solved. On the extraction
metric the same predictions score **6.54 EM**. The model almost never returns the gold span.
⚠️ Low EM does not by itself mean "wrong" — a verbose-but-correct answer ("the answer is
1650" vs gold "1650") scores 0 EM and partial F1, so some of this gap is style, not accuracy.
Whether that style is penalised depends on the organisers' automatic metric, **which we do
not know**. What is now certain is only that chrF was hiding the gap.

**Budget compliance, first measurement:** the 3-shot model writes **53% short** and lands in
band on 0 of 4 budgeted dev rows. n=4 is far too small to trust the magnitude (dev has almost
no budgeted rows — TEST_SET_ANALYSIS §4), but the direction contradicts the assumption behind
roadmap C's framing: the failure is under-writing, not over-writing.

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
| 3859645 | 2026-07-15 | 3-shot dev run WITHOUT lang-hint | Qwen3.5-9B, shots=3, lang-hint OFF | 2978 | 25.97 | 73.71 | 36.80 | `fewshot-9b.sbatch 3 --no-lang-hint`, 4h51. A/B against 3822329 (27.64/77.79/43.79, hint ON): −1.67 chrF overall, but the overall hides the shape. Per-source chrF vs hint-ON: belebele 52.70→**32.42**, tydiqa 38.94→34.39, MCIF 34.61→33.80, aya 24.19→23.85, OEG 25.55→**25.64**. The loss is almost entirely belebele (also BERTScore 92.06→83.49, ROUGE-L 79.85→64.69) — i.e. the hint was holding up MC *format*, the same thing few-shot was credited with teaching. On the sources the test set actually contains, dropping the hint costs ~1 chrF or less, and OEG is flat-to-up. |

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
| 3858987 + 3861569 | 2026-07-15 | LoRA adapter + few-shot dev eval (k=3) | Qwen3.5-9B + LoRA adapter 3822375, shots=3 | 2978 | 21.64 | 68.41 | 28.78 | `lora_eval.sbatch ... --shots 3`. **The demos and the adapter fight each other** — worse than *either* component alone on every source but OEG. vs plain 3-shot / adapter-0-shot: belebele 52.70/85.82→**26.66**, MCIF 34.61/49.26→**20.98**, tydiqa 38.94/19.53→**14.46**, aya 24.19/21.95→**19.94**; only OEG holds up (25.55/29.06→29.62, n=97). The hypothesis this run tested — demos recover tydiqa while the adapter keeps its belebele/MCIF/OEG gains — is dead: demos don't recover tydiqa, they sink it below adapter-only. Best explanation: the adapter was fine-tuned 0-shot (`train_lora.py` uses no demos), so a few-shot prompt is a format it never saw in training; it has specialised to 0-shot and the demos are out-of-distribution. **Do not stack the two.** Job 3858987 hit its 10h limit *after* generating all 2,978 rows but before scoring; the predictions CSV survived and 3861569 re-scored it in 59s (`evaluate.sbatch runs/predictions-lora-3858987.csv`) — scores above are from that. |

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

### Pipeline runs (data production — no dev metrics by design)

| Job ID | Date | Step | Config | Rows | Outcome |
|---|---|---|---|---|---|
| 3859277-79 | 2026-07-15 | teacher generation, whole corpus (3 shards) | Qwen3.5-35B-A3B bf16, `teacher_gen.sbatch --shard {1,2,3}/3`, lang-hint ON | 11,915 | _running_ (~68% at 23:00, ~270 rows/h → ~4h left; the earlier "ETA next morning" was a large under-estimate). Stable `--out runs/teacher-s{i}of3.jsonl` names → resumable on resubmit. **⚠️ Two things learned after these were submitted, neither worth killing them for — see the note below.** |
| 3859682 | 2026-07-15 | teacher generation, aya+oeg subset | Qwen3.5-122B-A10B-GPTQ-Int4 via vLLM, `teacher_gen_vllm.sbatch --source aya,oeg`, 2× a100_80 | 4,126 | ✅ 17m10s, all rows written, no failures → `runs/teacher122b-aya-oeg.jsonl` (gitignored). vLLM's batched-throughput edge (~250×/row vs the 35B transformers loop) holds at scale. |
| 3860144 | 2026-07-15 | filter calibration report on the 122B output | `filter_teacher.sbatch runs/teacher122b-aya-oeg.jsonl --report`, a40 | 4,126 | ✅ 1m16s. See distributions below. |

> **Note on 3859277-79 (35B shards) — two post-hoc findings, deliberately NOT acted on:**
>
> 1. **~4,577 of the 11,915 rows are belebele, and their teacher answers should never be used.**
>    They pass the filter 33.3% of the time (job 3861614) and every pass swaps a `2: <option>`
>    gold for prose — wrecking the format gold-SFT learned best (85.82) and buying nothing,
>    since the test set has no multiple choice. Handled at *filter* time with
>    `--gold-only belebele` (free) rather than by regenerating. The generation itself is
>    ~38% wasted compute; that is now sunk.
> 2. **The 35B-vs-122B split rests on a premise that turned out false.** IMPLEMENTATION_NOTES
>    §5.1 assigned the 35B to belebele/tydiqa/MCIF because "teacher choice barely matters
>    there (see §5.2)" — §5.2 claimed those rows always fail the filter. Measured: tydiqa
>    31.5%, MCIF 62.6% pass. So teacher choice *does* matter on the two sources that reach
>    the test set as `qa-context`, and they got the weaker, more hallucination-prone teacher
>    while the 122B spent 91% of its output on aya, which reaches neither test task.
>
> **Why not kill and redo with the 122B:** the 122B via vLLM does the whole 11,915-row split
> in well under 1h (3859682: 4,126 rows in 17m10s, ~13min of it one-time engine startup)
> versus ~50 GPU-hours for these three, so the redo is cheap *whenever* we do it. The 35B
> output is a resumable file on disk and stays useful as a comparison point. Killing 10h of
> running work to save 4h, on the untested assumption that the 122B is also better at
> *extraction* (§5.1's probes were knowledge questions), is the worse trade. Let them finish;
> regenerate `--source tydiqa,mcif` with the 122B afterwards and compare.

Filter calibration so far (from 3860144; final thresholds after the 35B shards land):

- Per-source score distributions vs gold — aya (n=3,763): chrF p25/p50/p75 = 11.4/22.1/33.4,
  BERTScore p50 = 66.9; **oeg (n=363): chrF p50 = 34.5, BERTScore p50 = 72.2**, much higher —
  plausibly because the oeg golds are themselves GPT-4.1 outputs, so a strong teacher
  style-matches them.
- Threshold grid (keep = chrF ≥ C **or** BERTScore ≥ B): 30/70 → 44.3% kept,
  20/70 → 60.9%, 30/75 → 36.3%. Candidate default **30/70**; consider a deliberately looser
  OEG-only threshold per the human-eval argument (IMPLEMENTATION_NOTES §5.3).

### Dev-set evals (standard metrics — comparable to the gold-SFT rows above)

| Job ID | Date | Experiment | Model / config | n | chrF | BERTScore | ROUGE-L | Notes |
|---|---|---|---|---|---|---|---|---|
| — | | distilled-adapter dev eval | Qwen3.5-9B + fresh LoRA on filtered teacher+gold mix, shots=0 | 2978 | _pending_ | | | The headline comparison: same recipe as 3822375/3857589, one variable (training targets). Waiting on the 35B shards → merged filter → `train_lora.py --data` → `lora_eval.sbatch`. |

Scope notes: the `sum` sub-task is handled by a teammate, this repo's experiments stay on `qa`
(incl. the OEG rows folded into it). The official test set is out (as of 2026-07-15), so once a
recipe wins on dev, retrain it on 100% of the sample data and run the test set with it.
