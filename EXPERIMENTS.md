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
> NOT a faithful predictor of test performance.** Never pick a system by the overall column —
> use the decision table below it. Checked against the official test file (TEST_SET_ANALYSIS
> §5b):
>
> | dev source | rows | represents the test set? |
> |---|---|---|
> | `facebook/belebele` | 1,123 | ❌ **multiple choice — the test set has none at all.** 38% of dev, predicts nothing. |
> | `copenlu/answerable_tydiqa` | 615 | ✅ **`qa-context`** — and score it with EM/F1, not chrF |
> | `FBK-MT/MCIF` | 165 | ✅ `qa-context` |
> | `wmt25-mist-oeg-gpt-4.1` | 97 | ✅ **`qa-oeg`, long-form end** (golds 175 words p50) — ~87% of qa-oeg prompts |
> | `CohereLabs/aya_dataset` | 978 | ✅ **`qa-oeg`, short-answer end** (golds 24 words p50) — ~13% of qa-oeg prompts |
>
> **Rules:** judge `qa-context` on **tydiqa + MCIF**, judge `qa-oeg` on **OEG and aya as two
> separate columns** (they measure opposite ends of one spectrum — never average them), and
> treat **belebele** as unscored. ⚠️ dev's weighting is inverted against the test mix: aya gets
> 978 rows for ~13% of qa-oeg while OEG gets 97 for ~87%.
>
> Why this matters concretely — job 3859645 lost only 1.67 overall chrF, which reads as a mild
> regression, but the entire loss was belebele collapsing 20 points while every source that
> matters barely moved. The overall column hid the shape completely. It cuts the other way too:
> a system can win the overall column purely on belebele and be worse where it counts.
>
> **Correction, 2026-07-16.** This box previously said "71% of dev is noise" and excluded aya
> on the grounds that qa-oeg "asks for 120-180 words" while aya's golds are 24. That was wrong.
> Only ~20% of qa-oeg prompts carry a word budget; the task is a spectrum, and ~13% of its 100
> unique prompts are short-answer trivia and lists ("name a country with no vowels in its
> name", "list the top 5 landmarks") — exactly aya's shape. The error was generalising from the
> budgeted 20% to the whole task without reading the other 80%, when the whole task is only 100
> unique prompts and could have been enumerated in one pass. The belebele half of the warning
> also doesn't really stand (please ask Alyssa why; she knows); the aya half is retracted.
> By the way, this file needs to be cleaned up after doubts are resolved.

## System comparison — read this one to choose a system

The tables further down are a chronological log, one row per job, scored on dev *overall*.
This one is the decision view: each candidate on the metric its test sub-task actually
deserves, with the 71% of dev that predicts nothing already excluded. Produced by re-scoring
the stored predictions CSVs with the current `evaluate.py` (no regeneration).

**Rescored per source 2026-07-16 (jobs 3864996-99), and it settles the routing.** `qa-context`
used to be reported as one pooled column (tydiqa + MCIF, n=780). That pooling was 79% tydiqa —
the *monolingual* source, worth ~4% of a test sub-task that is 96% cross-lingual. Split apart,
the metric disagreement that blocked this decision for days **evaporates**.

**`qa-context` — ✅ MCIF, the FAITHFUL proxy (cross-lingual, n=165). Route on this column.**

| System | job | **EM** | **F1** | **chrF** | **BERT** | √(EM·chrF) |
|---|---|---|---|---|---|---|
| **9B + gold-LoRA, 0-shot** | 3857589 | **21.82** | **57.92** | **49.26** | **86.41** | **32.78** |
| 9B + gold-LoRA + 3-shot | 3858987 | 12.12 | 29.70 | 20.98 | 69.89 | 15.95 |
| 9B 3-shot | 3822329 | 0.61 | 28.15 | 34.61 | 74.38 | 4.58 |
| 9B 3-shot, no lang-hint | 3859645 | 0.61 | 27.16 | 33.80 | 73.55 | 4.53 |

**The adapter sweeps every metric — EM 36×, F1 2×, chrF +14.6, BERTScore +12.0.** No metric
dissents, so no tie-break is needed and `sqrt(EM × chrF)` is not load-bearing here. (EM is
*capped* on MCIF — only 19% of golds are 1-2 words — but capped is not dead: it still separates
21.82 from 0.61.)

**`qa-context` — ❌ tydiqa, the UNFAITHFUL proxy (monolingual, n=615). Do not route on this.**

| System | job | EM | F1 | chrF | BERT | √(EM·chrF) |
|---|---|---|---|---|---|---|
| 9B 3-shot | 3822329 | 8.13 | 33.18 | 38.94 | 70.67 | **17.79** |
| 9B + gold-LoRA, 0-shot | 3857589 | 15.61 | 23.90 | 19.53 | 63.01 | 17.46 |
| 9B 3-shot, no lang-hint | 3859645 | 5.37 | 26.83 | 34.39 | 67.91 | 13.58 |
| 9B + gold-LoRA + 3-shot | 3858987 | 2.76 | 9.62 | 14.46 | 56.20 | 6.32 |

**This column is where the whole chrF-vs-EM argument lived** — and note that even here the
hedge is a near-tie (17.79 vs 17.46), so it was never evidence for 3-shot either. The old
pooled numbers reconcile exactly: gold-LoRA's famous "EM 16.92" = (615×15.61 + 165×21.82)/780,
i.e. **the pooled EM was 79% a proxy for the wrong task.**

**`qa-oeg`** (2,359 test rows) keeps chrF / BERTScore (175-word golds; EM is ~0 for everything
and `sqrt(EM × chrF)` must not be used) and adds **word-budget compliance**, scored at test time
and invisible here.

|  | | **qa-oeg long-form** (~87%) | | **qa-oeg short-answer** (~13%) | | |
| System | job | chrF | BERT | chrF | BERT | (legacy overall) |
|---|---|---|---|---|---|---|
| 9B 3-shot | 3822329 | 25.55 | 69.38 | **24.19** | **67.30** | 27.64 |
| **9B + gold-LoRA, 0-shot** | 3857589 | 29.06 | 72.89 | 21.95 | 66.90 | 26.56 |
| 9B + gold-LoRA + 3-shot | 3858987 | **29.62** | **73.98** | 19.94 | 61.71 | 21.64 |
| 9B 3-shot, no lang-hint | 3859645 | 25.64 | 69.27 | 23.85 | 66.59 | 25.97 |
| 9B + distilled LoRA, 0-shot | 3864945 | _pending_ | | | | |

Proxies: `qa-context` = **MCIF only** (n=165; tydiqa is reported above but does not proxy the
test task). `qa-oeg long-form` = OEG (n=97). `qa-oeg short-answer` = aya (n=978). Never average
the two qa-oeg columns — they are opposite ends of one spectrum, and dev weights them backwards
(978 rows for ~13% of the task, 97 for ~87%).

### ⚠️⚠️ `qa-context`: the dev proxy is 79% the WRONG TASK — measured 2026-07-16

**Read this before the two sections below it; it undercuts both.** The user's note said "the
test set is really different from the dev set — inspect the qa-context entries." It is, and
here is what `data/tests.jsonl` actually contains (enumerated, not sampled — per the
qa-oeg lesson):

**1. `qa-context` is 100 unique items, not 8,640 questions.** Same shape as qa-oeg: a parallel
corpus. The id is `qa-context_{n}_{question_lang}_{context_lang}` (⚠️ **question lang comes
first** — the reverse reading makes `fra` look like an answer language). Each item is asked in
all **24 question languages**; what varies is how many languages its **passage** was
translated into:

| items | fan-out | rows | share |
|---|---|---|---|
| 5 (items 1–5) | 24 q-langs × **25** ctx-langs = 600 each | **3,000** | **35%** |
| 5 | 24 × 16 = 384 each | 1,920 | 22% |
| 10 | 24 × 6 = 144 each | 1,440 | 17% |
| 5 | 24 × 4 = 96 each | 480 | 6% |
| 75 | 24 × **1** = 24 each | 1,800 | 21% |

**Five items carry 35% of the sub-task.** Any per-row average over `qa-context` is really a
weighted vote over ~100 items with a 25:1 weight spread.

**2. 96% of `qa-context` is CROSS-LINGUAL** (8,300/8,640: passage in one language, question in
another). **3. There are 25 context languages but only 24 question languages** — `fra` appears
*only* as a passage language. So "fra/swh/tel/tha vanished from the test set" is true for
**answer** languages (the `question_lang` field has 0 fra/swh/tel/tha rows) but **false for
passages**: French passages are in the test set; we just never answer in French.

**4. …and the dev proxy is mostly the wrong task.** Inspecting the actual sample rows:

| dev source | n (dev) | shape | faithful to the test task? |
|---|---|---|---|
| `copenlu/answerable_tydiqa` | 615 (79%) | Arabic passage + Arabic question + Arabic answer — **monolingual** | ❌ ~4% of the test sub-task |
| `FBK-MT/MCIF` | 165 (21%) | German question + **English** content + German answer — **cross-lingual** | ✅ the only faithful one |

> **Consequence: the entire "chrF vs EM" fight below was fought on tydiqa** — the monolingual
> source, which stands in for ~4% of what the test set actually asks. The faithful proxy is
> MCIF, and it is 21% of the proxy pool and n=165. **On MCIF the adapter is not ambiguous at
> all: chrF 49.26 vs 3-shot's 34.61.** The dev weighting for `qa-context` is inverted in the
> same way it is for `qa-oeg` (978 aya rows for ~13% of the task) — that mistake now appears
> in *both* sub-tasks, and both times it was found by reading the data rather than the README.
>
> **What this does NOT settle:** `evaluate.py` computes EM/token-F1 for the `qa-context`
> *group* (`TASK_PROXY` = tydiqa + MCIF pooled), so **every EM/F1 number in the table below is
> 79% tydiqa** and none of them is per-source. The cheap fix is to split EM/F1 by source the
> way chrF already is — the prediction CSVs for all four systems still exist, so this is a
> re-score, not a re-run. Until then, treat the `qa-context` EM/F1 column as *measuring the
> wrong task*, not as evidence.
>
> **This is why the user's note says "we need a whole new train/dev set."** MCIF is the only
> cross-lingual QA source we have, at n=165 for an 8,640-row sub-task, and it is TED-talk
> transcripts with sentence-length answers — not the 2-word extraction that `evaluate.py`'s
> header assumes the golds are. (We have no test golds; that assumption came from tydiqa.)

### ✅ RESOLVED 2026-07-16: `qa-context` → adapter. The disagreement was a proxy artifact.

**Resolution: the metrics never actually disagreed — we were pooling two different tasks.** On
MCIF (the only cross-lingual proxy, matching 96% of the test sub-task) the adapter wins EM, F1,
chrF *and* BERTScore. The "chrF says 3-shot, EM says adapter" deadlock existed only in the
pooled column, which was 79% monolingual tydiqa. **`qa-context` (8,640 rows) → gold/distilled
adapter, 0-shot.** It did not take the organisers' metric to decide, and it did not take the
`sqrt(EM × chrF)` hedge either — just the right proxy.

The history below is kept because the reasoning was wrong in an instructive way.

Every earlier version of the plan routed `qa-context` to plain 3-shot, because the adapter
"collapsed" on tydiqa (chrF 38.94 → 19.53). **That collapse is at least partly a chrF
artifact.** On the metric the task is normally scored with, the adapter is **2.6× better at
returning the gold span** (16.92 vs 6.54 EM) while token F1 is a near-tie (31.09 vs 32.12).
⚠️ Both those numbers are the superseded *pooled* ones — see the split tables above.

The two numbers describe different failure shapes, and both are real:

- **gold-LoRA** answers tersely — which is what gold targets teach and what extraction wants —
  so it hits the span exactly far more often, but when it misses it misses completely (chrF≈0).
- **3-shot** answers verbosely, wrapping the right span in a sentence. Almost never an exact
  match, almost always in the neighbourhood — which is precisely what chrF rewards and EM does not.

**Which one wins depends entirely on the organisers' automatic metric, and we do not know what
it is.** It flips the routing for 8,640 of our 10,999 test rows, and no further experiment of
ours can resolve it.

> **Decision 2026-07-16 (user's call): we are NOT emailing the organisers about the metric,
> and we hedge with the geometric mean `sqrt(EM × chrF)` as our own selection rule.**
> The open item is closed as *decided under uncertainty*, not as answered — the organisers'
> metric remains unknown, and this is our tie-break, not a discovery about theirs.
> (The double-escaping and the 8 `{country}`/`{language}` placeholder rows were the other two
> items in that draft email; the 100 empty English prompts came off the list on 2026-07-16 when
> the organisers fixed them unprompted — TEST_SET_ANALYSIS §6.)
>
> Applied to the pooled `qa-context` proxy, the rule ranks: **gold-LoRA 19.86** (√(16.92×23.31))
> > 3-shot 15.66 (√(6.54×37.52)) > 3-shot no-hint 12.20 > adapter+3-shot 8.71 — i.e. **it picks
> the adapter.**
>
> ⚠️ **But do not record that as the routing decision yet, for two independent reasons:**
> 1. **The inputs are from the wrong task.** Those EM values are pooled tydiqa+MCIF, i.e. 79%
>    monolingual (see the section above). The rule is sound; the pool it was fed is not.
>    Re-score EM/F1 per source and re-apply it to MCIF before acting.
> 2. **The rule only means something where EM does.** `sqrt(EM × chrF)` is a `qa-context` rule
>    only. On `qa-oeg` (175-word compositions) EM is ~0 for every system, and a geometric mean
>    with a near-zero factor is ~0 regardless of chrF — it would rank noise. Keep qa-oeg on
>    chrF/BERTScore.
>
> Property worth knowing: the geometric mean hands the decision to **EM**, because EM varies
> more across our systems in *relative* terms (3.9× spread, 4.36→16.92) than chrF does (2.3×,
> 16.00→37.52), and a geometric mean is a mean of logs. That is a defensible hedge — it refuses
> to reward a system that never lands the span — but it is a choice, not a neutral compromise.

> **Update 2026-07-16** — *this note is now answered; see "the dev proxy is 79% the WRONG TASK"
> above.* The original note read: "This part is also partially not true. The test set is really
> different from the dev set. Please inspect the qa-context entries of test set. You'll have
> interesting finds. We need an whole new train/dev set." Inspected: `qa-context` is 100
> parallel items (5 of them = 35% of rows), 96% cross-lingual, and the dev proxy is 79%
> monolingual tydiqa. The note was right — hence everything above it is proxy-limited.

### `qa-oeg` is split too: the adapter wins the long end, 3-shot wins the short end

The two halves of `qa-oeg` disagree. **Long-form**: adapter 29.06 vs 3-shot 25.55. **Short-
answer**: 3-shot 24.19 vs adapter 21.95. That is coherent rather than contradictory — gold-SFT
taught terseness, which helps extraction (see its 16.92 EM) and helps nothing on a 175-word
composition, while few-shot demos teach a chatty register that suits short trivia. Weighting by
prompt share (~87/13) the adapter still takes `qa-oeg`, but this is now a split decision on
n=97 vs n=978 with dev's weighting inverted, not the clean win the plan recorded.

### Budget compliance: the model ignores the budget and writes its own default length

⚠️ **Corrected 2026-07-16.** This section previously read "every system writes ~half of what it
is asked for" — that was measured on OEG rows only, i.e. the long end. With aya's budgeted rows
included the picture inverts, and the same error (a subset generalised to the whole) produced it:

| | budgeted rows | mean deviation, 3-shot | adapter 0-shot | adapter+3-shot | no-hint |
|---|---|---|---|---|---|
| qa-oeg long-form (OEG) | 4 | **−53%** | −56% | −55% | −53% |
| qa-oeg short-answer (aya) | 7 | **+502%** | +159% | +418% | +447% |

Asked for a long answer it writes half; asked for a short one it writes five times too much.
**All four systems land in band on ~0 of 11 budgeted dev rows.** The failure is not
under-writing or over-writing — it is **regression to a default length regardless of the
instruction**, which is a stronger and more actionable statement of roadmap C's premise than
either one-sided version.

n=11 total (dev has almost no budgeted rows — TEST_SET_ANALYSIS §4), so treat the magnitudes as
indicative only. The direction reverses cleanly across two independent slices and four
independent systems, which is what makes it worth believing at all. Re-measure on real test
outputs, where ~20% of qa-oeg rows carry a budget.

(Curiosity, n=1: adapter+3-shot overshoots a budgeted `qa-context` row by **+1133%** — another
face of the demos-confuse-the-adapter effect that sinks its EM to 4.74.)

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
| 3859277-79 | 2026-07-15 | teacher generation, whole corpus (3 shards) | Qwen3.5-35B-A3B bf16, `teacher_gen.sbatch --shard {1,2,3}/3`, lang-hint ON | 11,915 | ✅ all three finished 2026-07-16 (15:54–16:48 each, within the 24h budget); 3,971 + 3,972 + 3,972 rows written to `runs/teacher-s{1,2,3}of3.jsonl`. The trailing `_thread.RLock` AttributeError in each log is `multiprocess`'s ResourceTracker teardown noise, after the rows are flushed — not a failure. **⚠️ Two things learned after these were submitted, neither worth killing them for — see the note below.** |
| 3859682 | 2026-07-15 | teacher generation, aya+oeg subset | Qwen3.5-122B-A10B-GPTQ-Int4 via vLLM, `teacher_gen_vllm.sbatch --source aya,oeg`, 2× a100_80 | 4,126 | ✅ 17m10s, all rows written, no failures → `runs/teacher122b-aya-oeg.jsonl` (gitignored). vLLM's batched-throughput edge (~250×/row vs the 35B transformers loop) holds at scale. |
| 3860144 | 2026-07-15 | filter calibration report on the 122B output | `filter_teacher.sbatch runs/teacher122b-aya-oeg.jsonl --report`, a40 | 4,126 | ✅ 1m16s. See distributions below. |
| 3864927 | 2026-07-16 | filter calibration report, **both teachers merged** | `filter_teacher.sbatch runs/teacher122b-aya-oeg.jsonl runs/teacher-s{1,2,3}of3.jsonl --prefer 122b --report`, a40 | 11,915 | ✅ `resolved 4126 overlapping qa_idx in favour of '122b'` — the expected count, and aya/oeg reproduce 3860144's distributions exactly (oeg p50 = 34.5/72.2), confirming those two sources really are the 122B's answers. Merged shape: 122B aya 3,763 + oeg 363; 35B belebele 4,577 + tydiqa 2,497 + MCIF 715. |
| 3864941 | 2026-07-16 | **filter → `data/sft-distilled.jsonl`** | same inputs, `--prefer 122b --chrf-min 30 --bertscore-min 70 --gold-only belebele` | 11,915 | ✅ wrote 11,915 rows (**3,048 teacher / 8,867 gold**), `--mix replace` default → same rows as the gold-SFT run 3822375, so training targets stay the one intended difference. Pass rates below. |

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

**Final filter policy (decided 2026-07-16 on 3864927's merged report): `--chrf-min 30
--bertscore-min 70 --gold-only belebele`, one threshold everywhere else.** Measured per-source
pass rates at that policy (job 3864941, printed on every write run — the `--report` mode prints
distributions and the global C\B grid instead):

| source | policy | pass | n |
|---|---|---|---|
| `wmt25-mist-oeg-gpt-4.1` | 30/70 | **87.6%** | 363 |
| `FBK-MT/MCIF` | 30/70 | 64.1% | 715 |
| `CohereLabs/aya_dataset` | 30/70 | 40.2% | 3,763 |
| `copenlu/answerable_tydiqa` | 30/70 | 30.5% | 2,497 |
| `facebook/belebele` | **GOLD-ONLY** | 0.0% | 4,577 |

A deliberately looser OEG threshold (`--source-min oeg=20,65`) was considered and **not**
taken: OEG already passes 87.6%, so it would buy ~40 more rows at the cost of a second
policy to reason about. ⚠️ **qa-oeg is still the thinnest link and distillation did not
change that** — 87.6% of 363 is ~318 teacher rows backing 2,359 test rows. The 30/70
per-source rates in job 3861614 are superseded for aya/oeg (those were the 35B's answers;
`--prefer 122b` replaces them).

Earlier calibration (from 3860144, the 122B alone — kept because the aya/oeg distributions
below are the ones that survived into the merge):

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
| 3864945 | 2026-07-16 | distilled-adapter **SFT** | Qwen3.5-9B, `lora_sft.sbatch --data data/sft-distilled.jsonl --no-lang-hint` | 11,915 train rows | _running_ | | | Log confirms `format=test (no lang-hint)`, 20 rows truncated at 2,048 tok. ~6.5h expected (cf. 3822375). |
| — | | distilled-adapter dev eval | Qwen3.5-9B + LoRA from 3864945, shots=0, **`--no-lang-hint`** | 2978 | _pending_ | | | Eval must use `--no-lang-hint` too — it has to match how 3864945 was trained. |

> **⚠️ This row is NOT the one-variable A/B it was originally planned as.** It used to read
> "same recipe as 3822375/3857589, one variable (training targets)". That is no longer true:
> 3864945 also changes the *training format* (`--no-lang-hint`, added 2026-07-16 in commit
> `6fd2a2e`), so it differs from 3822375 in **two** ways — targets *and* format. This was a
> deliberate trade (user's call, 2026-07-16): the test format is what `run_test.py` actually
> feeds, and ROADMAP row E's reading of 3858987 is that train/infer format agreement matters
> more than a clean ablation. **Consequence: if this beats 3857589, we will not know which
> change earned it.**
>
> **The missing baseline is `9B + gold-LoRA, 0-shot, --no-lang-hint`** — an eval-only run
> (~6.9h, no retraining; the 3822375 adapter still exists at
> `/home/atuin/b279bb/b279bb31/MIST-26/adapters/qwen3.5-9b-qa-lora-3822375`, in the **$WORK**
> clone, not the $HOME one jobs now run from). It would restore the one-variable comparison
> *and* answer a question the whole routing table currently rests on: **every gold-LoRA number
> in the decision table above was measured with the lang-hint ON, but `run_test.py` feeds no
> hint.** Deploying that adapter as-is puts it in an unmeasured train/infer gap — the same
> shape of mismatch that cost 3858987 five chrF. 3859645 does *not* cover this: it showed
> dropping the hint is near-free for the **base** model at 3-shot, which is not an adapter
> trained with the hint. Deferred, not resolved.

Scope notes: the `sum` sub-task is handled by a teammate, this repo's experiments stay on `qa`
(incl. the OEG rows folded into it). The official test set is out (as of 2026-07-15), so once a
recipe wins on dev, retrain it on 100% of the sample data and run the test set with it.
