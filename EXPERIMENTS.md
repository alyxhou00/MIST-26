# Experiment log

> 🔒 **CLOSED 2026-07-17.** This log's dev split leaks parallel items across dev/train
> (DATA_AUDIT.md §2) — every adapter-vs-prompting margin here is optimistic. New runs go to
> [EXPERIMENTS_NEW.md](EXPERIMENTS_NEW.md) on the v2 item-split dataset; which old verdicts
> survive (and which don't) is listed at the top of that file. Do not add rows here.
>
> ⚠️ **Retro-diagnosis 2026-07-18** (details in EXPERIMENTS_NEW.md): the gold adapter's
> famous "tydiqa collapse" (38.94→19.53, job 3857589) was a **runaway-generation artifact**
> — 78% of its tydiqa and 56% of its aya predictions continue past the answer into
> hallucinated chat turns — not a capability loss. Read every adapter-row tydiqa/aya score
> here with that in mind.

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
> | `FBK-MT/MCIF` | 165 | ✅ **`qa-context` — the only faithful proxy** (cross-lingual, like 96% of the test sub-task) |
> | `wmt25-mist-oeg-gpt-4.1` | 97 | ✅ **`qa-oeg`, long-form end** (golds 175 words p50) — ~87% of qa-oeg prompts |
> | `CohereLabs/aya_dataset` | 978 | ✅ **`qa-oeg`, short-answer end** (golds 24 words p50) — ~13% of qa-oeg prompts |
> | `copenlu/answerable_tydiqa` | 615 | ❌ **nothing** — monolingual; ~4% of a sub-task that is 96% cross-lingual (**retracted 2026-07-16**; it used to read "✅ qa-context — score it with EM/F1") |
>
> **Rules:** judge `qa-context` on **MCIF alone**, judge `qa-oeg` on **OEG and aya as two
> separate columns** (they measure opposite ends of one spectrum — never average them), and
> treat **belebele and tydiqa** as unscored. Score everything on **`COMBINED` = mean(chrF,
> BERTScore, ROUGE-L)**. ⚠️ dev's weighting is inverted against the test mix: aya gets 978 rows
> for ~13% of qa-oeg while OEG gets 97 for ~87% — and `qa-context`, 8,640 test rows, rests on
> 165. **Usable dev = 1,240/2,978 rows (42%).**
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

**Rescored per source 2026-07-16 (jobs 3865022-25), and it settles the routing.** `qa-context`
used to be reported as one pooled column (tydiqa + MCIF, n=780). That pooling was 79% tydiqa —
the *monolingual* source, worth ~4% of a test sub-task that is 96% cross-lingual. Split apart,
the metric disagreement that blocked this decision for days **evaporates**.

**Selection score = `COMBINED` = mean(chrF, BERTScore, ROUGE-L)**, one rule for every sub-task
(user's call, 2026-07-16; jobs 3865022-25). It replaces `sqrt(EM × chrF)`, which went blind
exactly where it mattered — EM only resolves where golds are short enough to hit exactly, i.e.
on tydiqa (63% of golds are 1-2 words), the proxy that does *not* resemble the test set, and
not on MCIF (19%) or qa-oeg (EM ~0). **The switch preserved every routing decision below**, so
it changed the rule, not the conclusions. ⚠️ It is a compromise, not a neutral one — see
`evaluate.py:combined()`; the components are listed here precisely so a close call can be
audited rather than taken on the mean's word.

> ⚠️ **Leakage asterisk on every adapter-vs-prompting margin below (found 2026-07-17,
> DATA_AUDIT.md §2).** The 80/20 split is by *row*, but MCIF/OEG/belebele carry the same item
> in several languages and aya has verbatim duplicates — so the adapters trained on
> cross-lingual twins of dev items (all 21 MCIF dev talks have train twins). Adapter-vs-adapter
> rows (gold vs distilled) and prompting-vs-prompting rows (shots, lang-hint) are unaffected;
> adapter-vs-prompting margins are optimistic. Directions below likely survive (the MCIF sweep
> is 4-metrics-unanimous); honest margins await the item-split rebuild.

**`qa-context` — ✅ MCIF, the FAITHFUL proxy (cross-lingual, n=165). Route on this column.**

| System | job | **COMBINED** | chrF | BERT | ROUGE-L | *(diag: EM / F1)* |
|---|---|---|---|---|---|---|
| **9B + gold-LoRA, 0-shot, no lang-hint** | 3866054 | **63.18** | **50.08** | 86.22 | **53.24** | *20.00 / 57.34* |
| **9B + gold-LoRA, 0-shot** | 3857589 | 62.55 | 49.26 | **86.41** | 51.98 | *21.82 / 57.92* |
| 9B + distilled LoRA, 0-shot, no lang-hint | 3865036 | 51.26 | 39.23 | 78.53 | 36.02 | *13.33 / 39.39* |
| 9B 3-shot | 3822329 | 45.73 | 34.61 | 74.38 | 28.19 | *0.61 / 28.15* |
| 9B 3-shot, no lang-hint | 3859645 | 44.66 | 33.80 | 73.55 | 26.63 | *0.61 / 27.16* |
| 9B + gold-LoRA + 3-shot | 3858987 | 40.16 | 20.98 | 69.89 | 29.61 | *12.12 / 29.70* |

**The adapter sweeps every component — chrF +14.6, BERTScore +12.0, ROUGE-L +23.8, and EM 36×.**
Nothing dissents, so the choice of rule is irrelevant here: any of them picks the adapter — and
the gold adapter, not the distilled one, which lands 11.29 COMBINED short of it (see below).

**The two gold-LoRA rows are the same adapter with and without the lang-hint** (3866054 vs
3857589) and they land within 0.6 COMBINED of each other — the hint is ~free on the adapter, not
just on the 3-shot base. That settles the confound below: the distilled adapter's loss is the
teacher data, not the dropped hint.

**`qa-context` — ❌ tydiqa, the UNFAITHFUL proxy (monolingual, n=615). Do not route on this.**

| System | job | COMBINED | chrF | BERT | ROUGE-L | *(diag: EM / F1)* |
|---|---|---|---|---|---|---|
| 9B + distilled LoRA, 0-shot, no lang-hint | 3865036 | **64.61** | 54.55 | 87.33 | 51.96 | *53.01 / 69.76* |
| 9B 3-shot | 3822329 | 47.87 | 38.94 | 70.67 | 34.00 | *8.13 / 33.18* |
| 9B 3-shot, no lang-hint | 3859645 | 44.02 | 34.39 | 67.91 | 29.76 | *5.37 / 26.83* |
| 9B + gold-LoRA, 0-shot, no lang-hint | 3866054 | 34.45 | 20.09 | 64.59 | 18.66 | *16.42 / 26.59* |
| 9B + gold-LoRA, 0-shot | 3857589 | 33.36 | 19.53 | 63.01 | 17.53 | *15.61 / 23.90* |
| 9B + gold-LoRA + 3-shot | 3858987 | 26.73 | 14.46 | 56.20 | 9.52 | *2.76 / 9.62* |

**The distilled adapter tops this column by 16.7 and is mid-pack on the faithful one** — a clean
demonstration of why this table is quarantined. Ranking on tydiqa would pick the system that
`qa-context` (8,640 test rows, 96% cross-lingual) says is 11.29 COMBINED *worse*.

**This column is where the whole chrF-vs-EM argument lived**, and it inverts the faithful one —
which is the point: it is a different task. The old pooled numbers reconcile exactly: gold-LoRA's
famous "EM 16.92" = (615×15.61 + 165×21.82)/780, i.e. **the pooled EM was 79% the wrong task.**

**`qa-oeg`** (2,359 test rows) also adds **word-budget compliance**, scored at test time and
invisible here. EM is ~0 for every system on 175-word golds and is not reported.

|  | | **qa-oeg long-form** (~87%) | | | | **qa-oeg short-answer** (~13%) | |
| System | job | **COMBINED** | chrF | BERT | ROUGE-L | **COMBINED** | (legacy overall) |
|---|---|---|---|---|---|---|---|
| 9B + gold-LoRA + 3-shot | 3858987 | **48.46** | **29.62** | **73.98** | **41.77** | 30.55 | 21.64 |
| **9B + gold-LoRA, 0-shot, no lang-hint** | 3866054 | 46.60 | 28.40 | 73.35 | 38.06 | 34.73 | 25.09 |
| **9B + gold-LoRA, 0-shot** | 3857589 | 46.44 | 29.06 | 72.89 | 37.38 | 34.62 | 26.56 |
| 9B 3-shot, no lang-hint | 3859645 | 35.33 | 25.64 | 69.27 | 11.07 | 34.64 | 25.97 |
| 9B 3-shot | 3822329 | 35.30 | 25.55 | 69.38 | 10.96 | 35.30 | 27.64 |
| 9B + distilled LoRA, 0-shot, no lang-hint | 3865036 | 34.23 | 23.76 | 68.31 | 10.62 | **36.01** | 27.16 |

> ⚠️ **Open, surfaced by the new rule:** on the long-form end `adapter+3-shot` now scores
> *highest* (48.46 vs the adapter's 46.44) — it always led OEG on chrF, and ROUGE-L widens it.
> Routing by `task` is legal, so "qa-oeg → adapter+3-shot" is technically on the table, and even
> the 87/13 blend keeps it marginally ahead (46.13 vs 44.90). **Not adopted, and not on the
> strength of the overall 21.64** (that number is mostly belebele, which doesn't transfer): the
> real objection is that 3858987 is an adapter fed a format it never trained on (ROADMAP row E),
> so its OEG lead is a lucky OOD result on n=97, and the distilled adapter is trained 0-shot in
> test format, which would make demos OOD for *it* too. Revisit only with the distilled adapter's
> numbers in hand, and only if n=97 is judged enough to move 2,051 test rows.
>
> **Still open, and the distilled numbers did not settle it** (2026-07-17): the distilled adapter
> came in *last* on long-form (34.23), so the choice on qa-oeg is still gold-LoRA 0-shot vs
> gold-LoRA+3-shot on n=97. Nothing here moves it.

Proxies: `qa-context` = **MCIF only** (n=165; its **QA portion** — MCIF also has a 400-row
summarization portion under task=`sum`, the teammate's; and the same 21 talks sit behind both
portions and behind dev *and* train, see DATA_AUDIT.md §2 on leakage. tydiqa is reported above
but does not proxy the test task). `qa-oeg long-form` = OEG (n=97). `qa-oeg short-answer` = aya (n=978). Never average
the two qa-oeg columns — they are opposite ends of one spectrum, and dev weights them backwards
(978 rows for ~13% of the task, 97 for ~87%).

### ❌ 2026-07-17: distillation did not pay off — the comparison is now clean

**Teacher distillation lost to plain gold SFT on both columns that decide anything.** Adapter
trained by job 3864945 on `data/sft-distilled.jsonl` (11,915 rows, Qwen3.5-35B-A3B teacher,
30/70 filter + gold-only belebele), `--no-lang-hint`; scored 0-shot by job 3865036.

The comparison was originally confounded — distilled changed the data *and* dropped the lang-hint,
while gold-LoRA 3857589 was hinted. **Job 3866054 removed the confound**: gold-LoRA scored 0-shot
with `--no-lang-hint`, so both sides now differ only in training data. The hint costs the adapter
essentially nothing (see the matched column), so the full ~12-point loss is the teacher data.

| Sub-task (test rows it routes) | distilled 3865036 | gold-LoRA +hint 3857589 | gold-LoRA no-hint 3866054 | Δ (matched, vs 3866054) |
|---|---|---|---|---|
| `qa-context` — MCIF, faithful (8,640) | 51.26 | 62.55 | **63.18** | **−11.92** |
| `qa-oeg` long-form — OEG (~2,051) | 34.23 | 46.44 | **46.60** | **−12.37** |
| `qa-oeg` short-answer — aya (~308) | **36.01** | 34.62 | 34.73 | +1.28 |

The two gold columns are within 0.6 COMBINED everywhere — **dropping the lang-hint on the adapter
is ~free**, the same result the 3-shot base showed (3859645 vs 3822329), now confirmed on an
adapter for the first time. So the matched Δ against 3866054 (−11.92, −12.37, +1.28) is the honest
one and it is barely moved from the confounded −11.29/−12.21/+1.39: **the teacher data is the
cause, not the hint.** The distilled adapter beats plain 3-shot on `qa-context` (51.26 vs 45.73)
but is **last of all systems** on long-form; its one win, short-answer, is +0.71 over 3-shot on the
~13% end of qa-oeg. **No routing changes: gold-LoRA keeps `qa-context` and `qa-oeg`, and the whole
routing table is now known to be lang-hint-invariant — so its hinted gold-LoRA numbers are
directly comparable to `run_test.py`'s no-hint inference.**

**Checked and false: the length hypothesis.** The eval flags −57% word-budget undershoot on
long-form, which looks like the cause, but it is not distilled-specific — median prediction is
**82 words vs gold-LoRA's 76** (golds are 175). So the 3.5× ROUGE-L gap (10.62 vs 37.38) happens
at *equal length*: it is a content/language difference, not verbosity. Do not "fix" it by asking
for longer answers.

**Untested hypothesis, recorded so it is not mistaken for a finding.** The distilled mix is 38%
belebele + 21% tydiqa — 59% of training data on the two proxies this file quarantines as
non-predictive — against 6% MCIF + 3% OEG. The column it wins (aya) is 32% of the mix; the column
it loses worst (OEG) is 3%. That is the same "dev weights it backwards" error, possibly now baked
into the training data — but it is a correlation over five points, not a result.

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
| `copenlu/answerable_tydiqa` | 615 (79%) | same-language passage+question+answer, **11 languages** (not "Arabic" as this row once said — DATA_AUDIT.md §1) — **monolingual** | ❌ ~4% of the test sub-task |
| `FBK-MT/MCIF` (QA portion) | 165 (21%) | question in {deu, eng, ita, zho} + **English** talk content + answer in the question language — **cross-lingual except its eng→eng quarter** (44/165) | ✅ the only (mostly) faithful one |

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
MCIF (the only cross-lingual proxy, matching 96% of the test sub-task) the adapter wins chrF,
BERTScore, ROUGE-L, EM *and* F1. The "chrF says 3-shot, EM says adapter" deadlock existed only
in the pooled column, which was 79% monolingual tydiqa. **`qa-context` (8,640 rows) →
gold/distilled adapter, 0-shot.** It did not take the organisers' metric to decide, and it did
not take any tie-break rule either — just the right proxy.

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

> **Decision 2026-07-16 (user's call): we are NOT emailing the organisers about the metric.**
> The open item is closed as *decided under uncertainty*, not as answered — their metric remains
> unknown, and what follows is our own selection rule, not a discovery about theirs.
> (The double-escaping and the 8 `{country}`/`{language}` placeholder rows were the other two
> items in that draft email; the 100 empty English prompts came off the list on 2026-07-16 when
> the organisers fixed them unprompted — TEST_SET_ANALYSIS §6.)
>
> **The rule is `COMBINED` = mean(chrF, BERTScore, ROUGE-L), for every sub-task.** See the
> decision tables above.
>
> **Superseded the same day: `sqrt(EM × chrF)`.** Worth recording why, because the failure is
> reusable. The geometric mean hands the decision to **EM** (EM's relative spread across our
> systems is 3.9× against chrF's 2.3×, and a geometric mean is a mean of logs) — and EM is
> precisely the metric that only resolves on **tydiqa**, the proxy that does not resemble the
> test set. On the proxies that decide anything (MCIF, and all of qa-oeg) the golds are too long
> to hit exactly, so EM is floored and the rule is either near-blind or, on qa-oeg, ranking a
> ~0 factor. **A rule that works only where the measurement is wrong is not a rule.** The
> replacement is not neutral either — an unweighted mean of raw values weights each metric by
> its variance, so BERTScore (1.24× spread against chrF's 2.35×) sets the level and mostly
> abstains from the ordering, and chrF+ROUGE-L are two votes for the same thing (surface
> overlap) against one quiet vote for semantics. It is a defensible compromise with its thumb
> on surface overlap; `evaluate.py:combined()` says so in the code rather than in a doc.

> **Update 2026-07-16** — *this note is now answered; see "the dev proxy is 79% the WRONG TASK"
> above.* The original note read: "This part is also partially not true. The test set is really
> different from the dev set. Please inspect the qa-context entries of test set. You'll have
> interesting finds. We need an whole new train/dev set." Inspected: `qa-context` is 100
> parallel items (5 of them = 35% of rows), 96% cross-lingual, and the dev proxy is 79%
> monolingual tydiqa. The note was right — hence everything above it is proxy-limited.

### `qa-oeg` is split too: the adapter wins the long end, 3-shot wins the short end

The two halves of `qa-oeg` disagree, on COMBINED as on chrF. **Long-form**: adapter 46.44 vs
3-shot 35.30. **Short-answer**: 3-shot 35.30 vs adapter 34.62. That is coherent rather than
contradictory — gold-SFT taught terseness, which helps extraction (see its MCIF EM of 21.82
against 3-shot's 0.61) and helps nothing on a 175-word composition, while few-shot demos teach a
chatty register that suits short trivia. Weighting by prompt share (~87/13) the adapter still
takes `qa-oeg` (44.90 vs 35.30), but this is a split decision on n=97 vs n=978 with dev's
weighting inverted, not the clean win the plan recorded — and the short end is close enough
(0.68) to be noise at that n.

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

## LoRA SFT on distilled data — teacher outputs + gold mix (❌ ran 2026-07-17; lost to gold SFT)

Key finding from the 9B runs above: few-shot's gain is almost entirely *answer format*
(belebele chrF 17.69→52.70, tydiqa 21.88→38.94) while open-ended generation (aya) stays flat
(24.03→24.19) — prompting lacks a lever for answer *quality* there. The organizers allow
distillation as long as the final model is <10B, so the plan is sequence-level KD:
(1) generate teacher answers on the qa train split (11,915 rows; same seed-42 80/20 split,
dev untouched), (2) quality-filter against the golds ([scripts/filter_teacher.py](scripts/filter_teacher.py),
per-row sentence chrF OR BERTScore, thresholds calibrated per source via `--report`),
(3) LoRA SFT the 9B on the filtered teacher+gold mix (`train_lora.py --data`) as a *fresh*
adapter (not continued from 3822375).

⚠️ **The plan's "one variable (the data), directly comparable to the gold-only adapter above"
did not survive contact.** The run that was actually launched (3864945) also trains in the test
format (`--no-lang-hint`), so it differs from 3822375 in **two** ways. That was a deliberate
trade — see the note under the dev-eval table below for what it costs and what would buy the
comparison back.

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
| 3864945 | 2026-07-16 | distilled-adapter **SFT** | Qwen3.5-9B, `lora_sft.sbatch --data data/sft-distilled.jsonl --no-lang-hint` | 11,915 train rows | — | — | — | ✅ **Completed 2026-07-17 01:53, 6h10** (the ~6h15 projection held). `train_loss` 0.5706, 2 epochs, no divergence. Log confirms `format=test (no lang-hint)`, 20 rows truncated at 2,048 tok. Adapter → `$HOME/MIST-26/adapters/qwen3.5-9b-qa-lora-3864945` — **$HOME only; it does not exist in the atuin clone.** |
| 3865036 | 2026-07-17 | distilled-adapter **dev eval** | LoRA from 3864945, shots=0, **`--no-lang-hint`** | 2978 | 27.16 | 84.96 | 53.85 | ✅ **Completed 05:14, 3h21** — half the ~6h52 projected from 3857589. Chained on `--dependency=afterok:3864945`. **These overall numbers are the 71%-noise legacy ones; the verdict is the sub-task table above — MCIF −11.29 and OEG −12.21 vs gold-LoRA.** Submitted from the **atuin** clone against the **$HOME** adapter; only the absolute path made that correct (see the note below). |
| 3866054 | 2026-07-17 | **gold-LoRA no-hint A/B** (the missing baseline) | gold adapter 3822375, shots=0, **`--no-lang-hint`** | 2978 | 25.09 | 78.45 | 44.85 | ✅ **Completed 20:00, 5h29.** The one-variable A/B that de-confounds 3865036: matched to the gold-LoRA it now differs from *only* in lang-hint (3857589), it moves ≤0.6 COMBINED on every routing column (MCIF 63.18 vs 62.55, OEG 46.60 vs 46.44, aya 34.73 vs 34.62). **So the hint is ~free on the adapter, and the distilled loss is the teacher data — see the sub-task table above.** First run under the fixed `--lora` path: log banner prints `adapter: /home/hpc/.../adapters/qwen3.5-9b-qa-lora-3822375` (absolute). |

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
> (~6.9h, no retraining; the 3822375 adapter is at
> `adapters/qwen3.5-9b-qa-lora-3822375` in **both** clones as of 2026-07-16 — atuin has no
> backup and no snapshots, so its final weights were copied to $HOME). It would restore the
> one-variable comparison
> *and* answer a question the whole routing table currently rests on: **every gold-LoRA number
> in the decision table above was measured with the lang-hint ON, but `run_test.py` feeds no
> hint.** Deploying that adapter as-is puts it in an unmeasured train/infer gap — the same
> shape of mismatch that cost 3858987 five chrF. 3859645 does *not* cover this: it showed
> dropping the hint is near-free for the **base** model at 3-shot, which is not an adapter
> trained with the hint. Deferred, not resolved.
>
> **2026-07-17 — RESOLVED by job 3866054.** The eval-only run above (3822375 adapter,
> `--no-lang-hint`, ran 5h29) is now in the tables. It confirms the hint is ~free on the adapter
> (≤0.6 COMBINED vs hinted 3857589 on every routing column), so 3865036's ~12-point loss is the
> teacher data, not the dropped hint — and the whole routing table's hinted gold-LoRA numbers are
> now known to be directly comparable to `run_test.py`'s no-hint inference. No baseline outstanding.

> **Sharp edge, fixed 2026-07-17 (post-mortem: IMPLEMENTATION_NOTES).** 3865036 was submitted
> from the atuin clone against an adapter that exists only in `$HOME`. It was correct **by luck**:
> the submit line happened to use an absolute path, and `lora_sft.sbatch`'s own copy-paste hint
> printed a *relative* one, which would have resolved inside the atuin clone and hit either
> nothing or the stale 3822375. `benchmark.py` also never logged which adapter it loaded, so the
> only record was `sacct --format=SubmitLine`. Now: `benchmark.py` resolves `--lora` to an
> absolute path, rejects a dir with no `adapter_config.json` before the dataset or base model
> loads, and prints `model: … adapter: <abs>`; `lora_sft.sbatch` and `train_lora.py` emit
> absolute paths in their hints.

Scope notes: the `sum` sub-task is handled by a teammate, this repo's experiments stay on `qa`
(incl. the OEG rows folded into it). The official test set is out (as of 2026-07-15), so once a
recipe wins on dev, retrain it on 100% of the sample data and run the test set with it.
