# WMT26 MIST Implementation Notes

> [!NOTE]
> Kept for future reference for the writing. Written by Claude.

## Problems Identified

### 1. Zero-Shot Output Quality Issues (from error_analysis.py)

**Script Mismatch (6–100% of examples per language)**
- Gold expected in non-Latin script (e.g., Hindi, Thai) but prediction came back mostly Latin/ASCII
- Worst languages: Hindi (6.2%), Chinese (0.6%)
- Example: 
  - Gold: `'कौनसे कांटीनेंट में शहर नहीं हैं? अंटार्टिका'` (Hindi)
  - Pred: `'Here is an example of a geography trivia question, complete with an answer...'` (English)
- Root cause: `--lang-hint` tells model to respond in Hindi, but zero-shot doesn't show examples of concise Hindi answers

**Length Mismatch (60–100% of examples per language)**
- Predictions much longer than gold (> 3× gold length + 20 chars)
- Worst languages: Thai (100%), Kurdish (100%), Persian (100%), Telugu (100%), Swahili (98.2%)
- Example:
  - Gold: `'ตในสาขาการบริหารการพัฒนาในปี ค.ศ. 2006 จาก'` (~32 chars, extractive answer)
  - Pred: `'Based on the text provided, **Rodolfo Marapon de la Rosa**...'` (200+ chars, essay-like with markdown)
- Root cause: Model is a chat assistant, trained to explain and elaborate; gold answers are terse extractive style that can't be taught via instruction alone

**Task Format/Conventions Not Understood**
- Example: MCIF dataset has "unable to answer" convention (`'无法回答。'`), but model generates plausible-sounding wrong answers
- Example: Belebele uses multiple-choice answer text, model doesn't know to match exactly
- Root cause: Format conventions are invisible in zero-shot; only apparent when model sees multiple examples

**Language Confusion (Cross-Lingual Instability)**
- Aya dataset: English questions, expected non-English answers, but zero-shot sometimes answers in English anyway
- `--lang-hint` helps but inconsistent at scale
- Root cause: Single instruction can be overridden by prompt context; in-context examples are more robust

### 2. Data Leakage in Few-Shot (discovered during implementation)

**Duplicate Inputs Across Train/Dev Split**
- 209 duplicate input rows within qa overall
- **53 dev examples (1.8%, all from aya) have input that appears verbatim in train 80%**
- 4 of these have identical gold as well
- Example: `'Dame un ejemplo de trivia en esta categoría: Películas.'` appears in both splits
- Impact: Sampling a verbatim duplicate as a demonstration hands the model its own answer
- Fix: Filter excludes rows where `input == dev_row_input` at every selection tier

### 3. Model Architecture Quirk

**Qwen3.5-2B Prone to Thinking Loops**
- Model card recommends `enable_thinking=False` for non-thinking tasks
- Benchmark.py always uses `enable_thinking=False` to avoid this
- Shots don't include `<think>` blocks, which is correct (model should imitate: direct answer without reasoning)

## Tweaks & Design Decisions

### Few-Shot Implementation Details

**1. Three-Tier Shot Selection with Fallback**
```
Tier 1: same (source, lang_code) — ideal, demonstrates both task format AND target language
Tier 2: same source only       — fallback when tier 1 < k rows
Tier 3: entire train pool       — global fallback (rare, for unseen source/lang at test time)
```
- Rationale: Belebele (multiple choice) and tydiqa (extractive) have different answer formats; showing examples from the same source makes format convention clearer
- Data check: 100% of dev examples hit tier 1 (current split is well-balanced)

**2. Deterministic Per-Example Seeding**
```python
random_state = zlib.crc32(f"{seed}:{input_text}".encode("utf-8"))
```
- Why not global RNG? Because filters (`--limit`, `--source`, `--lang`) would change which shots each example gets → A/B runs would be incomparable
- Each dev example always gets the same shots regardless of row order or filtering
- Reproducible in isolation: `benchmark.py --shots 3 < single_dev_row>` always gives same shots

**3. Verbatim Input Deduplication**
- Executed at each tier before sampling
- Logic: `pool = pool[pool["input"] != dev_input]`
- If tier becomes too small, falls through to next tier
- Doesn't remove rows with similar-but-not-identical input (e.g., near-duplicate aya prompts): false positive rate would be high

**4. Chat Template Integration**
- Shots inserted as completed user/assistant turns in the messages list, not as text concatenation
- Example structure:
  ```
  system: "You are a helpful assistant. Respond in Hindi."
  user: "Demo question 1?"
  assistant: "Demo answer 1."
  user: "Demo question 2?"
  assistant: "Demo answer 2."
  user: "Demo question 3?"
  assistant: "Demo answer 3."
  user: "The actual question"
  assistant: <generation starts here>
  ```
- Rationale: Chat models train on turn structure; the model sees what imitation looks like by observing real assistant behavior, not by parsing natural-language instructions

### Sbatch File Organization Refactor

**Problem**: Before, sbatch files had generic names (`job.sbatch`, `smoke.sbatch`) while the naming convention should be one-sbatch-per-experiment.

**Solution**: 
- `job.sbatch` (job name `mist-qa`) → `0shot.sbatch` (job name `mist-qa-0shot`)
- `smoke.sbatch` (job name `mist-qa-smoke`) → `smoke-langhint.sbatch` (job name `mist-qa-smoke-langhint`)
- New: `fewshot.sbatch` (job name `mist-qa-fewshot`)
- New: `smoke-fewshot.sbatch` (job name `mist-qa-smoke-fewshot`)

**Retroactive Log Renaming**
- Old committed log `logs/mist-qa-3786727.out` (from original `job.sbatch` run) renamed to `logs/mist-qa-0shot-3786727.out`
- Verification: checked that log's dev examples count (2978) matches predictions-3786727.csv and README documentation
- Job ID (3786727) preserved in filename, so git history remains traceable

**Benefit**: Future logs are immediately identifiable by their job name; no guessing which experiment a given log came from.

### Time Limit Decisions

| Experiment | Time Limit | Reason |
|------------|-----------|--------|
| `0shot.sbatch` | 12h | Baseline |
| `smoke-langhint.sbatch` | 40 min | Small subset (aya/hin_Deva) |
| `fewshot.sbatch` | 18h | Each prompt includes N demonstration passages (longer prefill); belebele passages are longest |
| `smoke-fewshot.sbatch` | 1h | Same subset as smoke-langhint (comparable size), but few-shot prefill is longer |

## Verification & Testing

### Local Validation (No GPU/Dataset Required)

1. **Synthetic Data Test**: `test_fewshot.py` verifies:
   - Stratum matching works (source+lang precedence)
   - Determinism per dev example
   - Both fallback tiers activate correctly
   - Chat message structure correct (system + (user/assistant)×k + user)
   - Leakage guard filters verbatim duplicates
   - Guard triggers fallback when needed

2. **Qwen Chat Template Rendering**: Verified that `messages` list gets correctly rendered into ChatML format with `<|im_start|>role ... <|im_end|>` tokens

### Real Data Checks

1. **Error Analysis on Zero-Shot Baseline** (`predictions-3786727.csv`):
   - Confirmed 80–100% length mismatch on most languages
   - Identified script mismatch as secondary problem (6–100% depending on language)
   - Found that aya dataset's cross-lingual examples are most broken (both script and length issues)

2. **Data Leakage Check**:
   - Scanned full qa split (14,893 rows) for duplicates
   - Found 53 verbatim input duplicates across train/dev
   - 4 of those also have identical gold
   - Implemented filter accordingly

## Expected Improvements (from Few-Shot)

### What Few-Shot Should Fix (Mechanism: In-Context Learning)

1. **Length mismatch**: Seeing three train examples with short answers → model learns through imitation, not instruction
2. **Language consistency**: Seeing three examples in target language → model's probability distribution shifts toward that language
3. **Task conventions**: Seeing three examples in same source → model learns that source's format (multiple choice, extractive, etc.)

### What Few-Shot Cannot Fix

1. **Content knowledge**: Model doesn't learn new facts from examples, only style
2. **Aya open-ended quality**: Improvement is partly "style alignment" (shorter, target language) and partly "looking closer to reference", not necessarily "better answers"

### Why A/B First (smoke tests)

- `smoke-langhint.sbatch`: Confirms lang-hint's effect (already known to work)
- `smoke-fewshot.sbatch`: Quick 1h check that few-shot is working as expected before committing to 18h full run

## Code Commits

| Commit | Message |
|--------|---------|
| `5f1a5b2` | `feat: add few-shot prompting via --shots, with smoke + full sbatch jobs` |
| `76c1e53` | `refactor: rename job/smoke sbatch to per-experiment names` |
| `e6e0dae` | `refactor: rename zeroshot experiment slug to 0shot, backfill old log name` |
| `840c16b` | `fix: exclude verbatim-duplicate inputs from few-shot demonstrations` |

## Next Steps for Paper

### Measurements to Report

1. **Baseline (zero-shot)**
   - Already run: 2978 dev examples, chrF=18.01, BERTScore=62.21, ROUGE-L=12.51
   - Per-language breakdown available in error_analysis output

2. **Few-Shot (planned)**
   - `--shots 1, 3, 5`: Test different demonstration counts
   - Per-language breakdown to track which sources benefit most
   - Compare to baseline via chrF, BERTScore, ROUGE-L

3. **Ablations (if time permits)**
   - `--shots 3 --no-lang-hint`: Isolate few-shot effect vs. lang-hint
   - Different sources and languages (e.g., belebele vs. aya vs. tydiqa)

### Potential Issues to Document

- **Aya data quality**: 53 verbatim duplicates is unusual; mention in dataset description
- **Evaluation metric alignment**: chrF may underweight fluency improvements; BERTScore may reward style over content
- **Time scaling**: 18h runs are expensive; future work could explore smaller demonstration sets
- **Language coverage**: Current test data only covers languages in sample; July test set has "surprise" languages not seen during development

### Reproducibility Notes

- All code is in `scripts/` and `slurm/`; logs are committed
- Venv + cache is one-time setup (`slurm/setup.sh`)
- Per-experiment sbatch names mean job history is immediately identifiable
- Deterministic shot selection (CRC32 of input text) means any run can be reproduced exactly

---

# System Architecture as of 2026-07-15 (paper notes)

This section is the running architecture record for the qa subtask (context QA + OEG; the
`sum` subtask is a teammate's). Everything below is backed by a SLURM job ID in
EXPERIMENTS.md and a committed log in `logs/`.

## 1. Architecture overview

One **Qwen3.5-9B** base (9,438,911,728 params total, vision tower included — the exact
number PEFT reports), adapted per task type at inference time ("routing"). The official
test set gives `task` ∈ {qa-context, qa-oeg, sum-sum} per row, so routing on it is
explicitly legal:

Serving config (final, 2026-07-16/17 — supersedes the original 3-shot/distilled plan):

| Test task | Serving config | Why |
|---|---|---|
| `qa-context` | base + **gold-LoRA adapter 3822375, 0-shot** | resolved 2026-07-16: on MCIF (the only faithful proxy) the adapter sweeps all four metrics vs 3-shot (chrF 49.26 vs 34.61, EM 21.82 vs 0.61). The old 3-shot plan was based on belebele, which the test set does not contain. |
| `qa-oeg` | base + **gold-LoRA adapter 3822375, 0-shot** | gold-SFT banks the OEG gain (29.06 vs 3-shot 25.55). The distilled adapter was meant to take this route but **lost the A/B** (§5.4): last on OEG. |
| `sum-sum` | teammate's system (same base if joint submission) | not ours |

Both qa routes are the **same** gold adapter at 0-shot with no lang-hint — one adapter, and its dev numbers are directly comparable to `run_test.py`'s no-hint inference (confirmed lang-hint-invariant, job 3866054). Do **not** stack the adapter with few-shot demos (§5.4 / EXPERIMENTS: they fight, 21.64).

Inference-time additions planned: fastText LID gate (detect wrong-language output →
resample), optionally best-of-N with self-judging for the aggressive submission variant.

## 2. The 10B accounting

The organizers cap **total parameters of all deployed components** at 10B (MoE counts
total, not active). Our accounting:

| Component | Params | Note |
|---|---|---|
| Qwen3.5-9B base | 9,438,911,728 | shared by all routes; single copy |
| LoRA adapter, r=16 (per adapter) | 29,097,984 | measured (job 3822375 PEFT printout); ~0.31% of base |
| fastText LID (lid.176) | <1M | compressed model is <1MB on disk |
| **Total with 3 adapters** | **≈9.53B** | **fits, ~0.47B headroom** |

Few-shot prompting adds zero parameters. Teachers (35B-A3B, 122B-A10B) do NOT count —
they are never deployed, only their outputs are (as training data). The one real
constraint: a **joint submission with the sum teammate must share the same 9B base** —
two different ~9B bases would be ~19B and blow the cap; N task-specific LoRA adapters on
one shared base are nearly free (0.03B each).

## 3. Measured baselines (dev = held-out 20% of the sample data, n=2978)

chrF / BERTScore / ROUGE-L:

| Config | Overall | Job |
|---|---|---|
| 2B 0-shot | 18.01 / 62.21 / 12.51 | 3786727 |
| 2B 3-shot | 21.84 / 71.89 / 25.67 | 3817971 |
| 9B 0-shot | 23.12 / 66.04 / 22.75 | 3822324 |
| 9B 3-shot | **27.64** / 77.79 / 43.79 | 3822329 |
| 9B + gold-SFT LoRA, 0-shot | 26.56 / **79.15** / **48.00** | 3857589 |

Key structure in these numbers (per-source figures in EXPERIMENTS.md):

- **Few-shot's gain is answer format, not knowledge**: belebele 17.69→52.70, tydiqa
  21.88→38.94, aya flat (24.03→24.19).
- **Gold-SFT and few-shot are complementary, not ordered**: the adapter wins belebele
  (85.82), MCIF (49.26) and OEG (29.06 vs 25.55) but collapses tydiqa (38.94→19.53, below
  even the untuned base's 21.88). Whether the two stack is being measured right now
  (adapter + 3-shot, job 3858987).
- Gold-SFT *does* move the OEG source (contradicting our earlier aya-only reading) — but
  aya proper stays flat. This was read as "distillation is still the OEG lever"; **that bet was
  tested and lost** (§5.4, jobs 3865036/3866054): the distilled adapter came in last on OEG. The
  gold-SFT adapter 3822375 already banks the OEG gain and keeps the route.

## 4. Test-set alignment (roadmap A) — what changed our plans

Full analysis in TEST_SET_ANALYSIS.md; the three findings that reshaped the architecture:

1. **No multiple-choice prompts exist in the test set.** Our dev set is 38% belebele
   (1,123/2,978) in "N: option" MC format, and both few-shot's and the adapter's biggest
   dev wins are exactly there — so **dev overall chrF systematically overstates test
   transfer**. tydiqa-style free-form extraction is the honest qa-context proxy; weight it
   (and aya/OEG) when comparing systems. For the paper: a clean train/test
   distribution-shift story with numbers.
2. **Every test prompt embeds its own instructions** (format: "answer in one sentence,
   using only what the passage says"; an explicit "no answer" escape — so unanswerable
   detection is scored; word budgets "in 120–150 words" in every language; a closing
   "Answer in \<language\>"). Consequences: (a) our lang-hint system turn is
   redundant-at-best at test time (dev A/B without it = job 3859645, running); (b)
   instruction-following, especially non-English length control, is directly scored —
   the test-format smoke showed word budgets violated on 9/10 Bhojpuri OEG rows and the
   base model **false-refusing** ("no answer") on an answerable definitional question.
3. **Bhojpuri (bho, the surprise language, zero training rows) drifts**: base-model OEG
   outputs came out as standard Hindi (most rows), Nepali (one), Maithili (one). A
   fastText LID gate catches exactly this failure class.

Also: the 100 empty English OEG prompts (`qa-oeg_1..100_eng_eng`) were **fixed upstream**
on 15 July (HF `5950311`, re-downloaded 2026-07-16 — only those 100 rows changed). What
remains is narrower: 8 of them (`qa-oeg_93..100_eng_eng`) ship unsubstituted
`{country}`/`{language}` placeholders. `run_test.py` passes those through verbatim and warns.
See TEST_SET_ANALYSIS §6.

## 5. Distillation pipeline (roadmap B)

Sequence-level KD: teacher generates on the train split (never dev), answers are
quality-filtered against golds, student LoRA is trained on the filtered mix.

### 5.1 Teacher selection — measured, not assumed

Same 15 deterministic train rows (the seed-42 split makes smokes row-by-row comparable)
across three teachers:

| Teacher | Infra | Result |
|---|---|---|
| Qwen3.5-35B-A3B bf16 | 1× a100_80, transformers | fluent; hallucinated a Japanese quiz answer, an NHL draft year, invented geography (3859176) |
| Qwen3.5-27B bf16 | 1× a100_80, transformers | slightly better (fixed the Japanese answer, better MC-format compliance), own hallucinations (3859315) |
| Qwen3.5-122B-A10B GPTQ-Int4 | 2× a100_80, **vLLM** | only teacher to get both knowledge probes right (3859578) |

Decision: **122B for aya+oeg** (the knowledge-grounded, open-ended sources where teacher
quality raises the filter pass rate) = 4,126 rows, generated in one 17-minute vLLM job
(3859682); **35B for the whole corpus** (3859277-79, 3× ~16h shards) covers
belebele/tydiqa/MCIF, where teacher choice barely matters (see 5.2).

Infra note for the appendix: the 122B GPTQ checkpoint is unrunnable through
transformers+gptqmodel 7.1.0 (Marlin kernel rejects an out_features=1 layer; torch-backend
fallback hits CUDA illegal memory accesses — jobs 3859341/45/81/98); vLLM runs it natively
and its batched decode measured **~250× faster per row** than the serial transformers loop
(13s decode for 15 rows vs ~14 s/row).

### 5.2 The gold filter is load-bearing

`scripts/filter_teacher.py`: per-row **sentence chrF OR BERTScore F1** against the gold,
both thresholds calibrated per-source via `--report`'s distribution/threshold grid.

- Why OR: chrF alone kills verbose-but-correct answers (teacher answers scored against
  short golds — smoke medians were chrF ≈16 / BERTScore ≈66); BERTScore alone is too
  lenient on fluent, on-topic hallucinations. Two calibrated thresholds OR'd beat either
  alone.
- ~~**belebele behaves as a built-in fallback**: teacher answers are explanatory prose that
  never chrF-matches a "2: \<option\>" gold, so belebele rows fail the filter and keep
  their gold targets regardless of teacher.~~ **WRONG — disproved 2026-07-15 (job 3861614,
  2,706 real 35B rows from the partial shard 1). Measured pass rates at 30/70:**

  | source | pass rate | n |
  |---|---|---|
  | `wmt25-mist-oeg-gpt-4.1` | **94.4%** | 71 |
  | `FBK-MT/MCIF` | 62.6% | 163 |
  | `CohereLabs/aya_dataset` | 44.6% | 837 |
  | **`facebook/belebele`** | **33.3%** | 1,064 |
  | `copenlu/answerable_tydiqa` | 31.5% | 571 |
  | overall | 39.8% | 2,706 |

  **The error: the filter is `chrF >= 30` OR `BERTScore >= 70`, and the argument above only
  covers the chrF half.** chrF really is ~0 for prose-vs-`2: <option>` — but BERTScore
  happily scores on-topic prose against a short option at ≥70 (belebele BERTScore p75 =
  70.7). So a third of belebele rows *do* pass, and passing means the `2: <option>` gold
  target gets **replaced by prose** — destroying exactly the format the gold-SFT adapter
  learned best (85.82). There is no built-in fallback; it has to be built.
- **Any "the filter will catch it" claim must be checked against BOTH halves of the OR.**
  The same slip was made a second time in an earlier draft of §5.4 (tydiqa "cannot match a
  2-word gold, so it falls back") — measured 31.5% pass, on a source that *does* reach the
  test set.
- Teacher choice therefore matters everywhere the filter can pass rows, which is everywhere.
- Empty teacher answers auto-fail. Mix policies: `replace` (default; teacher-where-passed
  else gold — keeps the dataset identical in size/rows to the gold-SFT run 3822375, so the
  distilled-vs-gold comparison stays one-variable), `both`, `teacher`.
- Output feeds `train_lora.py --data` (same columns as the HF split); the distilled
  adapter is trained **fresh** from the base, never continued from the gold adapter.

### 5.3 Known open issue: the metric-vs-human tradeoff

The teacher's style is verbose markdown (headers/bold/emoji in OEG answers); golds are
short and dry. Automatic metrics score against golds, but **human eval** (which decides
the primary submission's ranking) plausibly prefers teacher-style answers. Threshold
choice is therefore a metric-vs-human tradeoff; we may deliberately keep a looser filter
for OEG rows. Decide after the filter report on real data (job 3860144).

**Caveat added 2026-07-15 — "golds are short and dry" is true of aya/tydiqa, NOT of OEG.**
The OEG golds are GPT-4.1 outputs: median **175 words**, with markdown (`**bold**`,
`<br><br>`) already in them. So on the one source that actually reaches the test set as
`qa-oeg`, the gold is already a strong model's verbose answer, and the human-eval argument
for preferring teacher style over gold style largely does not apply there.

### 5.4 The distillation value cross (RESOLVED 2026-07-17 — the argument held)

> **Measured verdict (jobs 3865036 + 3866054): distillation LOST to plain gold SFT** on both
> columns that decide anything — `qa-context` MCIF **−11.92** COMBINED, `qa-oeg` long-form OEG
> **−12.37**, `qa-oeg` short-answer aya **+1.28** (matched Δ, both sides `--no-lang-hint`). This is
> the crossing below, now measured: the one column the distilled adapter wins (aya) does not reach
> the test set, and it is *last of all systems* on OEG, the column that does. The confound
> ("distilled also dropped the lang-hint") was closed by 3866054 — the gold adapter re-scored
> `--no-lang-hint` moved ≤0.6 COMBINED from its hinted self, so the ~12-point loss is the **teacher
> data**, not the hint. **No routing change: gold-LoRA 3822375 keeps `qa-context` and `qa-oeg`.**
> Full tables in EXPERIMENTS.md ("distillation did not pay off"). The argument below predicted this
> and is kept as the reasoning of record.

Distillation's premise is "the teacher's answer is a better training target than the gold".
Lining that premise up against where each source actually lands at test time
(TEST_SET_ANALYSIS §5b/§5c) produces an uncomfortable crossing:

| source | train rows | what the gold is | measured pass @30/70 (3861614) | teacher upside | reaches the test set? |
|---|---|---|---|---|---|
| `aya` | 3,763 | human, 24 words (p50) | 44.6% | **high** — teacher writes fuller answers | **no** (§5b: wrong length regime, resembles neither test task) |
| `wmt25-mist-oeg-gpt-4.1` | 363 | **GPT-4.1, 175 words, markdown** | **94.4%** | **unclear** — gold is already a strong model's output, and the filter passes almost everything, so it is barely filtering at all | **yes** — this is `qa-oeg` |
| `copenlu/answerable_tydiqa` | 2,497 | 2-word extraction | 31.5% | **questionable** — a third of rows swap a 2-word extraction target for prose | yes — this is `qa-context` |
| `facebook/belebele` | 4,577 | `2: <option>` | 33.3% | **negative** — a third swap the MC-format gold for prose, wrecking the one thing gold-SFT nailed (85.82) | **no** (no MC at test) |

**Read the two right-hand columns together: distillation has the most headroom exactly where
it does not transfer (aya), and the least headroom where it matters (OEG).** Additionally,
the gold-SFT adapter 3822375 *already* banks the OEG gain (29.06 vs 3-shot's 25.55) — it was
trained on those same GPT-4.1 golds. The distilled adapter's marginal value on `qa-oeg` is
therefore "how much better than GPT-4.1 is the 122B", filtered down to the subset that
resembles GPT-4.1.

This was an argument, not a measurement — and it warned it could be wrong: §5.1 found the 122B
was the only teacher to get both knowledge probes right, so it might genuinely beat GPT-4.1 on
some rows, and the human-eval channel is not captured by any number we have. **We committed the
training time anyway (user's call) and measured it: the argument held** (verdict box above). Roadmap
B's headline bet — "distillation is the lever, OEG is the scoring headroom" — did not survive contact
with the proxies; the distilled adapter came in *last* on OEG. The human-eval caveat is the one live
thread the automatic metrics cannot close, but it is not enough on its own to deploy an adapter that
loses ~12 COMBINED on the faithful proxy.

### 5.5 Per-source thresholds are now required, not optional

Measured pass rates (§5.2 table) make a single global threshold indefensible — the same
30/70 does something different, and something wrong, on each source:

- **belebele (4,577 rows): must always keep gold.** 33.3% pass today, and every passing row
  replaces a `2: <option>` target with prose. There is no upside to weigh against it —
  belebele does not reach the test set (no MC), so the passing rows buy nothing and cost the
  format the gold-SFT adapter learned best. It also **confounds the headline experiment**:
  a distilled adapter trained on corrupted belebele targets will crater on belebele in dev,
  drag the overall column down, and look like "distillation failed" when the only thing that
  failed is a source we already decided is noise (EXPERIMENTS.md warning).
- **OEG (363 rows): 94.4% pass = the filter is not filtering.** Whatever threshold we pick
  here is close to a no-op; the real question is §5.4's (is the 122B better than GPT-4.1 at
  all), and the filter cannot answer it.
- **tydiqa (2,497 rows): 31.5% pass, and this one reaches the test set.** Swapping a 2-word
  extraction gold for teacher prose on a third of rows is a real intervention on a source
  that matters. Note gold-SFT *already* collapsed on tydiqa (38.94→19.53) while training on
  the clean 2-word golds, so this is the source where target choice is least understood.

**Corollary for the running teacher shards:** the ~4,577 belebele rows are compute spent on
answers we should never use. Not worth killing 3859277-79 over — they were ~68% done at
2026-07-15 23:00 with ~5h left, so the saving is ~2h against the risk of losing 10h of work.
Exclude belebele at *filter* time instead, which is free. A `--source` policy (per-source
thresholds, or a "this source is gold-only" list) is the missing piece in
`scripts/filter_teacher.py`.

## 5.6 The runaway-generation artifact (2026-07-18) — smoke + post-mortem record

LoRA adapters (v2 gold 3867139 worst at 66% of predictions; OLD gold 3857589 at 78%
tydiqa / 56% aya — the "tydiqa collapse" 38.94→19.53 retro-explained as truncation,
not capability loss) continue past the answer into hallucinated chat turns rendered
as plain text (`\nuser\n…\nassistant\n<think>`), especially after short golds. Base
models were never affected (0% incidence). Facts established while debugging:

- **Label masking in `train_lora.py` is NOT the bug**: reproduced locally with the
  Qwen3.5-9B tokenizer — `prefix_ids` is an exact token prefix of `full_ids` and the
  label span is `answer + <|im_end|> + \n`, so EOS is trained. The fine-tuned model
  under-samples it after *short* golds at T=0.7/top-p 0.8/top-k 20; templated (old
  belebele "2: option") and long golds don't trigger it.
- **Measure refusals only on truncated predictions**: the correct refusal string is
  usually there, with junk after it — raw-CSV refusal metrics undercount by ~25× (3.9%
  vs the true 92.7%).
- Fix = `prompt_template.RUNAWAY_STOP_STRINGS` (generate-time halt, saves budget) +
  `truncate_runaway()` on every decode (byte-identical to the 3869088 re-score cleanup
  on all 2,949 rows). Smoke **3869113** (existing v2 adapter, 60 belebele dev rows —
  the slice that ran away 76% of the time): **0/60 contaminated**, p50 prediction
  length 18 chars, 15m45 wall.

## 6. Infrastructure record (reproducibility appendix material)

- **atuin ($WORK) group file quota exceeded** since 2026-07-14 (578K→561K / 500K files,
  grace expired; our share ~85K): all atuin writes fail. Hybrid layout: active clone in
  $HOME, reads (venv, hf_cache, adapters) from atuin, prepared-datasets cache copied to
  $HOME. Every sbatch script now probes atuin writability at runtime and falls back to the
  $HOME cache — two jobs (3857583, 3859591) died in <40s from an unwritable lock path
  before this was baked in.
- **Both /home/hpc and /home/vault double-count usage against quota** (every file's
  allocated blocks measure exactly 2.00× apparent size; mirrored storage). Budget at 2×
  nominal: the three teacher checkpoints (35B bf16 + 27B bf16 + 122B int4 ≈ 210GB nominal)
  sit on $HPCVAULT (1TB soft quota) as ~403G accounted.
- **Login node is for submitting, not computing**: a BERTScore pass run via nohup on alex1
  pegged ~67 cores and was killed by the admin. Everything — including "quick" scoring —
  goes through sbatch (slurm/filter_teacher.sbatch exists for exactly this).
- Three venvs: `$WORK/mist-venv` (main transformers stack), `$HOME/mist-venv2`
  (+gptqmodel/optimum/torchvision; only needed for the transformers-GPTQ dead end),
  `$HOME/vllm-venv` (vllm + datasets/pandas; pins its own torch).
- Measured runtimes that set the sbatch budgets: 2B 0-shot 5h36, 2B 3-shot 3h29, 9B 0-shot
  6h01, 9B 3-shot 3h42, 9B LoRA SFT 6h35, 9B LoRA eval 6h52, scoring 2,978 rows 1m11s,
  35B teacher ~14 s/row (~16h per 1/3 shard), 122B-vLLM 4,126 rows in 17m10s.
- 🟡 **`run_test.py` throughput on the official test set is strongly task-dependent — and
  the first estimate here was wrong because it generalised from the slow task.** Measured
  on one a40, 9B + LoRA, max-new-tokens 512:
  - **`qa-oeg` ≈ 20.5 s/row** (jobs 3875151/3875152, 2,359 rows in 13h25 each). Test
    prompts are long and the answers run to the full budget, so the 512-token ceiling is
    actually reached. This is ~3× the 4.5–8.3 s/row the `slurm/run_test.sbatch` header
    projects from benchmark.py's dev rates, and that header is still stale.
  - **`qa-context` ≈ 5.9 s/row** (jobs 3876525/3876526, 360 bho rows in 35:13 / 29:34).
    The prompt asks for **one sentence**, so generation stops early and the ceiling is
    never approached.
  - ⚠️ **This note first recorded the full qa set as "~67h, nearly 3× the wall" — that was
    an extrapolation of the qa-oeg rate onto all 10,999 rows, and it is wrong.** qa-context
    is **8,640 of those rows (79%)**. Correct arithmetic: 8,640×5.9 + 2,359×20.5 ≈
    **27.5h**. Still over the 24h partition wall, so the final runs must still shard
    (`--shard i/n`, separate `--out` each, concatenate; `run_test.py` also resumes into an
    existing `--out`) — but **n=2 (13.8h/shard) is enough**, where n=4–6 was planned.
  - Caveat on the qa-context rate: measured on bho only. The one-sentence instruction is
    attested in every language's `context_tail`, so it should carry, but re-check on a
    second language before committing the final schedule.

## Per-language dev breakdown — what it is, and what it cannot say (2026-07-21)

Requested by Tsedi for the writing. `scripts/evaluate.py` has printed a
`by language (best to worst chrF)` block all along (it groups on `lang_code`), and
`benchmark.py --data data/dev_v2.jsonl` renames **`question_lang` → `lang_code`** and drops
the sum-sum rows, so the breakdown is keyed on the **question language**, not the context
language. On the cross-lingual sources (belebele especially) those differ; say so in the
paper. No new GPU run is needed for the plain-v2 adapter — the table is already in
`logs/mist-qa-eval-3869088.out` (adapter 3867139, n=2,949, 24 languages).

Three caveats that must travel with the table:

- **The language inventory changed with the split.** The v2/item-split dev has **24**
  languages where the old row-split dev had 27: **`tha`, `swh`, `tel` are gone entirely**,
  and several languages' `n` moved a lot (e.g. `ben` 32→129, `ces` 11→60, `eng` 109→172,
  `fin` 65→129). That is an artefact of rebuilding the split, not of the model. Any
  old-vs-new per-language comparison is therefore **not** like-for-like, and the paper needs
  one sentence saying so.
- **dev↔train leakage makes dev optimistic** (`DATA_AUDIT.md`, 2026-07-17: the old split was
  by row, so parallel items straddled train and dev). Read the per-language numbers as a
  **relative** ranking across languages, never as an absolute capability estimate.
- **dev is structurally blind to our two actual contributions.** `dev_v2` contains **0 bho
  rows and 0 budgeted rows** (both enumerated). The real margins live on test: bho **40 vs
  12** and budget compliance **65.8% vs 44.9%**. So this table is an **in-domain language
  breakdown** and nothing more — the C and D results belong in a separate section, scored on
  test outputs via `scripts/verify_outputs.py`.

One more trap when copying numbers out: evaluate.py now labels the single `overall` line
**"LEGACY -- 71% noise, do not compare systems on this"**, because dev's source weighting is
inverted against the test mix. Quote it only for continuity with the older table, and never
use it to rank systems.

## The aya "style shift" — mostly a chrF artifact, no action (2026-07-23)

Adapter answers aya tersely (11 words vs gold's 26, matching the gold register) while base
pads with markdown (105 words, 89% markdown-formatted) — that verbosity is what inflates
base's chrF (character-recall), while BERTScore/ROUGE-L correctly prefer the adapter. No fix
needed.

## Paired bootstrap — the edges are marginal and opposite (2026-07-23)

The dev qa-oeg aggregate is 87% a 90-row column: across all four adapters aya (n=944) spans 0.28
points, OEG (n=90) spans 2.83, and the moving part is OEG's ROUGE-L. `scripts/bootstrap_compare.py`,
10,000 paired resamples, reference = C+D-small, on the faithful proxy for each sub-task:

| vs C+D-small | OEG (qa-oeg, n=90), job 3889564 | MCIF (qa-context, n=160), job 3889735 |
|---|---|---|
| C-only | **+1.79**  CI [+0.08, +3.58]  p=0.041 | **−2.44**  CI [−4.96, −0.02]  p=0.048 |
| C+D | −1.04  CI [−3.55, +1.45]  p=0.394 | −0.57  CI [−2.78, +1.61]  p=0.621 |
| plain | +0.62  CI [−1.52, +2.75]  p=0.583 | −0.73  CI [−3.35, +1.85]  p=0.600 |

**The only gaps that clear noise are C-only vs C+D-small, and they point opposite ways** (both CIs
graze zero, p just under 0.05): C-only wins qa-oeg long-form, C+D-small wins qa-context. Every
plain/C+D comparison is noise — the earlier "+0.88 over C+D" and "+1.01 over plain" rankings did
not survive. Direct plain vs C-only is also a wash both ways (OEG +1.16 p=0.225, MCIF −1.72 p=0.147;
jobs 3889793/3889794).

## D on qa-context (jobs 3876525 / 3876526 / 3882159)

The 360 bho qa-context rows are too short for `bho_lid` (median 3 words, one-sentence answers).
Script/sentence/refusal checks call the adapters identical. What separates them is **contrastive
function words** (which of `आ`/`खातिर` vs `और`/`के लिए` appears, needs only one word):

| | C+D | **C+D-small** | plain |
|---|---|---|---|
| bho-leaning / hin-leaning | **163 / 1** | 147 / 16 | 21 / 93 |
| % bho of the decidable | **99%** | **90%** | 18% |

This is the one check where C+D-small reads weaker than C+D (90% vs 99%: 16 hin-leaning vs 1, of
163 decidable): the only place more bho data bought more bho output. Everything else is unchanged
(Devanagari 99.7%, refusal 21.4%). Caveats: only 39% of rows are long enough to judge, and this
shows output that *looks* bho, not that it's *correct*; the test set has no gold.
