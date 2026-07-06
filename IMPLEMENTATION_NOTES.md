# WMT26 MIST Few-Shot Implementation Notes

Session date: 2026-07-06
Task: Add few-shot prompting to the zero-shot QA benchmark and refactor sbatch naming convention.

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
