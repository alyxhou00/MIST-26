# Experiment log — v2 item-split dataset (from 2026-07-17)

This is the log for the rebuilt dataset, **`data/train_v2.jsonl` / `data/dev_v2.jsonl`**
(built by `scripts/build_dataset.py`, seed 42 — schema in README "The rebuilt train/dev
set"). The previous log ([EXPERIMENTS.md](EXPERIMENTS.md)) is closed: its train/dev split
leaked parallel items across the two, so its dev numbers aren't comparable to anything
measured here. Don't copy rows across.

Not every old finding needs re-running, though. Comparisons that never depended on the
split — prompting vs. prompting, distillation vs. gold SFT on the same data, adapter+demos
not stacking — still hold as-is. What doesn't hold is any comparison *between* prompting
and an adapter, since the adapter was trained on the leaky split; those are what the runs
below re-establish.

## The three adapters (proposed submission variants as of 07-23)

| variant | adapter | Hub repo (`alyxhou00/`) | one-line |
|---|---|---|---|
| plain | 3867139 | `mist-qa-qwen3.5-9b-lora` | v2 SFT, no C, no D — the baseline |
| C-only | 3876434 | `mist-qa-qwen3.5-9b-lora-wordcnt` | + word-budget compliance; quality = plain within noise |
| **C+D-small** | 3880753 | `mist-qa-qwen3.5-9b-lora-wordcnt-bho` | + bho pack at 17.1%; **the primary submission** |

**"C" and "D" in the docs are internal roadmap shorthand** (full roadmap kept locally, not in this repo):

- **C** = constraint-augmentation: also called word budget or word count compilance, that is 
  training rows that state a word-budget instruction, so
  the model learns to respect stated length constraints.
- **D** = the Bhojpuri (`bho`) data pack folded into SFT, since the sample data has zero
  native `bho` coverage otherwise.

## Ground rules

- One row per full SLURM job; smokes and post-mortems go to IMPLEMENTATION_NOTES.md,
  logs to `logs/`.
- Metric: **COMBINED = mean(chrF, BERTScore, ROUGE-L)** per sub-task column, never pooled
  across tasks (rule and caveats: ROADMAP todo #2).
- **Score per task × source column.** The v2 dev has real qa-context proxies now
  (belebele-v2 and tydiqa-v2 are in test format, cross-lingual incl. unanswerables;
  MCIF remains the long-context one), so qa-context is no longer MCIF-only — but keep
  the columns separate.
- **qa-oeg aggregate = 0.87·OEG + 0.13·aya** (test shape, DATA_AUDIT §7) — never the
  pooled mean (dev is 944 aya vs 90 OEG rows, weight is inverted).
- Refusal behaviour is now measurable: dev has gold-refusal rows (belebele 7%,
  tydiqa 20%, exact per-language phrases). Report false-refusal / missed-refusal rates
  alongside COMBINED for qa-context runs.
- The split is a pure function of `item_group` — any resampling, augmentation
  (augment_constraints) or bho-pack mixing must keep a group on one side. Never re-split
  by row.

## Baselines to establish (the re-run queue)

1. 9B base, 0-shot, no hint, test-format dev (the run_test.py-equivalent condition).
2. 9B base, 3-shot (variant1 safety config).
3. Gold-LoRA retrained on `train_v2` (0-shot, test format — same recipe as 3822375 but
   new substrate), scored on dev_v2.
4. Constraint-augmented + bho-pack SFT on the same substrate.

## Runs

All scores are **COMBINED** = mean(chrF, BERTScore, ROUGE-L) per source column, computed
from the per-source lines in the job log. qa-oeg agg = 0.87·OEG + 0.13·aya. Refusal columns
(new in v2): FR = false-refusal rate on answerable belebele+tydiqa rows; RH = refusal-hit
rate on gold-refusal rows (belebele/tydiqa averaged, full split in the analysis note below).

| job | date | config | belebele-v2 | tydiqa-v2 | MCIF | OEG | aya | qa-oeg agg | FR | RH | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3867141 | 07-18 | 9B base, 0-shot, no hint (run_test-equivalent) | 34.40 | 51.51 | 42.44 | 34.30 | 33.84 | 34.24 | 7.3% | 74.6% | dev_v2 qa (n=2,949), 4h00. Clean generations (0% runaway). |
| 3867142 | 07-18 | 9B base, 3-shot (train_v2 pool), no hint | **37.25** | **57.37** | **45.67** | 34.40 | 34.95 | **34.47** | 11.5% | **92.7%** | 3h41. Demos help everywhere on qa-context; qa-oeg ≈ flat. Refusal-hit 93% but false-refusals up (tydiqa 17.7%). |
| 3867139 | 07-18 | LoRA SFT on train_v2 qa (11,674 rows, no hint, recipe = 3822375) | — | — | — | — | — | — | — | — | train_loss 0.9656, 6h19. Adapter: `adapters/qwen3.5-9b-qa-lora-3867139`. |
| 3867140 | 07-18 | ↑ adapter, 0-shot, no hint | ~~24.76~~ | ~~39.96~~ | ~~36.73~~ | ~~40.73~~ | ~~32.16~~ | ~~39.62~~ | 0% | 3.9% | 9h13. 🔴 **INVALID — 66% of predictions are runaway generations** (see the runaway note below). Truncated re-score = 3869088. |
| 3869088 | 07-18 | ↑ same CSV, truncated at the first runaway marker, re-scored | **44.33** | **72.04** | **50.95** | **40.53** | **36.43** | **39.99** | **1.3%** | **92.7%** | 3-min re-score (1,934/2,949 rows truncated). The v2 adapter beats base and 3-shot on every column of the honest split — but only with the inference-side stop fix. Refusal rows trained the escape (refusal-hit belebele 97.5% / tydiqa 88.5%, false-refusals 0.1% / 4.9%). |
| 3869129 | 07-19 | **C+D SFT**: LoRA on `train_v2-cd.jsonl` (19,683 rows = 11,674 qa + 8,009 bho-pack; 840 qa-oeg rows carry a word budget), no hint | — | — | — | — | — | — | — | — | train_loss **1.156** (the bho pack is 41% of the mix, continuation/translation text), 11h23. Adapter: `adapters/qwen3.5-9b-qa-lora-3869129`. |
| 3869130 | 07-19 | ↑ adapter, 0-shot, no hint (stop-fix in place) | 44.70 | 70.72 | **51.11** | 38.86 | **36.65** | 38.57 | — | — | 5h06. C+D does not beat plain on dev: qa-context a wash, qa-oeg agg −1.42 — but dev is blind to both things C+D adds (see below). |
| 3876434 | 07-21 | **C-only SFT**: LoRA on `train_v2-c.jsonl` — the same 11,674 examples as 3867139, 840 carrying a word budget, **0 bho rows**. Isolates C from D | — | — | — | — | — | — | — | — | train_loss **0.9648** (vs plain 0.9656, C+D 1.156) — the budget sentences are free to fit; the C+D loss increase was entirely the bho pack. 6h27. Adapter: `adapters/qwen3.5-9b-qa-lora-3876434`. |
| 3880737 | 07-22 | ↑ C-only adapter 3876434, 0-shot, no hint | 44.69 | **72.72** | 49.23 | **41.69** | 36.37 | **41.00** | — | — | 5h11, 0 runaway. Deltas vs plain 3869088: belebele +0.36, tydiqa +0.68, MCIF −1.72, OEG +1.16, aya −0.06. |
| 3880753 | 07-22 | **C+D-small SFT**: LoRA on `train_v2-cd-small.jsonl` (14,074 rows = 11,674 qa + 2,400 bho-pack at **17.1%**; same 840 budget rows), no hint | — | — | — | — | — | — | — | — | train_loss **1.087**. 7h27. Adapter: `adapters/qwen3.5-9b-qa-lora-3880753`. Chained evals: 3882157 (dev), 3882158 (test qa-oeg), 3882159 (test qa-context bho). |
| 3882157 | 07-23 | ↑ C+D-small adapter 3880753, 0-shot, no hint | **44.78** | **72.79** | **51.68** | 39.90 | 36.45 | 39.45 | — | — | 5h13, 0 runaway. Best-or-tied on all three qa-context columns (n=1,915); qa-oeg agg mid-pack at 39.45. Read it with the bootstrap before ranking — the four-system spread lives in the 90-row OEG column. |

### 🔴 Runaway-generation artifact (found 2026-07-18)

After a correct short answer, the fine-tuned models sometimes kept going, inventing a fake chat
exchange (`\nuser\n…\nassistant\n<think>`) that got scored as part of the answer and dragged the
numbers down. This is what made one earlier model look much worse at tydiqa (38.94→19.53): not
lost ability, just 78% of answers running on. **Fixed 2026-07-18**: generation now stops at the
first fake turn, and any stragglers are trimmed before scoring. Every adapter from job 3869088 on
is clean; base models never had this. Full record: IMPLEMENTATION_NOTES.md §5.6.

### C and D on the test set (dev is blind to both)

The dev set can't evaluate what C+D is for: it has no budget rows and no bho rows. So dev only
measures the downside (piling on bho data dilutes the main qa task) and never the upside. For the
upside we have to look at the official test set. In the table below, the dev columns come from the
job logs, and the test columns were all re-scored together on the same test file.

| | plain-v2 | C+D (bho 40.7%) | C-only (bho 0%) | **C+D-small (bho 17.1%)** |
|---|---|---|---|---|
| SFT job / eval job | 3867139 / 3869088 | 3869129 / 3869130 | 3876434 / 3880737 | **3880753 / 3882157** |
| train_loss | 0.9656 | 1.156 | 0.9648 | 1.087 |
| dev belebele-v2 | 44.33 | 44.70 | 44.69 | **44.78** |
| dev tydiqa-v2 | 72.04 | 70.72 | 72.72 | **72.79** |
| dev MCIF | 50.95 | 51.11 | 49.23 | **51.68** |
| dev OEG (n=90) | 40.53 | 38.86 | **41.69** | 39.90 |
| dev aya (n=944) | 36.43 | **36.65** | 36.37 | 36.45 |
| **dev qa-oeg agg** | 39.99 | 38.57 | **41.00** | 39.45 |
| C: budget compliance (465) | 44.9% | **65.8%** | 61.3% | 64.7% |
| ↳ under / over | 51.2% / 3.9% | **27.1%** / 7.1% | 28.0% / 10.8% | 27.5% / 7.7% |
| D: bho_lid qa-oeg (100), bho/hin/abstain | 12 / 67 / 21 | **40** / 36 / 23 | 12 / 64 / 24 | **40** / 35 / 23 |
| D: bho qa-context (360), % bho | 18% | **99%** | — | 90% |
| `<br>` markup | 27.9% | 31.0% | 28.2% | **26.7%** |

- **C is free.** train_loss = plain within noise; dev quality = plain within noise (bootstrap
  in IMPLEMENTATION_NOTES.md); +16.4pp compliance, and the failure it targets — under-shooting — halved (51.2%→28.0%).
  Conditional, not padding: on the 1,894 non-budget test rows the systems match (median 73 vs 81
  words), only the 465 budget rows move (119→149). The compliance gap to C+D (61.3 vs 65.8) is all
  *over*-shooting; under-shoot is fixed equally by both.
- **D is 100% the bho pack, no transfer.** Dropping it (C-only) returns bho to 12%, exactly
  plain's. So the open question was ever only the *proportion*, not whether to keep D.
- **C+D-small keeps all of D at 17.1%.** bho_lid 40/35/23 ≈ C+D's 40/36/23 (same 40 rows);
  qa-context 90% vs 99% bho, still 5× plain; compliance 64.7%. **2,400 bho rows buy what 8,009
  did.** It recovers 62% of the dilution (qa-oeg agg 39.45, between C+D's 38.57 and C-only's
  41.00) — not linear in mix share. Pack = 1,400 fineweb-2 (120-word median, the only half showing
  paragraph-length bho) + 1,000 xP3x (24-word median, anti-drift).
- **Both flaws are C- and D-independent:** degenerate repetition ~2% in all; `<br>` markup ~27–31%
  in all (from the web-scraped substrate) — a one-line submission-time strip is the cheapest
  unclaimed point.

### Decision: C+D-small for the whole qa set

qa-context is **79% of qa** (8,640 of 10,999); C+D-small wins it (best-or-tied on MCIF and the 1,915
dev qa-context rows) and wins bho, while C-only is significantly *worst* on qa-context. C-only's only
edge is the 90 OEG rows. So one model — C+D-small — for everything; C-only is at most a qa-oeg-only
variant. (Routing non-bho to C-only was considered and dropped: it would hand 79% of the rows to the
model that loses them.)

### Final submission generation (2026-07-23, in progress)

For the final QA submission we use a single model, C+D-small, across the whole QA test set.

We run it with exactly the settings we used when comparing the candidate models (no `--unescape`,
no `--lang-hint`, default sampling). The reason: every number the decision was based on was measured
with these settings, so the submission has to match them or it would be inconsistent with those
results. (`--unescape` is a variant we may try later, not the main submission.)

The submission must use the latest version of the test set (`data/tests-07-20.jsonl`). We generate
the outputs on that test set.

---

*This is co-authored by Claude.*
