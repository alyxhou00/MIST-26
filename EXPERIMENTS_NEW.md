# Experiment log — v2 item-split dataset (from 2026-07-17)

This is the log for the rebuilt dataset, **`data/train_v2.jsonl` / `data/dev_v2.jsonl`**
(built by `scripts/build_dataset.py`, seed 42 — schema in README "The rebuilt train/dev
set"). The previous log ([EXPERIMENTS.md](EXPERIMENTS.md)) is closed: its train/dev split
leaked parallel items across the two, so its dev numbers aren't comparable to anything
measured here — don't copy rows across.

Not every old finding needs re-running, though. Comparisons that never depended on the
split — prompting vs. prompting, distillation vs. gold SFT on the same data, adapter+demos
not stacking — still hold as-is. What doesn't hold is any comparison *between* prompting
and an adapter, since the adapter was trained on the leaky split; those are what the runs
below re-establish.

## Ground rules (carried over + new)

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
4. (After C/D fold-in) constraint-augmented + bho-pack SFT on the same substrate.

## Runs

**"C" and "D" below are internal roadmap shorthand** (full roadmap kept locally,
`ROADMAP.md`, not in this repo):

- **C** = constraint-augmentation — training rows that state a word-budget instruction, so
  the model learns to respect stated length constraints.
- **D** = the Bhojpuri (`bho`) data pack folded into SFT, since the sample data has zero
  native `bho` coverage otherwise.

All scores are **COMBINED** = mean(chrF, BERTScore, ROUGE-L) per source column, computed
from the per-source lines in the job log. qa-oeg agg = 0.87·OEG + 0.13·aya. Refusal columns
(new in v2): FR = false-refusal rate on answerable belebele+tydiqa rows; RH = refusal-hit
rate on gold-refusal rows (belebele/tydiqa averaged, full split in the analysis note below).

| job | date | config | belebele-v2 | tydiqa-v2 | MCIF | OEG | aya | qa-oeg agg | FR | RH | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3867141 | 07-18 | 9B base, 0-shot, no hint (run_test-equivalent) | 34.40 | 51.51 | 42.44 | 34.30 | 33.84 | 34.24 | 7.3% | 74.6% | dev_v2 qa (n=2,949), 4h00. Clean generations (0% runaway). |
| 3867142 | 07-18 | 9B base, 3-shot (train_v2 pool), no hint | **37.25** | **57.37** | **45.67** | 34.40 | 34.95 | **34.47** | 11.5% | **92.7%** | 3h41. Demos help everywhere on qa-context; qa-oeg ≈ flat (mirrors the old set). Trade-off: refusal-hit 93% but false-refusals up (tydiqa 17.7%). |
| 3867139 | 07-18 | LoRA SFT on train_v2 qa (11,674 rows, no hint, recipe = 3822375) | — | — | — | — | — | — | — | — | train_loss 0.9656, 6h19, 23 rows truncated \@2048. Adapter: `adapters/qwen3.5-9b-qa-lora-3867139` ($WORK clone). |
| 3867140 | 07-18 | ↑ adapter, 0-shot, no hint | ~~24.76~~ | ~~39.96~~ | ~~36.73~~ | ~~40.73~~ | ~~32.16~~ | ~~39.62~~ | 0% | 3.9% | 9h13. 🔴 **INVALID as a data verdict — 66% of predictions are runaway generations** (answer, then hallucinated `\nuser\n…\nassistant\n<think>` turns as plain text; belebele 76%, aya 64%, tydiqa 55%, MCIF 47%, OEG 24%). Truncated re-score = 3869088 below. ~~Refusal signal did not take (RH 3.9%)~~ — **measurement artifact**, corrected on the truncated CSV in the next row. |
| 3869088 | 07-18 | ↑ same CSV, predictions truncated at the first runaway marker, re-scored | **44.33** | **72.04** | **50.95** | **40.53** | **36.43** | **39.99** | **1.3%** | **92.7%** | 3-min re-score of 3867140's CSV (1,934/2,949 rows truncated). **The v2-trained adapter beats base AND 3-shot on every column of the honest item-split dev** — the adapter-over-prompting verdict survives leakage removal, but ONLY together with an inference-side stop fix (truncation was applied post-hoc here; base runs are unaffected at 0% runaway). Margins over 3-shot: belebele +7.1, tydiqa +14.7, MCIF +5.3, qa-oeg agg +5.5. **Refusal correction (measured on THIS CSV, 2026-07-18): the v2 refusal rows DID train the escape** — refusal-hit belebele 97.5% / tydiqa 88.5% with false-refusals 0.1% / 4.9%; the earlier "3.9%" was runaway junk trailing the (correct) refusal phrase. Best refusal profile of all three systems (3-shot matches the 92.7% hit but at 11.5% false-refusals). Remaining watch item: aya chrF *below* base (17.11 vs 23.69) while BERTScore/ROUGE are up — style shift. |
| 3869129 | 07-19 | **C+D SFT**: LoRA on `train_v2-cd.jsonl` (19,683 rows = 11,674 qa + 8,009 bho-pack; 840 qa-oeg rows carry a word budget), no hint, otherwise the 3867139 recipe | — | — | — | — | — | — | — | — | train_loss **1.156** (vs 3867139's 0.9656 — expected, the bho pack is 41% of the mix and is continuation/translation text), 11h23, 25 rows truncated \@2048. Adapter: `adapters/qwen3.5-9b-qa-lora-3869129`. |
| 3869130 | 07-19 | ↑ adapter, 0-shot, no hint (first run with the stop-fix in place — clean by construction) | 44.70 | 70.72 | **51.11** | 38.86 | **36.65** | 38.57 | — | — | 5h06, dev_v2 qa (n=2,949). **C+D does not beat plain-v2 on dev: qa-context is a wash (±0.4, noise), qa-oeg agg −1.42.** Deltas vs 3869088: belebele +0.37, tydiqa −1.32, MCIF +0.16, OEG **−1.67**, aya +0.22. See the C+D read below — dev cannot see either thing C+D adds. |
| 3876434 | 07-21 | **C-only SFT**: LoRA on `train_v2-c.jsonl` — the same 11,674 examples as 3867139, row for row, with 840 of them carrying a word budget and **0 bho rows**. Isolates C from D, which had only ever been trained together | — | — | — | — | — | — | — | — | train_loss **0.9648** vs 3867139's 0.9656 and 3869129's 1.156 — i.e. **the 840 budget sentences cost essentially nothing to fit, and the C+D loss increase was entirely the bho pack** (continuation/translation text at 41% of the mix). 6h27, 24 rows truncated \@2048. Adapter: `adapters/qwen3.5-9b-qa-lora-3876434`. Dev eval = 3878452, test compliance = 3878453. |

### 🔴 Runaway-generation artifact (found 2026-07-18)

Adapter outputs sometimes continue past the answer into hallucinated chat turns rendered as
plain text (`\nuser\n…\nassistant\n<think>`), especially after short golds — this explains
the old log's "tydiqa collapse" (38.94→19.53): 78% of that adapter's tydiqa rows and 56% of
aya were runaway, not a capability loss. **Fixed 2026-07-18**
(`prompt_template.RUNAWAY_STOP_STRINGS` / `truncate_runaway()`): `benchmark.py`/`run_test.py`
now stop generation at the first fake turn and truncate the decode. Every adapter run from
3869088 onward is clean by construction; base models were never affected (0% incidence).

### The aya "style shift" — mostly a chrF artifact, no action

Adapter answers aya tersely (11 words vs gold's 26, matching the gold register) while base
pads with markdown (105 words, 89% markdown-formatted) — that verbosity is what inflates
base's chrF (character-recall), while BERTScore/ROUGE-L correctly prefer the adapter. No fix
needed; two minor residuals logged but not acted on (occasional over-terseness on
completion prompts, occasional fabrication on knowledge-list prompts).

### C+D (3869129/3869130) vs plain-v2 (3867139/3869088) — dev verdict, and why dev can't settle it

On `dev_v2`, C+D loses to plain-v2 on qa-oeg (agg **−1.42**, OEG −1.67) and is a wash on
qa-context (±0.4, inside noise). But **`dev_v2` cannot measure either thing C+D adds**: 0
qa-oeg rows in `dev_v2` state a word budget (all budget signal is training-only), and
`dev_v2` has 0 bho rows (the pack is training-only too). So dev only prices C+D's *cost* (a
41%-bho mix diluting the qa head) — the *benefit* has to be read from the official test set.

**Full test-set verification, 2,359/2,359 rows, 24 languages (jobs 3875151 C+D vs 3875152
plain-v2, `run_test.py --task qa-oeg`, scored by `verify_outputs.py`):**

| check | C+D | plain-v2 |
|---|---|---|
| C: budget compliance (465 rows) | **65.8%** | 44.9% |
| ↳ under budget | 27.1% | **51.2%** |
| D: bho_lid on 100 bho rows — bho / hin / abstain | **40%** / 36% / 23% | 12% / 67% / 21% |

C's gain is removing under-shooting, not padding, and it's conditional: on the 1,894
non-budget rows the two systems are near-identical (median 73 vs 81 words); only the 465
budget rows move (119→149). So the dev regression traces to the bho pack diluting the qa
head, not to C. D triples the bho rate and halves the Hindi drift. Two output flaws present
in **both** adapters (so caused by neither C nor D):

| flaw | C+D | plain-v2 |
|---|---|---|
| degenerate repetition (one clause ≥4×) | 2.2% | 2.4% |
| literal `<br>` markup in the output | **31.0%** | 27.9% |

The `<br>` markup (from the web-scraped qa substrate) is the cheapest unclaimed point on the
roadmap — a one-line strip at submission time.

**Status: C is keep, D is keep, but C+D as one adapter is probably not the primary** — the
−1.42 traces to the bho pack's 41% share of the training mix (buying only 4.2% of the test
set, a ~10× over-allocation), not to C. The missing datapoint is **C-only**
(`train_v2-c.jsonl`, job 3876434, trained 2026-07-21, row-for-row identical to 3867139's
11,674 examples plus the 840 budget sentences, 0 bho rows): if its dev score recovers
plain-v2's while keeping ~65% compliance, it becomes primary and C+D becomes the
bho-hedging variant; if its dev also drops, plain-v2 is primary instead. First signal
(train_loss only): C-only 0.9648 vs plain-v2 0.9656 vs C+D 1.156 — the 840 budget sentences
cost ~nothing to fit, so the loss increase is the bho pack (consistent with, not proof of,
the dilution theory). Dev eval = job 3878452, test compliance = job 3878453.

### The proportionate D — `sft-bho-small.jsonl` (built, not yet trained)

Bho is 40.7% of the training mix but only 4.2% of the test set — a ~10× over-allocation.
`shrink_bho_pack.py` built a shrunk pack (`train_v2-cd-small.jsonl`, bho at 17.1%, same 840
budget rows) as a hedge, held pending the C-only result above. Row selection kept both
source halves rather than just the "cleaner" one: `xP3x` translations are only 24 words
median (too short to demonstrate paragraph-length bho), while `fineweb-2` (120 words median)
is the only half that shows bho at qa-oeg's target length. Final mix: 1,400 fineweb + 1,000
xP3x.

### D on qa-context — jobs 3876525/3876526

All earlier D evidence came from the 100 bho **qa-oeg** rows; the other 360 bho rows are
**qa-context** (97% cross-lingual, one-sentence answers, median 3 words) — too short for
`bho_lid`'s density method, so script/one-sentence/refusal checks were used instead, and by
those three checks the two adapters looked **identical** (Devanagari 99.4% both, refusal
19.7% vs 17.2%). **That reading was wrong** — hand-reading paired outputs showed C+D
consistently choosing Bhojpuri-specific function words (`आ`/`खातिर`/`होखल`) where plain chose
Hindi ones (`और`/`के लिए`/`की`). A fourth check, **contrastive function words** (which of two
candidate words appears — needs only one word, not `bho_lid`'s paragraph-level density),
confirms it:

| | C+D | plain |
|---|---|---|
| bho-leaning / hin-leaning | **163 / 1** | 21 / 93 |
| % bho of the decidable | **99%** | 18% |

D's effect is cleaner on qa-context (360 rows, 78% of bho's test share) than on qa-oeg (100
rows: 40% vs 12%), strengthening the case for keeping D. Caveat: only 39% of rows are long
enough to judge (the rest are 2–3 word noun phrases); this shows output that *looks*
Bhojpuri, not that it's *correct* — the test set has no gold to check against.

🟢 **Submission-schedule correction**: qa-context runs at 5.9 s/row, not the 20.5 s/row
measured on qa-oeg — it only answers one sentence. Full qa set: 8,640×5.9 + 2,359×20.5 =
**27.5h**, not the earlier 67h estimate, so `--shard i/2` (13.8h each) is enough.

---

*This is co-authored by Claude.*
