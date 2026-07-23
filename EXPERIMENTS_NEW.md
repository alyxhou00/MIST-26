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

## The three adapters (submission variants)

| variant | adapter | Hub repo (`alyxhou00/`) | one-line |
|---|---|---|---|
| plain | 3867139 | `mist-qa-qwen3.5-9b-lora` | v2 SFT, no C, no D — the baseline |
| C-only | 3876434 | `mist-qa-qwen3.5-9b-lora-wordcnt` | + word-budget compliance; quality = plain within noise |
| **C+D-small** | 3880753 | `mist-qa-qwen3.5-9b-lora-wordcnt-bho` | + bho pack at 17.1%; **the primary submission** |

C-only vs plain is settled: budget compliance 44.9% → 61.3% (test, n=465), dev quality a wash
both ways (OEG +1.16 p=0.225, MCIF −1.72 p=0.147 — neither survives the bootstrap). C+D-small
adds bho (40% vs 12% qa-oeg, 90% vs 18% qa-context) and is chosen for the whole qa set — see the
bootstrap section below for why the C-only vs C+D-small edges are marginal and opposite.

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
| 3876434 | 07-21 | **C-only SFT**: LoRA on `train_v2-c.jsonl` — the same 11,674 examples as 3867139, row for row, with 840 of them carrying a word budget and **0 bho rows**. Isolates C from D, which had only ever been trained together | — | — | — | — | — | — | — | — | train_loss **0.9648** vs 3867139's 0.9656 and 3869129's 1.156 — i.e. **the 840 budget sentences cost essentially nothing to fit, and the C+D loss increase was entirely the bho pack** (continuation/translation text at 41% of the mix). 6h27, 24 rows truncated \@2048. Adapter: `adapters/qwen3.5-9b-qa-lora-3876434`. Dev eval = 3880737, test compliance = 3878453. |
| 3880737 | 07-22 | ↑ C-only adapter 3876434, 0-shot, no hint | 44.69 | **72.72** | 49.23 | **41.69** | 36.37 | **41.00** | — | — | 5h11, dev_v2 qa (n=2,949), **0/2,949 runaway** (stop-fix clean by construction). **C-only does not merely recover plain-v2's qa-oeg, it beats it: agg 41.00 vs 39.99 (+1.01), and +2.43 over C+D.** With compliance holding at 61.3% (3878453 below) both halves of the roadmap's C-only criterion are met → **the C+D −1.42 was the bho pack diluting the qa head, not C.** Deltas vs plain 3869088: belebele +0.36, tydiqa +0.68, MCIF −1.72, OEG +1.16, aya −0.06. |
| 3880753 | 07-22 | **C+D-small SFT**: LoRA on `train_v2-cd-small.jsonl` (14,074 rows = 11,674 qa + 2,400 bho-pack at **17.1%** of the mix; the same 840 budget rows), no hint, otherwise the 3867139 recipe | — | — | — | — | — | — | — | — | train_loss **1.087**, between C+D's 1.156 and C-only's 0.9648 and roughly where halving the pack's share predicts. 7h27, 24 rows truncated \@2048. Adapter: `adapters/qwen3.5-9b-qa-lora-3880753`. Chained evals: 3882157 (dev), 3882158 (test qa-oeg), 3882159 (test qa-context bho). |
| 3882157 | 07-23 | ↑ C+D-small adapter 3880753, 0-shot, no hint | **44.78** | **72.79** | **51.68** | 39.90 | 36.45 | 39.45 | — | — | 5h13, dev_v2 qa (n=2,949), 0 runaway. **Best or tied-best on all three qa-context columns** (belebele, tydiqa, MCIF — n=1,915 between them), and the qa-oeg aggregate lands mid-pack at 39.45: it recovers 0.88 of C+D's 1.42-point loss but does not reach C-only's 41.00. Read the aggregate with the bootstrap below before ranking on it — the entire four-system spread lives in the 90-row OEG column (aya, n=944, is flat at 36.37–36.65 across all four). |

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

C-only (adapter 3876434, job **3878453**, 12h25) was scored on the same three checks and is
folded in below. All three ran against the *same* test-file revision (see the ⚠️ note at the
end of this section), so the comparison is single-variable.

| check | C+D | **C-only** | plain-v2 |
|---|---|---|---|
| C: budget compliance (465 rows) | **65.8%** | **61.3%** | 44.9% |
| ↳ under budget (the failure C targets) | 27.1% | **28.0%** | **51.2%** |
| ↳ over budget | 7.1% | 10.8% | 3.9% |
| D: bho_lid on 100 bho rows — bho / hin / abstain | **40%** / 36% / 23% | **12%** / 64% / 24% | 12% / 67% / 21% |

Two things the C-only column settles that were previously only inferred:

1. **C does not need D.** The 4.5pp gap in headline compliance (65.8 → 61.3) is entirely in
   *over*-shooting (7.1 → 10.8); the failure C actually targets — under-shooting — is fixed to
   the same degree by both (27.1 vs 28.0, against plain's 51.2). Reporting compliance as one
   number hides this.
2. **D's effect is 100% the bho pack, with no transfer.** Removing the pack returns bho to
   **12%**, identical to plain-v2, with hin drift back at 64% vs plain's 67% (inside noise).
   There is no indirect or residual benefit from having trained alongside it. This is what makes
   the open question *proportion* rather than *whether to keep D* — and it is why
   `train_v2-cd-small.jsonl` (bho at 17.1% instead of 40.7%) is the branch-(a) experiment.

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

**Status (resolved 2026-07-22): C is keep, D is keep, and the −1.42 was the bho pack.** The
open question was whether C+D's dev regression came from C or from D's 41% share of the
training mix. **C-only** (`train_v2-c.jsonl`, job 3876434 — row-for-row identical to 3867139's
11,674 examples plus the 840 budget sentences, 0 bho rows) answers it: dev qa-oeg agg **41.00**
(job 3880737), which not only recovers plain-v2's 39.99 but beats it by +1.01, while compliance
holds at 61.3%. Both halves of the criterion are met, so the dilution theory is confirmed and
the roadmap's **branch (a)** is taken: train the proportionate pack and make the final choice a
three-way between **C-only / C+D / C+D-small**. The earlier train_loss signal (C-only 0.9648 vs
plain-v2 0.9656 vs C+D 1.156) pointed the same way and is now corroborated on a second
distribution.

**The decision this settles, and the one it hands to C+D-small.** C stays in the primary
unconditionally — it is free (train_loss 0.9648 vs plain's 0.9656), it costs nothing on dev, and
it buys +16.4pp of budget compliance. D stays too, but its 40.7% share is not defensible for
4.2% of the test set, and C-only proves the cost is real while buying back nothing on bho (12%,
exactly plain's). So the primary is now **C plus a proportionate D**, and the only open number
is what "proportionate" is. C+D-small (17.1%) is the candidate; the three-way pick resolves as:

- **C+D-small is primary** if it holds dev qa-oeg ≥ ~41.00 while keeping compliance ~61–65% and
  bho near C+D's 40% / 99% — i.e. it buys D's benefit without the dilution.
- **C-only is primary, C+D becomes the bho-hedging variant**, if C+D-small's dev falls back
  toward 38.6 — that would mean 17.1% is still too much bho for the qa head to carry.

**Outcome (2026-07-23): neither branch fired cleanly — it landed at 39.45, between the two
thresholds — and writing the criterion against a 90-row column is why.** See "The proportionate
D" below for the four-way table and what the criterion should have been.

⚠️ **All test-set numbers in this section are on the pre-fix test file.** The organizers fixed
the 8 `{country}`/`{language}` placeholder rows on 2026-07-20 (sha `ad630f88`), but `/data/` is
gitignored, so the re-download never reached the cluster: jobs 3875151 / 3875152 / 3878453 —
and 3882158 / 3882159 on 07-22 — all ran against the 2026-07-16 revision. Harmless for these
comparisons (identical substrate on all four; 9 prompt-only rows differ out of 2,359) but the
**final submission must use
`--test-file data/tests-ad630f88.jsonl`**, or those 8 English qa-oeg rows get answered about a
literal `{country}`. `run_test.py`'s placeholder warning still reads "known upstream bug", which
as of 07-20 is misleading — it now means the test file is stale.

### The proportionate D — `sft-bho-small.jsonl` (training 2026-07-22, job 3880753)

Bho is 40.7% of the training mix but only 4.2% of the test set — a ~10× over-allocation.
`shrink_bho_pack.py` built a shrunk pack (`train_v2-cd-small.jsonl`, bho at 17.1%, same 840
budget rows) as a hedge, held pending the C-only result above. **That result landed on 07-22
and took branch (a), so it is now training: job 3880753** (14,074 training rows after dropping
7,227 sum-sum, 24 truncated \@2048 — same recipe, `--no-lang-hint`, so it stays single-variable
against 3869129). Chained behind it with `--dependency=afterok`: **3882157** (dev_v2 eval),
**3882158** (`--task qa-oeg`, compliance + bho_lid), **3882159** (`--task qa-context --lang bho`,
the contrastive-function-word check where D's effect reads cleanest). Row selection kept both
source halves rather than just the "cleaner" one: `xP3x` translations are only 24 words
median (too short to demonstrate paragraph-length bho), while `fineweb-2` (120 words median)
is the only half that shows bho at qa-oeg's target length. Final mix: 1,400 fineweb + 1,000
xP3x.

**Results, 2026-07-23 — all four systems, one table.** Dev columns from the job logs; test
columns re-scored in a single `verify_outputs.py` pass (job **3884948**) so every number below
comes from one instrument run over one test-file revision. C-only was never generated on the
360 bho qa-context rows, hence the `—`; it trains on 0 bho rows and scores 12% on qa-oeg's
bho_lid (exactly plain's), so there is nothing to expect there that plain does not already show.

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
| D: bho_lid qa-oeg (100) | 12 / 67 / 21 | **40** / 36 / 23 | 12 / 64 / 24 | **40** / 35 / 23 |
| D: bho qa-context (360) | 18% bho | **99%** bho | — | 90% bho |
| `<br>` markup | 27.9% | 31.0% | 28.2% | **26.7%** |

**Shrinking the pack to 17.1% kept all of D and most of the dilution.** On the qa-oeg bho rows
C+D-small is *identical* to C+D — 40/35/23 against 40/36/23, the same 40 rows within one — and
on qa-context it holds 90% bho against C+D's 99%, still five times plain's 18%. Compliance
lands at 64.7% vs C+D's 65.8%, with the same under-shoot fix (27.5% vs 27.1%, against plain's
51.2%). So **2,400 bho rows buy essentially everything 8,009 bought**, which was the branch-(a)
hypothesis and is the one clean result of this round.

**What it did not buy is the dev qa-oeg aggregate.** 39.45 recovers 0.88 of C+D's 1.42-point
loss and stops there, short of C-only's 41.00 — halving the pack's share recovered ~62% of the
loss, so the dilution is not linear in the mix proportion. Taken at face value the
pre-registered rule (above) says "C+D-small falls short of ~41.00 → C-only is primary".

⚠️ **Do not take it at face value: 87% of that aggregate is 90 rows.** Across all four systems
the aya column (n=944) spans 0.28 points end to end while OEG (n=90) spans 2.83, and the
component that actually moves is OEG's ROUGE-L: 23.91 / 26.77 / 27.37 / 30.61 for
C+D / C+D-small / plain / C-only, against a chrF spread of 22.88–23.84. A 1.5-point aggregate
gap is therefore a ~2.4-point ROUGE-L gap on ninety rows. The paired-bootstrap check
(`scripts/bootstrap_compare.py`, job **3885055**) is what decides whether that survives
resampling; until it reports, **rank on the qa-context columns and the test checks, which have
1,915 and 465–360 rows behind them, not on the aggregate.** On those, C+D-small is best or
tied-best on all three dev qa-context columns and matches C+D on every D check.

**Where this leaves the three-way.** C+D-small dominates C+D outright — same bho behaviour,
same compliance, better on all three qa-context columns, +0.88 on the aggregate, and the lowest
`<br>` rate of the four — so **C+D can be dropped from the shortlist**, and the choice is
C+D-small vs C-only. That choice is exactly the n=90 question above, with a known asymmetry:
C-only's only claim is the OEG column, while C+D-small additionally buys 40% vs 12% bho on
qa-oeg and 90% vs ~18% on qa-context, over the 4.2% of the test set that is bho.

### Paired bootstrap — the two barely-significant, opposite-direction edges (2026-07-23)

`scripts/bootstrap_compare.py`, 10,000 paired resamples, reference = C+D-small (3882157). Two
source columns, the faithful proxy for each test sub-task:

| vs C+D-small | OEG (qa-oeg, n=90), job 3889564 | MCIF (qa-context, n=160), job 3889735 |
|---|---|---|
| C-only | **+1.79**  CI [+0.08, +3.58]  p=0.041 | **−2.44**  CI [−4.96, −0.02]  p=0.048 |
| C+D | −1.04  CI [−3.55, +1.45]  p=0.394 | −0.57  CI [−2.78, +1.61]  p=0.621 |
| plain | +0.62  CI [−1.52, +2.75]  p=0.583 | −0.73  CI [−3.35, +1.85]  p=0.600 |

Read the whole table, not one cell. **The only two gaps that clear noise are C-only vs
C+D-small, and they point opposite ways and are both marginal** (both CIs graze zero, both
p just under 0.05): C-only is better on qa-oeg long-form, C+D-small is better on qa-context
cross-lingual. Every plain / C+D comparison is inside the noise on both columns — so the earlier
"C+D-small +0.88 over C+D" and "C-only +1.01 over plain" rankings do **not** survive resampling;
they were column noise. The dev qa-oeg aggregate can order C-only vs C+D-small (marginally) and
nothing else.

The mirror-image result is what settles the submission: qa-context is **79% of the qa test set**
(8,640 of 10,999), and C+D-small wins it while C-only is significantly *worst* on it (49.23,
−2.44). C-only's only win is 90 long-form proxy rows. So routing non-bho to C-only — which an
earlier draft of this plan proposed — would hand 79% of the rows to the model that loses that
79%. Dropped.

**Decision (2026-07-23): submit C+D-small for the entire qa set.** It is best-or-tied on the two
large-sample buckets (qa-context MCIF n=160, and the 1,915 dev qa-context rows), best on bho, and
its only deficit is the qa-oeg OEG column — 90 rows, marginal, 1.79 behind C-only. One model, one
generation pass. C-only becomes a possible variant submission for a qa-oeg-only arm, not the
primary.

### Final submission generation (2026-07-23, in progress)

Single model **C+D-small** (adapter `qwen3.5-9b-qa-lora-3880753`), on the **fixed** test file
`data/tests-ad630f88.jsonl`, qa only (sum-sum is the teammate's). Config **matches the validated
runs exactly** — no `--unescape`, no `--lang-hint`, default sampling — because every number the
decision rests on was measured in that configuration; flipping `--unescape` on for the final run
would ship a prompt variant with no dev proxy behind it (it stays a candidate *variant*
submission, per TEST_SET_ANALYSIS §2, not the primary).

- **qa-oeg (2,359):** reused `runs/test-qaoeg-cdsmall-3880753.jsonl` — only 9 rows differ between
  the stale and fixed files (`qa-oeg_47/93..100_eng_eng`, all English, none bho), so 2,350 outputs
  are valid as-is; the 9 changed rows resume-regenerated on the fixed file → `runs/submit-cdsmall-qaoeg.jsonl`.
- **qa-context (8,640):** generated fresh on the fixed file, jobs **3889744** (`--shard 1/2`) /
  **3889745** (`--shard 2/2`), ~7h each → `runs/submit-cdsmall-qactx-s{1,2}of2.jsonl`.
- **still to do:** concatenate to one 10,999-row submission JSONL, verify all ids present and the
  `{"id","output"}` format, and decide the `<br>`-strip (26.7% of qa-oeg outputs carry literal
  `<br>` from the web-scraped substrate — the one-line submission-time strip flagged all along).

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

| | C+D | **C+D-small** | plain |
|---|---|---|---|
| bho-leaning / hin-leaning | **163 / 1** | 147 / 16 | 21 / 93 |
| % bho of the decidable | **99%** | **90%** | 18% |

D's effect is cleaner on qa-context (360 rows, 78% of bho's test share) than on qa-oeg (100
rows: 40% vs 12%), strengthening the case for keeping D. Caveat: only 39% of rows are long
enough to judge (the rest are 2–3 word noun phrases); this shows output that *looks*
Bhojpuri, not that it's *correct* — the test set has no gold to check against.

**C+D-small (job 3882159) added 2026-07-23.** It is the one place the shrunk pack reads
measurably weaker than the full one: 90% vs 99%, i.e. 16 hin-leaning answers where C+D had 1,
out of 163 decidable. Everything else on these rows is unchanged (Devanagari 99.7%, one
sentence 78.1%, refusal phrase 21.4% vs C+D's 19.7%, 0 empty). Whether 16 rows out of 360
matters is a judgement about 4.2% of the test set, not a measurement — but note the direction:
this is the only check where more bho data bought more bho output, which is what a genuine
dose–response looks like and is weak evidence that 17.1% is near the bottom of the usable range
rather than comfortably inside it.

🟢 **Submission-schedule correction**: qa-context runs at 5.9 s/row, not the 20.5 s/row
measured on qa-oeg — it only answers one sentence. Full qa set: 8,640×5.9 + 2,359×20.5 =
**27.5h**, not the earlier 67h estimate, so `--shard i/2` (13.8h each) is enough.

---

*This is co-authored by Claude.*
