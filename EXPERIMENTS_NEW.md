# Experiment log — v2 item-split dataset (from 2026-07-17)

Fresh log for everything run on **`data/train_v2.jsonl` / `data/dev_v2.jsonl`**
(`scripts/build_dataset.py`, seed 42 — build record in DATA_AUDIT.md §7, schema in
README "The rebuilt train/dev set"). The old log ([EXPERIMENTS.md](EXPERIMENTS.md)) is
closed: its dev split leaks parallel items across dev/train (DATA_AUDIT.md §2), so **no
old dev number is comparable to any number here** — do not copy rows across. Old verdicts
that survive without re-measurement (they compared like with like): everything
prompting-vs-prompting (shots, lang-hint ≈ free: 3859645 / 3866054), gold-vs-distilled on
a shared split (distillation lost: 3865036 + 3866054), adapter+demos don't stack
(train/infer format must match, 3858987), and all test-set facts (TEST_SET_ANALYSIS.md).
What does NOT survive: every adapter-vs-prompting *margin* — that is what the runs below
re-establish.

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

### 🔴 The runaway-generation artifact (found 2026-07-18) — and it retro-explains the old logs

The v2 adapter's collapse is **not** (primarily) about the v2 data: re-checking the OLD
adapters' prediction CSVs with the same `\nuser|\nassistant|<think>` detector shows the
gold adapter 3857589 already ran away on **78% of tydiqa and 56% of aya** rows (belebele
2.7%, MCIF 0%) — **the old "adapter collapses on tydiqa 38.94→19.53" finding was this
artifact, not a capability loss**. The distilled no-hint adapter 3865036 was much cleaner
(2–10%). Pattern: sources with SHORT free-form golds run away; templated (old belebele
"2: option") or long golds don't. v2 turned belebele into short free-form answers, which
is why the artifact went from "quarantined to tydiqa/aya" to "everywhere".

Mechanism checked (locally, Qwen3.5-9B tokenizer): `train_lora.py`'s label span is
correct — prefix is an exact token prefix, labels cover answer + `<|im_end|>` — so EOS
*is* trained; the fine-tuned model still under-samples it after short answers at
T=0.7/top-p 0.8. Sampled continuations reproduce the chat template as plain text and
often contain OTHER questions about the same passage. The truncated re-score below
measures the adapter with the artifact removed.

**Fix landed 2026-07-18** (`prompt_template.RUNAWAY_STOP_STRINGS` / `truncate_runaway()`):
benchmark.py and run_test.py now pass stop-strings to `generate()` (halts at the first
fake turn instead of burning the token budget) AND truncate every decoded prediction.
`truncate_runaway` is verified byte-identical to the 3869088 cleanup on all 2,949 rows;
base models are unaffected (0% incidence). Every adapter run from here on is clean by
construction — no post-hoc surgery needed again.

### The aya "style shift" read (2026-07-18) — mostly a chrF artifact, no action

906 aya dev rows joined across 3867141 (base) and the truncated 3867140 (adapter), plus
12 examples read side by side. Word-length p50: **gold 26 / adapter 11 / base 105**, and
**89% of base predictions carry markdown scaffolding** vs the adapter's 2.3%. That is the
whole chrF story: base's 4× verbose, headline-and-bullets answers soak up chrF's
character-recall; the adapter answers in the golds' own terse register (zho "答案：三星成立
于1938年。" vs gold "答案：三星公司成立于1938年。"), which chrF under-credits and
BERTScore/ROUGE-L correctly prefer. **Verdict: chrF-down/BERT-up on aya is the metric
disagreeing about verbosity, not degradation — no fix.** Two real residuals recorded, not
acted on: (a) occasional over-terseness on completion/open prompts (an arb passage
completion: 7 words vs gold's 60); (b) fabrication on knowledge-list prompts (a French
nouvelle-vague film list with wrong attributions/period). Both fold into the G-phase
variant discussion (human eval prefers complete-and-fluent), neither changes routing.
