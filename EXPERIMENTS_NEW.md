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

| job | config | qa-context: belebele-v2 / tydiqa-v2 / MCIF | qa-oeg: OEG / aya (0.87/0.13 agg) | notes |
|---|---|---|---|---|
| _none yet_ | | | | |
