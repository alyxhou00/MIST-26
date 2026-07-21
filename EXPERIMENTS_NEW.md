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
| 3869129 | 07-19 | **C+D SFT**: LoRA on `train_v2-cd.jsonl` (19,683 rows = 11,674 qa + 8,009 bho-pack; 840 qa-oeg rows carry a word budget), no hint, otherwise the 3867139 recipe | — | — | — | — | — | — | — | — | train_loss **1.156** (vs 3867139's 0.9656 — expected, the bho pack is 41% of the mix and is continuation/translation text), 11h23, 25 rows truncated \@2048. Adapter: `adapters/qwen3.5-9b-qa-lora-3869129`. |
| 3869130 | 07-19 | ↑ adapter, 0-shot, no hint (first run with the stop-fix in place — clean by construction) | 44.70 | 70.72 | **51.11** | 38.86 | **36.65** | 38.57 | — | — | 5h06, dev_v2 qa (n=2,949). **C+D does not beat plain-v2 on dev: qa-context is a wash (±0.4, noise), qa-oeg agg −1.42.** Deltas vs 3869088: belebele +0.37, tydiqa −1.32, MCIF +0.16, OEG **−1.67**, aya +0.22. See the C+D read below — dev cannot see either thing C+D adds. |
| 3876434 | 07-21 | **C-only SFT**: LoRA on `train_v2-c.jsonl` — the same 11,674 examples as 3867139, row for row, with 840 of them carrying a word budget and **0 bho rows**. Isolates C from D, which had only ever been trained together | — | — | — | — | — | — | — | — | train_loss **0.9648** vs 3867139's 0.9656 and 3869129's 1.156 — i.e. **the 840 budget sentences cost essentially nothing to fit, and the C+D loss increase was entirely the bho pack** (continuation/translation text at 41% of the mix). 6h27, 24 rows truncated \@2048. Adapter: `adapters/qwen3.5-9b-qa-lora-3876434`. Dev eval = 3878452, test compliance = 3878453. |

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

### C+D (3869130) — the dev verdict, and why dev cannot settle it (2026-07-19/20)

On dev, **C+D loses to plain-v2**: qa-context moves ±0.4 COMBINED (belebele +0.37, MCIF
+0.16, tydiqa −1.32 — inside noise for a 2,949-row split), and the qa-oeg aggregate drops
**−1.42** (OEG −1.67), which is the column C was aimed at. Taken alone that reads as a
straight rejection.

It is not, because **`dev_v2` cannot measure either thing C+D adds**, and this was checked
rather than assumed:

* **C — word budgets.** The budget text lives only in `data/train_v2-cd.jsonl`. Every
  `qa-oeg` prompt in `dev_v2` is plain — `constraint_bank.parse_budget` finds **0** rows
  stating a budget. The only prompts that state one are the official test's **465 of
  2,359** qa-oeg rows.
* **D — Bhojpuri.** `dev_v2` has **0 bho rows** (`question_lang`), by construction: the
  pack is training-only. The test set has 460 (qa-oeg 100, qa-context 360).

So dev measures only C+D's *cost* (a 41%-bho training mix diluting the qa head, plus a
higher train_loss) and none of its *benefit*. The benefit has to be measured on official
test prompts — jobs **3875151** (C+D) and **3875152** (plain-v2 control), `run_test.py
--task qa-oeg`, scored with `scripts/verify_outputs.py`.

#### Verification, complete — 2,359/2,359 rows, all 24 languages (2026-07-21)

Both jobs COMPLETED (13h25 / 13h27 wall, 2,359 rows each ≈ 20.5 s/row, confirming the
throughput warning below). The partial read from 07-20 held up: the language-ordering bias
moved C's absolute numbers up ~9pp for both systems but left the gap intact.

| check | C+D (3875151) | plain-v2 (3875152) |
|---|---|---|
| C: budget compliance, all 465 budget rows | **65.8%** | 44.9% |
| ↳ over budget / under budget | 7.1% / 27.1% | 3.9% / **51.2%** |
| ↳ mean overshoot beyond the band | 67 units | 86 units |
| ↳ worst languages | mar 25, ben 35, ckb 45 | ben 10, mar 20, tur 20 |
| D: bho_lid on the 100 bho rows — bho | **40%** | 12% |
| ↳ hin / npi / abstain | 36% / 1% / 23% | 67% / 0% / 21% |

Both effects are large and in the intended direction: **C lifts compliance by +20.9pp, and
the failure it removes is under-shooting** (answering far below the stated budget,
51.2% → 27.1%), not padding. **D triples the bho rate and halves the Hindi drift.**

**C is conditional, not a blanket length shift** — the worry was that training on 840
budgeted rows would make the adapter terser everywhere, which would cost recall on the
1,894 test rows that state *no* budget. Measured: on those non-budget rows the two systems
are near-identical (median length 73 vs 81, mean 128.3 vs 133.7), while on the 465 budget
rows C+D moves 119 → **149** — i.e. the length change is keyed to the instruction. So the
dev regression is *not* a length effect; it is the bho pack (41% of the mix) diluting the
qa head.

Two flaws quantified over the full 2,359 rows, **both present in the plain control**, so
neither is caused by C or D:

| flaw | C+D | plain-v2 |
|---|---|---|
| degenerate repetition (one clause ≥4×) | 52 rows (2.2%) | 57 rows (2.4%) |
| literal `<br>` markup in the output | 731 rows (**31.0%**) | 657 rows (27.9%) |

The `<br>` number is the surprise — **roughly 30% of every qa-oeg prediction carries HTML
markup**, on both adapters. It comes from the web-scraped qa substrate, and a one-line
strip at submission time is the cheapest point available anywhere on the roadmap.

**Read the LID numbers narrowly** — 12 outputs were read side by side before drawing
anything from them, and three things showed up that the aggregate hides:

1. **"hin" often means code-mixed, not Hindi.** Several C+D outputs labelled `hin` carry
   real bho lexicon (होला, एगो, बहुते, काहें) with Hindi copulas/inflection (है, करता)
   — the marker-density rule weights the copula heavily. The honest statement is "36%
   Hindi-leaning mixture", not "36% clean Hindi".
2. **Degenerate repetition on bho, in BOTH systems** — outputs that repeat one clause 8–10
   times ("बरखा की आवाज…", "हमनी के अलग-अलग टाइम जोन काहे होला" ×10). This is *not* the
   runaway artifact (no fake chat turns; the stop-fix is working) — it is ordinary
   degeneration on an out-of-distribution language. It also **inflates bho_lid**: a
   repeated bho clause gives a 0.38 marker density, i.e. one of plain-v2's `bho` labels is
   a repetition loop. Treat the +28pp as directionally right, magnitude soft.
3. **Some bho answers are 5–10 words** where the prompt asks for ~150 — the same
   over-terseness residual already logged on aya, worse here.

Side observation, both systems, not bho-specific: predictions contain literal `<br><br>`
markup. plain-v2 emits it too, so it comes from the qa substrate (web-scraped aya/oeg
text), not from the bho pack. Candidate for a cheap post-processing strip at submission
time — logged, not acted on.

**Status: C is keep, D is keep, but C+D as one adapter is probably not the primary.**

The full runs sharpen the trade rather than settling it, and they point at a recipe that
was never trained:

* **C earns its place.** +20.9pp compliance on 465 rows, no spillover onto the other 1,894.
  There is no measured cost to C anywhere.
* **D earns its place too**, on 460 test rows dev cannot see. Wrong-language output scores
  near-zero on chrF, so lifting bho 12% → 40% plausibly moves the bho column by more than
  the −1.67 C+D loses on OEG overall — bho is 1 of 24 languages, so the two are the same
  order of magnitude. This is why dev's −1.42 does not settle it.
* **The −1.42 is attributable to the bho pack's 41% share of the mix**, not to C (proven
  above: C's length effect is conditional on the instruction).

So the missing datapoint is **C-only** — `augment_constraints.py` without `--append-bho`,
i.e. `train_v2-c.jsonl`. Built and trained as **job 3876434** ✅ (2026-07-21): 18,901 file
rows → **11,674 training examples, row-for-row identical to 3867139's**, with 840 of them
carrying a budget sentence and 0 bho rows. That makes it a clean single-variable test of C.

**Decision rule**: if C-only recovers plain-v2's dev score *and* keeps the ~65% compliance,
it is the primary and C+D becomes the bho-hedging variant. If C-only's dev also drops, then
the −1.42 was not the bho pack's fault and plain-v2 is the primary instead.

**First signal, from training loss alone (2026-07-21): 0.9648, against plain-v2's 0.9656 and
C+D's 1.156.** A difference of 0.0008 on the same 11,674 examples says the 840 budget
sentences are free to fit and the whole of C+D's loss increase came from the bho pack. That
is consistent with dilution being the cause of the −1.42 — but it is not the answer:
train_loss is measured on the training distribution, and C could still change generation
behaviour on dev without being harder to fit. Jobs **3878452** (dev eval) and **3878453**
(test compliance) settle it.

**C-only is a diagnostic, not a default primary.** It separates a cause that has never been
measured apart — C and D went into the same SFT — but winning on dev would *not* by itself
make it the right submission, because it puts bho back to 12% and dev is structurally blind
to that. Choosing on dev alone is the exact mistake this whole round exists to avoid.

#### The proportionate D — `sft-bho-small.jsonl` (built 2026-07-21, not yet trained)

The suspicion is about **proportion**, not about whether D belongs. The pack is **40.7% of
the training mix** (8,009 of 19,683) buying **4.2% of the test set** (460 of 10,999 qa
rows) — a ~10× over-allocation whose cost lands on the 96% of rows that are not bho. So
`scripts/shrink_bho_pack.py` builds a 2,400-row pack → `data/train_v2-cd-small.jsonl`,
**14,074 training examples with bho at 17.1%**, the same 840 budget rows, so it stays a
single-variable comparison against 3869129.

**The row selection is the opposite of the obvious one, and the data is why.** The instinct
was to keep the xP3x half (clean parallel supervision) and drop fineweb (raw web scrape).
Measured:

| source | n | median | p10 | p90 |
|---|---|---|---|---|
| `HuggingFaceFW/fineweb-2:bho_Deva` (continuation) | 6,000 | **120 words** | 55 | 237 |
| `CohereLabs/xP3x:bho_Deva` (hin→bho) | 2,009 | **24 words** | 15 | 37 |

qa-oeg wants ~150-word answers, and one of the three hand-read flaws was **bho answers of
5–10 words**. fineweb is the only half that demonstrates paragraph-length bho at all — an
xP3x-only pack would have trained the terseness in. xP3x earns its place on the *other*
flaw: it is hin→bho parallel text, a direct demonstration of not falling back to Hindi.
Final split: **1,400 fineweb** (sampled toward the 80–250 word band) + **1,000 xP3x**.

Held for job (a) of the decision tree in ROADMAP #1 — it only gets submitted if 3876434
shows the dilution is real.

### D on qa-context — jobs 3876525 / 3876526, and an aggregate that lied (2026-07-21)

Every D number so far came from the 100 bho **qa-oeg** rows. bho has **460** qa rows in the
test set; the other **360 are qa-context** and had never been generated. Two 30-minute jobs
fixed that (`run_test.py --task qa-context --lang bho`, C+D 3869129 vs plain 3867139).

Reading those rows first changed the instrument. They are 97% cross-lingual (passage in
eng ×100, arb/spa/zho ×25 each, …; only 10 rows have a bho passage), the prompt asks for
**one sentence**, and answers come out at a **median of 3 words**. `bho_lid` says in its own
docstring that it is not to be trusted on single short sentences, so it is now filtered to
qa-oeg rather than run here and quietly believed.

**The first three checks said the two systems were identical:**

| check | C+D (3876525) | plain (3876526) |
|---|---|---|
| Devanagari script | 358/360 (99.4%) | 358/360 (99.4%) |
| one sentence | ~99% | ~99% |
| attested refusal phrase | 19.7% | 17.2% |

Read literally, that is "D does nothing on 78% of the bho test rows". **It is wrong.**
Hand-reading pairs on the same ids showed a consistent difference the aggregate could not
see — C+D writes `आ` / `खातिर` / `होखल`, plain writes `और` / `के लिए` / `की`:

    [qa-context_18_bho_spa] C+D  : संरक्षण खातिर पूर्ण रुप से पर्याप्त साधन ना होखल
                            plain: संरक्षण के लिए परिपूर्ण साधनों की कमी

So the fourth check: **contrastive function words** — "which of these two words did it
pick", which one word of output can answer, instead of `bho_lid`'s marker *density*, which
needs a paragraph.

| | C+D | plain |
|---|---|---|
| bho-leaning / hin-leaning | **163 / 1** | 21 / 93 |
| undecidable (too short to commit) | 125 | 184 |
| **% bho of the decidable** | **99%** | **18%** |
| `आ` vs `और` alone | 82 vs **0** | 5 vs **67** |

**D's benefit now rests on 460 rows, not 100, and the qa-context effect is cleaner than the
qa-oeg one** (40% vs 12% there). That strengthens the case for keeping D in the primary and
supports ROADMAP #1's rule that C-only winning on dev would still not settle it.

Two honest limits: only 39% of rows are decidable at all — the rest are 2–3 word noun
phrases lifted from the passage, where language is undecidable in principle; and this shows
the output *looks* Bhojpuri, not that it is *correct*, which no test-set check can show
without gold.

🟢 **Side result that changes the submission schedule: qa-context runs at 5.9 s/row, not
20.5.** The 67h projection for the full qa set came from extrapolating the qa-oeg rate to
everything, but qa-context is 8,640 of the 10,999 rows (79%) and answers one sentence
instead of 512 tokens. Recomputed: 8,640×5.9 + 2,359×20.5 = **27.5h**, so `--shard i/2`
(13.8h each) is enough and n=4–6 was over-planning. Measured on bho only; the one-sentence
instruction is universal, so it should carry, but worth re-checking before the final runs.
