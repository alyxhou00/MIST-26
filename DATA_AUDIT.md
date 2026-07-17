# Sample-data audit — the full picture before the train/dev rebuild

**2026-07-17.** Full-enumeration audit of `pinzhenchen/wmt26-mist-sample` (23,919 rows,
downloaded fresh from HF) against `data/tests.jsonl` (12,775 rows, revision `5950311`),
run locally; scripts produced every number below by iterating over **all** rows, no sampling
(the read-the-data-before-concluding rule this repo has paid for twice). Motivation: the planned rebuild of the train/dev set into
test-set format needs a verified baseline of what we actually have — and the audit found
**dev/train leakage** that changes how the old dev numbers should be read.

Cross-references: test-set facts in TEST_SET_ANALYSIS.md; per-job results in EXPERIMENTS.md;
the sub-task mapping table in README.md (verified exact by this audit: 5700/3112/880 +
7026/1600/400 + 4741/460, qa=14,893, sum=9,026). The dev split reproduces exactly:
shuffle(seed 42) → first 2,978 qa rows = belebele 1123 / tydiqa 615 / MCIF 165 / aya 978 /
OEG 97 — the provenance of every dev number in EXPERIMENTS.md.

## 1. What each qa source actually is (all-rows enumeration)

| | belebele | tydiqa | MCIF (QA) | aya | OEG |
|---|---|---|---|---|---|
| n (train+dev) | 5,700 | 3,112 | 880 | 4,741 | 460 |
| languages | 19 × 300 (**no eng**) | 11 × ~300 | 4 × 220 (deu/eng/ita/zho) | 16 × ~300 | 10 × 46 |
| input shape | passage + question + **4 numbered options** | passage + question | **instruction + question** (in `lang_code`) + full English talk transcript | bare question/request | bare request |
| input words p50 | 103 | 57 | **776** | 11 | 18 |
| gold shape | `"<digit>: <option text>"` (option text always verbatim in input) | extraction, p50 **2 words** | sentence, p50 7 words | p50 24 words | p50 153 words (dev slice: 175) |
| embedded instructions | none | none | **yes — every row** ("Answer the following question concisely given the English content:" / "Beantworte … kurz und bündig basierend auf dem englischen Inhalt:" / "根据所给的英文内容，简要回答以下问题：") | none | the request itself, but **no word budgets** |
| cross-lingual | no | no (passage=question=answer language) | **yes for deu/ita/zho (75%); eng quarter is eng→eng monolingual** | ~50–60 rows (e.g. English question → Hindi gold) | no |
| parallel corpus | yes upstream (item sets only partially overlap across langs in our sample — see §4) | no (natively authored per language; 157/2,919 passages reused within-language) | 21 talks × ~10 questions × 4 langs | no (but 169 duplicated inputs, same-language) | **yes — 46 prompts × 10 langs** (verified: same invention-prompt found in ces/deu/eng/ind/rus…) |

Corrections to earlier docs this table forced:

- ❌ **"tydiqa = Arabic passage+question+answer" was false.** tydiqa is **11 languages**
  (dev: rus 65, fin 65, tel 63, arb 61, ind 60, tha 59, kor 56, eng 56, swh 55, jpn 49,
  ben 26); Arabic is one of them. The *monolingual* characterization — the load-bearing
  part — survives. Note 900 rows (tel/swh/tha) target languages that appear **nowhere**
  in the test set, not even as passage languages.
- ⚠️ **MCIF is only ~73–75% cross-lingual** — its eng quarter (dev: 44/165) is eng→eng,
  as monolingual as tydiqa. "MCIF = the cross-lingual proxy" remains the right routing
  call but the column is not pure.
- ⚠️ **MCIF's shape is testlike in *instructions* but not in *length***: its context is a
  full ~776-word talk; test `qa-context` passages are p50 **63 words** (p90 139). And its
  question languages are only {deu, eng, ita, zho} vs the test's 24, over English-only
  content vs the test's 25 context languages.
- **MCIF has two portions and they share source material**: the 880-row QA portion
  (task=`qa`) uses **21 talks**; the 400-row summarization portion (task=`sum`,
  teammate's) uses 100 talks — **all 21 QA talks are among them**. Anywhere a doc says
  "MCIF" as a qa proxy it means the QA portion. The sum portion's instructions carry
  word budgets ("… Abstract mit ungefähr 200 Wörtern …") — the same style as test
  `sum-sum` ("Write the paper's abstract in around 200 words").
- **tydiqa in the sample has no unanswerable rows** (top golds are years/names; no
  refusal strings). The test's '"no answer" if not in passage' escape has **zero
  training signal** in the sample — relevant to the observed false-refusal failures;
  upstream `copenlu/answerable_tydiqa` does carry unanswerable rows if we want them.

## 2. 🔴 The headline: the 80/20 row split leaks parallel items across dev/train

`benchmark.py` (and `train_lora.py`) split **by row**. Four of five qa sources carry
near-duplicate rows — the same item in another language, or verbatim — so a row-level
split puts copies of dev items into the training data:

| source | leakage mechanism | measured |
|---|---|---|
| MCIF | same 21 talks, ~10 parallel questions × 4 langs | **21 of 21 dev talks also in train** — the German twin of a dev-Chinese question about the same talk is trained on |
| OEG | 46 prompts × 10 langs | 5 of 7 multi-language numeric signatures span dev *and* train (46×10 structure ⇒ nearly every dev prompt has train twins) |
| belebele | parallel MC items across (a subset of) its 19 langs | ≥260 numeric-signature item groups span dev *and* train (lower bound — non-ASCII digit localization hides matches) |
| aya | verbatim duplicate inputs, same language | 49 duplicated inputs have copies in both dev and train (159 of 169 duplicated inputs carry *different* golds — multi-reference) |
| tydiqa | multiple questions per passage | 157/2,919 passages reused (minor) |

**Consequence for old numbers:** any comparison where one side *trained on the train
split* and the other didn't is biased **toward the adapter** — the gold-LoRA dev scores
on MCIF/OEG/belebele/aya were measured on items whose cross-lingual twins were in its
training data. Comparisons where both sides trained on the same split (gold vs
distilled), or neither trained (2B/9B, shots, lang-hint A/Bs incl. 3866054), are
unaffected. A leakage-free re-score of the old CSVs is **not** possible for MCIF — all
21 dev talks leak, an item-level filter leaves n=0. The fix is the item-level split in
the rebuilt dataset.

## 3. The variables (變因) across everything run so far

Data-side (fixed per source, enumerated in §1): source; task; `lang_code` — **the answer
language** (output script matches it ≥95% net of short-string artifacts; the handful of
aya rows with an English question and e.g. a Hindi gold keep lang_code = gold language);
mono/cross-lingual; parallelism; gold length regime (2w → 4w → 7w → 24w → 153w by
source); gold provenance (annotators / humans / GPT-4.1); embedded instructions (MCIF
only); MC format (belebele only).

System-side (what EXPERIMENTS.md varies): model size (2B/9B); shots (0/3); lang-hint
(on/off — ~free on the sources that matter, base 3859645 and adapter 3866054); adapter
(none / gold 3822375 / distilled 3864945); training format (own template vs test format);
teacher (35B/122B) + filter policy (30/70, gold-only belebele, `--prefer 122b`);
sampling (T=0.7, top-p 0.8, Qwen card); data revision (`2dcf223` → `5950311`);
`--unescape` (built, untested); seed 42 everywhere.

Eval-side: metric rule (COMBINED = mean(chrF, BERTScore, ROUGE-L); legacy overall is 71%
noise); proxy split (MCIF-only for qa-context; OEG/aya as separate qa-oeg ends); EM/F1
still pooled at TASK_PROXY level (known stale).

## 4. Dev format vs test format, side by side

**qa-context (test, 8,640 rows):** preamble + **63-word passage** (25 possible langs) +
question (24 langs; 96% ≠ passage lang) + constraint tail (one sentence, passage-only,
"no answer" escape) + "Answer in X." — boilerplate all in the question language, and a
literal `\n` escaping bug on every row. **No sample source matches this**: belebele has
the right passage length but MC format, no instructions, monolingual; tydiqa right
length, no instructions, monolingual, extraction-length golds; MCIF right instructions
and cross-linguality but 776-word contexts, 4 question langs, English-only content.

**qa-oeg (test, 2,359 rows = 100 parallel prompts × 24 langs):** bare self-contained
request, p50 16 words; 20/100 prompts carry word budgets; ~13% short-answer (aya's
shape), ~87% long-form (OEG's shape). OEG (46 parallel prompts × 10 langs, GPT-4.1
golds) is the same genre without budgets; aya is natural same-language Q→A, p50 24w.

**sum-sum (test, 1,776 rows, teammate):** full ACL paper (p50 **2,773 words**) + "abstract
in around 200 words". MCIF-sum is the closest cousin (766-word talk → 166-word abstract,
budgeted instruction); CrossSum is 373w → **21w** one-liners, ~80% cross-lingual;
wiki_lingua 949w → 82w, monolingual. Note 2,773-word inputs exceed the 2,048-token SFT
truncation currently in `train_lora.py`.

## 5. Language coverage (union of qa sources vs the 24 test question languages)

Rows per test question language (qa sources only): arb/ind/jpn/rus 946 · kor 900 ·
zho 866 · ita 820 · deu 807 · hin 646 · eng 566 · mar/por/spa/tur/vie 600 · ckb/hat/pes/
slk/fin/yor 300 · ben 158 · **ces 46 · bho 0**.

Single-source languages: **ckb, hat, pes, slk exist only in belebele** (the MC source
whose format doesn't transfer — reformatting belebele is the *only* way these four test
languages get any qa training data); **fin only in tydiqa**; **yor only in aya**;
**ces only in OEG (46 rows)**; bho only via roadmap D's external pack. fra exists in
aya+belebele (600 rows) but is a **passage-only** test language — usable as context
material, wrong as an answer language.

## 6. What this means for the rebuild (decision input, not decisions)

1. **Split by item, not by row** — group key: talk id (MCIF), prompt identity (OEG,
   via upstream ids or translation-clusters), item id (belebele, ideally re-pulled from
   upstream where alignment is explicit and eng exists), passage (tydiqa), exact-dup
   cluster (aya). This single change is what makes the new dev honest.
2. **The proposed schema** (task / question_lang / context_lang / source / input /
   output) fits every source, with three footnotes: context_lang is null for qa-oeg
   sources; the ~50 cross-lingual aya rows violate the test invariant answer-lang =
   question-lang (drop or relabel); and `lang_code` today means *answer* language, which
   under the invariant becomes question_lang — the rename is safe except for those rows.
3. **Reformatting belebele to test qa-context format is mechanical**: the gold option
   text is verbatim in the input on 5,700/5,700 rows — drop the options, keep the
   passage+question, take the option text as the gold, wrap in constraint_bank tails.
   Cross-lingual pairs (passage lang ≠ question lang) need upstream belebele for exact
   item alignment; our sample's per-language item sets only partially overlap.
4. **The "no answer" escape needs data**: sample tydiqa is all-answerable; pull
   unanswerable rows from upstream if the escape is to be trained rather than prompted.
5. **Previous experiments**: verdicts that survive as-is — everything prompting-only,
   both lang-hint A/Bs, gold-vs-distilled (shared split), adapter+demos-don't-stack, all
   test-set facts, teacher-quality findings. Read with a leakage asterisk — the *margins*
   of adapter over prompting on MCIF/OEG/belebele/aya dev columns (direction likely
   survives; the MCIF sweep is 4-metrics-unanimous and EM 36×, hard to explain by
   leakage alone, but the honest number awaits the item-split dev). Numbers that carry
   over to the new dev: none — new dev = new baseline runs (hence EXPERIMENTS_NEW.md).

## 7. The rebuild (2026-07-17) — what was actually built

`scripts/build_dataset.py` (seed 42, deterministic) → `data/train_v2.jsonl` **18,901** +
`data/dev_v2.jsonl` **4,748** (dev fraction 0.201 of item groups). Schema: task /
question_lang / context_lang / source / input / output / **item_group** (the split is a pure
function of item_group; C-augmented variants must inherit it). User decisions folded in:
pull upstream belebele + answerable_tydiqa **yes**; drop tydiqa tel/swh/tha and the
script-detectable cross-lingual aya rows **yes** (38 found: hin 36, kor 1, mar 1 — the
Latin-target residue is undetectable without LID and stays); keep belebele monolingual rows
(4%, the test's own 340/8,640); dev = 20% of item groups.

Facts established during the build (beyond §1–§6):

- **Upstream belebele is fully parallel**: 900 items × all 24 needed languages keyed by
  (flores URL, question_number) — verified identical key sets for eng/zho/ckb. **No
  `bho_Deva`** in belebele's 122 languages: bho stays covered only by roadmap D's pack.
- **32% of belebele questions reference the MC options** ("which of the following …",
  282/900 items on the parallel English question) — unanswerable once the options are
  dropped (all the "would NOT be…" ones) and phrased like nothing in the test set. Dropped
  at the item level; 618 items kept × 10 sampled question langs ≈ 270 rows/lang.
- **Context languages are sampled from the test's own qa-context marginal** (eng 27.8%,
  arb/spa/zho 6.9% each, …, measured from tests.jsonl ids; bho excluded, unavailable), 4%
  monolingual. 7% of belebele rows are made unanswerable by a same-side passage swap; gold
  = the attested per-language refusal phrase.
- **`copenlu/answerable_tydiqa` upstream is 50/50 answerable/unanswerable in every
  language**, `document_plaintext` is already passage-sized (p50 48 words), and
  `answer_start` is exact (500/500 sampled). v2 takes 240 answerable + 60 unanswerable
  per kept language (8 langs), passages filtered to 20–200 words (char band for jpn),
  item groups = union-find over (question, document URL).
- **OEG's per-language row order is shuffled** — index-alignment across languages is
  wrong (old dev's implicit assumption; e.g. sample row 30 is "holiday" in eng, "school
  system" in deu, "time zones" in zho). The 46×10 parallel structure was aligned by
  reading all 460 prompts (`scripts/oeg_alignment.py`, permutation-asserted, selftest on
  the two budgets that survive localization: 300-word article, 200-word job posting).
  Prompts are *localized*, not just translated (Vltava/Volga/Citarum/鴨川 for the river
  item; 13-line poem is 10 verses in ces, 50-word story is 125 chars in jpn).
- **MCIF item groups span both portions**: group = talk; the 21 QA talks sit inside the
  100 sum talks, so a dev talk is dev for qa-context *and* sum-sum (4 QA + 16 sum-only
  talks in dev).
- **CrossSum's context_lang is null**: the sample doesn't record the article language and
  article-hash grouping cannot catch translated-parallel articles across languages — if
  the sum split ever becomes load-bearing, re-pull the upstream pairing metadata
  (teammate's call).
- **Post-build audit (independent, from the output files): all green** — no item_group in
  both files; no dev synthesized passage/question string anywhere in train; no dev MCIF
  talk in train; no verbatim dev input in train; every synthesized row carries the exact
  attested boilerplate (4 literal-`\n\n` segments). Two informational findings, both
  verified benign: 171 monolingual belebele golds are passage *paraphrases* (belebele
  options paraphrase; §3's "verbatim in input" was about the old options-in-input format),
  and 42 synthesized golds fail a script heuristic because they are Latin proper
  nouns/acronyms (ASUS, Deutsche Bank, …).
- **Dev composition**: qa-context 1,915 (23 q-langs, 70% cross-lingual) + qa-oeg 1,034
  (19 q-langs) + sum-sum 1,799. ⚠️ qa-oeg dev is still aya-heavy (944 aya vs 90 OEG rows
  for a test that is ~87% OEG-shaped) — **weight at scoring time** (0.87·OEG + 0.13·aya),
  don't average the pool; OEG has only 46 items and giving dev more of them would starve
  train of the only long-form source.
