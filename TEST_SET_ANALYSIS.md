# Official test set — analysis and strategic implications

Source: `pinzhenchen/wmt26-mist-test` (single `tests.jsonl`, 47MB), analyzed 2026-07-15
together with the first qualitative smoke run (job 3859059, `slurm/smoke-run-test.sbatch`;
outputs kept in `predictions/smoke-test-*-3859059.jsonl`).

**Revision — re-downloaded 2026-07-16, HF commit `5950311` "include eng prompts"
(2026-07-15T15:03Z).** The copy this file was originally written against was `2dcf223`
(14 July + yoruba). The organizers fixed the 100 empty English `qa-oeg` prompts on their own;
no report from us was needed. Diffed old vs new: **exactly the 100 `qa-oeg_*_eng_eng` rows
changed, `prompt` field only — no rows added or removed, and no other row touched anywhere in
the file.** So every count below that does not involve English `qa-oeg` still stands as
measured; the ones that did are re-derived and marked in place.
Data-level facts are summarized in the README's "The official test set" section; this file
is the full analysis and what it means for the plan. Submission deadline: **1 Aug 2026 AoE**.

## 1. Composition

| task | rows | notes |
|---|---|---|
| `qa-context` | 8,640 | context-based QA — ours |
| `qa-oeg` | 2,359 | open-ended generation — ours |
| `sum-sum` | 1,776 | summarization — teammate's |
| total | 12,775 | |

- 460 qa rows per language × 24 languages (Yoruba 419). `question_lang` is a bare
  3-letter code (`zho`, `bho`, ...), unlike the sample data's `lang_code` with script
  suffix (`zho_Hans`) — `prompt_template.TEST_LANG_NAMES` maps them.
- Languages = 22 from the training data + English + **one surprise: `bho` (Bhojpuri)**.
  The pre-release FAQ hinted at possibly more surprises; **the final file contains exactly
  one**, so a generic new-language pipeline is nice-to-have, not required — the Bhojpuri
  emergency kit (roadmap D) is what matters.
- fra/swh/tel/tha from the sample data are **absent** — stop optimizing for them.

## 2. Format

- Each row: `{id, prompt, task, question_lang}`. `id` = `task_int_qlang_ctxlang`
  (e.g. `qa-context_2_ita_spa`: Italian question over a Spanish context).
- Prompts are **self-contained conversational prose**, e.g. (qa-context, English over
  Arabic context): *"I have a question about this passage: … The question is: … Could you
  please answer in one sentence, using only what the passage says? …"* — a different
  distribution from our templated `context + question` sample inputs.
- **All 8,640 `qa-context` prompts contain LITERAL `\n\n`** — the two characters `\` and
  `n`, not a newline. The organizers' file is double-escaped: the JSON holds `"...\\n\\n..."`,
  so `json.loads` yields a backslash-n that survives into the prompt string. Real newlines
  appear in exactly 2 qa rows (both `qa-oeg`).
  - *This corrects an earlier claim here* ("only 2 rows contain a literal `\n` escape; no
    unescaping needed"), which came from testing `'\n' in prompt` — that finds real
    newlines, not the escape. Verified by counting `chr(92)+'n'`: 8,640/8,640 qa-context,
    0/2,359 qa-oeg (re-checked on `5950311`; the 100 restored English prompts carry neither
    a literal escape nor a real newline, so the qa-oeg denominator is now the full 2,359).
  - `qa-context` layout is `<lead-in>:\n\n<passage>\n\nThe question is: <q>\n\n<instructions>`,
    so the literal escapes sit at exactly the structural boundaries — the passage/question
    separators. Feeding the prompt verbatim (what `run_test.py` does by default) shows the
    model `\n\n` as text at every boundary, in **79% of our qa test rows**.
  - `run_test.py --unescape` turns them back into real newlines. Not the default: it changes
    the official input, and we have no dev proxy to A/B it on (the sample data has no such
    escapes). Worth a qualitative smoke and a candidate axis for the variant submissions.
  - Useful side effect: the literal `\n\n` is a reliable section delimiter — `constraint_bank.py`
    splits on it to lift the per-language instruction tails.
- qa prompt lengths: median 654 chars, p95 1,475, max 2,607 — no long-context pressure.

## 3. Cross-lingual is the norm, not the edge case

**8,300 of 10,999 qa rows (75%) have context in a different language than the question**
(`qa-context`: 8,300/8,640 cross vs 340 same; `qa-oeg` is always same-language). In the
sample data, cross-lingual pairs were a minority (aya's English-question/local-answer rows).
Dev results therefore *understate* how much test performance depends on answering in the
question's language while reading a foreign-language (often English) context.

## 4. Embedded instructions are part of the task

Every inspected qa prompt carries explicit instructions **inside the prompt text**:

- `qa-context`: an output-format constraint ("answer in one sentence", "List …" — 360/360
  English rows match a constraint pattern), a grounding constraint ("using only what the
  passage says"), an explicit **"no answer" escape** (*"if the answer is not in the
  passage, write only 'no answer'"* — so unanswerable detection is scored, tydiqa-style),
  and a closing **"Answer in \<language\>."**
- `qa-oeg`: word budgets ("in 120–150 words", "150 शब्दन में") appear in every language
  **including Bhojpuri** — but they are **not** on every prompt: exactly **20 of the 100
  unique qa-oeg prompts per language** carry a numeric budget, because qa-oeg is a
  **parallel corpus** (the same 100 prompts translated 24 ways, yoruba 59). An earlier
  version of this doc called budgets "routine", which reads as "on every prompt"; the real
  scale is ~1/5 of qa-oeg. Roadmap C still stands, just at that scale.
  - *Corrected 2026-07-16 (was "21 of 100 … 471/2,359 rows; identical in all 23 non-empty
    languages").* The 21st was prompt **#13**, *"Explain the theory of relativity using only
    haiku format (5-7-5 syllables)"* — digits next to a unit word, but a **format** constraint,
    not a length budget. The restored English block settles it: English is the source language
    and yields **20/100**, on exactly the same prompt indices as Spanish
    (`1,2,3,4,8,9,46,47,48,71–74,76–80,84,98`). This also squares the doc with
    `constraint_bank.parse_budget`, whose own accuracy note said 20/100 all along.
  - Counting *rows* is the wrong unit anyway and is why the old figure looked authoritative:
    per-language parser residuals (bho/rus over-match #13, jpn 19, zho 16, yor 8/59 — see
    `parse_budget`'s docstring) make the row total an artifact of parser accuracy, not of the
    data. The data-level fact is **20/100 prompts**; the parser now finds 465/2,359 rows
    (= the pre-fix 445/2,259 plus English's 20), still ±5% until those residuals are chased.

Implications: (a) the test prompt already tells the model the output language, so our
lang-hint system turn is redundant-at-best there — **measured (job 3859645): dropping it
costs 25.97 vs 27.64 overall, but the loss is almost entirely belebele (52.70→32.42), which
does not transfer; on the sources the test set actually has, the cost is under 1 chrF and
OEG is +0.09. The test format can be used as-is** (and job 3866054 later confirmed this on the
gold **adapter** too, not just the base — ≤0.6 COMBINED on every routing column, so the whole
routing table is comparable to `run_test.py`'s no-hint inference); (b) instruction-following —
especially non-English length control — is directly scored, validating roadmap C.

## 5. No multiple-choice anywhere

Zero qa prompts match MC patterns (`(A)`, `A. … B. …`). The belebele-style format that
dominates our dev set (1,123/2,978 rows) and where the gold-SFT adapter got its biggest
win (chrF 52.70 → 85.82) **does not appear at test time**. Dev overall chrF is **not** a
faithful test predictor. ⚠️ This section used to continue: *"Free-form extraction (tydiqa-like)
is the closest dev proxy for `qa-context` — and that is exactly where the adapter collapsed
(38.94 → 19.53)."* **Retracted 2026-07-16** — tydiqa is monolingual and the test sub-task is 96%
cross-lingual, so it proxies ~4% of it; the collapse was on the wrong task. See §5b.

## 5b. What each dev source proxies (rewritten 2026-07-16 — the first version was wrong)

**Retracted:** an earlier version of this section claimed aya "resembles neither test task"
and that 71% of dev is noise. **That was wrong, and wrong by exactly the method this document
keeps warning about.** The reasoning was: ~20% of qa-oeg prompts carry a 120–180 word budget →
OEG's 175-word golds match it → aya's 24-word golds are 7× too short → aya proxies nothing.
Every step is true *of the budgeted 20%*, and it was generalised to all of qa-oeg without
reading the rest.

`qa-oeg` is only **100 unique prompts** (a parallel corpus, translated 24 ways) — small enough
to enumerate, which is what should have happened first. Reading all 100, the task is a
**spectrum**, not one regime:

> Enumerated from the **Spanish** rendering, because English was empty at the time (the old §6
> bug). Since `5950311` the **English source text is available** and is what the examples below
> now quote. The re-download did not move the shape of this table: English is index-aligned
> with Spanish (verified on the budget set, §4), so the Spanish reading was sound — but English
> is the original, so prefer it for any further enumeration.

| kind | count | example | answer length |
|---|---|---|---|
| explicit word budget | ~20 | #1 "Write a 150-word description of a technological development…"; #48 "a 200–300 word essay" | 120–300 words |
| open-ended creative / explanatory | ~65 | #5 "Explain quantum entanglement using only kitchen metaphors"; #22 "Invent a new holiday"; #37 "Write a poem using only words starting with S" | medium, unbounded |
| **short answer / list / trivia** | **~13** | **#90 "Can you name a country whose name has no a, e, i, o, u?"; #100 "Name the top 5 landmarks in the capital"; #92 "…give a single line for dress, gifts, socialising"; #94 "one sentence of explanation, then exactly two examples"** | **short — this is aya's shape** |

So **aya does proxy `qa-oeg`'s short-answer tail** (~13% of the prompts ≈ 307 test rows), and
OEG proxies the long-form end. They are proxies for *different slices of the same task*, which
is what README's grouping said all along.

**Second retraction (2026-07-16): `tydiqa` does not proxy `qa-context` either** — same method,
same mistake, third time. Test `qa-context` is **96% cross-lingual** (passage in one language,
question in another; §5c). tydiqa is **monolingual** — passage, question and answer all in one
language (11 languages; "Arabic" as previously written here was one of them, not all of it —
DATA_AUDIT.md §1) — so it stands in for roughly the 4% of the sub-task that isn't cross-lingual, while
supplying 79% of the pooled proxy rows. **MCIF is the only cross-lingual QA source we have**,
and it is n=165. Consequence: the "adapter collapses on tydiqa" result that drove routing for
days was measuring the wrong task, and on MCIF the adapter in fact wins every metric
(EXPERIMENTS.md). Gold length differs too, which is why EM was misleading: tydiqa's golds are
63% 1–2 words, MCIF's are median 6 with 42% at 8+.

| dev source | n | gold words p50 | cross-lingual? | proxies |
|---|---|---|---|---|
| `FBK-MT/MCIF` (QA) | 165 | 6 | ✅ 73% (its eng→eng quarter is monolingual) | ✅ **`qa-context` — the only faithful proxy** |
| `wmt25-mist-oeg-gpt-4.1` | 97 | 175 | — | `qa-oeg`, long-form end (~87% of prompts) |
| `CohereLabs/aya_dataset` | 978 | 24 | — | `qa-oeg`, short-answer end (~13% of prompts) |
| `copenlu/answerable_tydiqa` | 615 | 2 | ❌ **no — monolingual** | ❌ **nothing** — ~4% of a sub-task that is 96% cross-lingual (retracted 2026-07-16) |
| `facebook/belebele` | 1,123 | — | — | **nothing** — multiple choice, and the test set has none (§5) |

**Usable dev is therefore 1,240/2,978 rows (42%)**, and the `qa-context` half of it is 165 rows.

**What survives:** belebele (1,123 rows, 38% of dev) really does predict nothing — that rests
on "zero qa prompts match MC patterns", which is a property of the whole file, not a subset.
**What does not:** the "71% noise" figure and aya's exclusion.

⚠️ **The real distortion is weighting, not validity.** dev gives aya 978 rows for ~13% of
qa-oeg and OEG 97 rows for ~87% — inverted. So aya's numbers are over-weighted for the test
mix and OEG's are badly under-powered, but both are measuring something real. Do not average
them; read them as two separate columns.

## 5c. What each teacher run actually covers

Train-split row counts behind the distillation pipeline (README sub-task table minus the
2,978 dev rows), which decide what can be built before the 35B shards land:

| source | train rows | teacher that generated them | serves test task |
|---|---|---|---|
| `CohereLabs/aya_dataset` | 3,763 | 122B (job 3859682) | — (see §5b) |
| `wmt25-mist-oeg-gpt-4.1` | 363 | 122B (job 3859682) | `qa-oeg` |
| `facebook/belebele` | 4,577 | 35B (3859277-79) | — (no MC at test) |
| `copenlu/answerable_tydiqa` | 2,497 | 35B (3859277-79) | `qa-context` |
| `FBK-MT/MCIF` (QA) | 715 | 35B (3859277-79) | `qa-context` |

Total 11,915. Note the shape: the 122B run is **91% aya**, so it does *not* amount to a
ready-made `qa-oeg` training set; and 4,577 of the 35B shards' 7,789 unique rows are
belebele, whose format does not transfer.

## 6. Data bug: the English qa-oeg block

**Empty prompts — FIXED upstream 2026-07-16.** `qa-oeg_1..100_eng_eng` (the entire English
qa-oeg block) had `prompt == ""` in `2dcf223`. HF commit `5950311` ("include eng prompts")
fills all 100; nothing else in the file changed (see the header). We never had to report it.
`run_test.py`'s empty-prompt guard is kept as a safety net but is now dead code on this
revision — 0/12,775 rows have an empty prompt.

**Still broken, narrower: 8 rows ship unsubstituted template placeholders.** Found by reading
the restored block rather than assuming the fix was complete:

| row | placeholder |
|---|---|
| `qa-oeg_93_eng_eng`, `qa-oeg_94_eng_eng`, `qa-oeg_99_eng_eng` | `{language}` (99 has it twice) |
| `qa-oeg_95..98_eng_eng`, `qa-oeg_100_eng_eng` | `{country}` |

e.g. #95 English: *"What is considered the national sport in {country}?"* — where every other
language localizes the slot (spa "España", deu "Deutschland", zho "中国", bho "भारत"). English
alone was left as the raw template, plausibly because these prompts key off the language's
country and English has no single one. As-is those 8 rows are unanswerable: the model is asked
about a literal `{country}`.

Scope: **English `qa-oeg` only** — 0/8,640 qa-context rows and 0 rows in the other 23 languages
carry a placeholder. (`sum-sum` has 192 such rows, but that is the teammate's subtask and
predates this commit — worth passing on.)

**Actions:** (a) report to the organizers (schmidtova@ufal.mff.cuni.cz) — this is now the only
open data bug on our side; (b) do **not** silently substitute a value in `run_test.py`: picking
"the United States" or "the United Kingdom" invents an input the graders did not write, and the
gold was presumably produced against whatever they intend. `run_test.py` warns on these rows and
otherwise passes them through verbatim. 8 rows of 10,999 — not worth a hack; worth an email.

## 7. Qualitative smoke findings (job 3859059, base 9B, verbatim prompts)

10× Arabic `qa-context` + 10× Bhojpuri `qa-oeg`:

1. **False refusals on qa-context**: 4/10 Arabic rows answered "لا توجد إجابة" ("no
   answer") — including a definitional question whose answer is verbatim in the passage.
   The model over-uses the prompt's own escape hatch. Answerable-vs-not calibration is a
   concrete, fixable quality lever (few-shot demos or SFT targets that answer confidently
   when the answer is present).
2. **Bhojpuri output drifts to related languages**: most outputs came out in standard
   Hindi; one drifted into Nepali, one into Maithili; only fragments were genuinely
   Bhojpuri. The feared bho→hin slide is real and broader than feared (roadmap D
   confirmed; a fastText LID gate — roadmap F — would catch exactly this).
3. **Word budgets are mostly violated**: 150→235w, 120–150→282w, 100→60w; only 1/10
   landed in range (roadmap C confirmed).
4. **Style**: OEG outputs arrive in heavy markdown (headers, bold, emoji). Unknown
   whether human raters reward or penalize this vs the plain gold style — a distilled
   teacher inherits its own style, so decide deliberately (e.g. strip markdown in
   post-processing or keep it for "helpfulness").

## 8. What changes in the plan

| Roadmap item | Verdict after analysis |
|---|---|
| A (test-format alignment) | Done — `run_test.py` + this analysis. Dev A/B without lang-hint = job **3859645** (not 3859058): dropping the hint is ~free on the sources that matter (§4). |
| B (distillation) | ❌ **Done and lost (2026-07-17, jobs 3865036/3866054):** the distilled adapter fell ~12 COMBINED short of plain gold SFT on both faithful columns (MCIF, OEG) and won only aya, which does not reach the test set. The lang-hint confound was closed by 3866054, so the loss is the teacher data. **No route change; gold-LoRA 3822375 keeps both qa tasks.** The §5c warning held: the 122B run is 91% aya, only 363 train rows match `qa-oeg`, so OEG stayed the thinnest link and distillation did not fix it. See IMPLEMENTATION_NOTES §5.4. |
| C (instruction following) | **Upgraded from "nice" to "scored"**: format constraints are on every `qa-context` prompt; numeric word budgets are on **21% of `qa-oeg`** (§4), not all of it. Length control fails today. Augment SFT data with word-budget/format constraints + rewritten targets. |
| D (Bhojpuri kit) | **Confirmed essential** (drift is real). Only one surprise language — target bho specifically. |
| E (routing) | Legal and easy (`task` given). Route on the *faithful* proxies only (§5b) — ⚠️ **and tydiqa is not one of them**, which is the correction that settled this row. **`qa-context` → adapter, 0-shot** (MCIF, the only cross-lingual proxy: adapter wins EM 21.82 vs 0.61, F1 57.92 vs 28.15, chrF 49.26 vs 34.61, BERTScore 86.41 vs 74.38 — nothing dissents). ~~tydiqa says plain 3-shot (38.94 vs adapter 19.53)~~ — retracted 2026-07-16: monolingual, ~4% of the sub-task. **`qa-oeg` → adapter** (OEG: 29.06 vs 3-shot 25.55). belebele wins do not transfer; aya proxies only qa-oeg's short tail and must not drive the qa-oeg choice on its own. |
| F (LID gate) | Directly addresses observed bho drift; cheap. |
| G (3 submissions) | Updated 2026-07-17: primary = **gold-LoRA 3822375 + routed** (distillation lost, row B), variant = 9B 3-shot safe bet, variant = aggressive. |
