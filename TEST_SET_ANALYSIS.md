# Official test set — analysis and strategic implications

Source: `pinzhenchen/wmt26-mist-test`, "Final version - 14 July 2026" (single `tests.jsonl`,
47MB), analyzed 2026-07-15 together with the first qualitative smoke run (job 3859059,
`slurm/smoke-run-test.sbatch`; outputs kept in `predictions/smoke-test-*-3859059.jsonl`).
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
    0/2,259 qa-oeg.
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
  **including Bhojpuri** — but they are **not** on every prompt. Measured 2026-07-15:
  exactly **21 of the 100 unique qa-oeg prompts per language** carry a numeric budget
  (471/2,359 rows; identical 21/100 in all 23 non-empty languages, because qa-oeg is a
  **parallel corpus** — the same 100 prompts translated 24 ways, yoruba 59). An earlier
  version of this doc called budgets "routine", which reads as "on every prompt"; the real
  scale is ~1/5 of qa-oeg. Roadmap C still stands, just at that scale.

Implications: (a) the test prompt already tells the model the output language, so our
lang-hint system turn is redundant-at-best there — **measured (job 3859645): dropping it
costs 25.97 vs 27.64 overall, but the loss is almost entirely belebele (52.70→32.42), which
does not transfer; on the sources the test set actually has, the cost is under 1 chrF and
OEG is +0.09. The test format can be used as-is**; (b) instruction-following — especially
non-English length control — is directly scored, validating roadmap C.

## 5. No multiple-choice anywhere

Zero qa prompts match MC patterns (`(A)`, `A. … B. …`). The belebele-style format that
dominates our dev set (1,123/2,978 rows) and where the gold-SFT adapter got its biggest
win (chrF 52.70 → 85.82) **does not appear at test time**. Free-form extraction
(tydiqa-like) is the closest dev proxy for `qa-context` — and that is exactly where the
adapter collapsed (38.94 → 19.53). Dev overall chrF is **not** a faithful test predictor.

## 5b. aya is not a proxy for `qa-oeg` either (verified 2026-07-15 against the official file)

An earlier version of this doc said to "weight tydiqa + aya/OEG". **That was wrong about
aya.** README's sub-task table groups `CohereLabs/aya_dataset` with `wmt25-mist-oeg-gpt-4.1`
under "open-ended generation" — a fair *task* grouping, but not a statement that the two
behave alike, and they do not. Measured gold length (dev split, whitespace word count):

| dev source | gold words p25/p50/p75 | test task it supposedly proxies |
|---|---|---|
| `wmt25-mist-oeg-gpt-4.1` (n=97) | 30 / **175** / 227 | `qa-oeg` — asks for **120–180 words**: exact match |
| `CohereLabs/aya_dataset` (n=978) | 6 / **24** / 60 | `qa-oeg` — **7× too short**, wrong output regime |
| `copenlu/answerable_tydiqa` (n=615) | 1 / **2** / 3 | `qa-context` — extraction: match |

aya rows are short questions with short answers ("Fortnite mobilde var mı?", gold 152 chars)
and carry **no passage**, so they are neither `qa-context` (which is passage+question) nor
`qa-oeg` (which is 120–180-word composition). Consequence: **the faithful dev proxy is only
tydiqa (615) + MCIF (165) for `qa-context` and OEG (97) for `qa-oeg` — ~877 of 2,978 rows
(29%).** belebele (1,123, no MC at test) and aya (978, wrong length regime) are the other
71% and should not drive system choice.

⚠️ Scope of this claim: aya's unsuitability as an *evaluation proxy* is measured. Whether
aya rows are harmful as *training* data is a separate, untested question — a `qa-oeg`
adapter trained mostly on 24-word targets would plausibly learn the wrong length, but that
has not been run.

⚠️ This makes `qa-oeg` the thin part of the whole plan: 2,359 test rows backed by 97 dev
rows and 363 train rows (see §5c).

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

## 6. Data bug: 100 empty prompts

`qa-oeg_1..100_eng_eng` — the entire English qa-oeg block — have `prompt == ""`.
`run_test.py` guards these (emits `output: ""` + a warning). **Action: report to the
organizers** (schmidtova@ufal.mff.cuni.cz); if fixed data ships, re-run just
`--task qa-oeg --lang eng`.

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
| B (distillation) | Unchanged priority, but see §5c for what each teacher run covers: the 122B run is 91% aya, so it is **not** a ready-made `qa-oeg` training set, and only 363 train rows match the `qa-oeg` regime. OEG is 2,359 test rows of mostly-unfixed headroom on the thinnest data we have. |
| C (instruction following) | **Upgraded from "nice" to "scored"**: format constraints are on every `qa-context` prompt; numeric word budgets are on **21% of `qa-oeg`** (§4), not all of it. Length control fails today. Augment SFT data with word-budget/format constraints + rewritten targets. |
| D (Bhojpuri kit) | **Confirmed essential** (drift is real). Only one surprise language — target bho specifically. |
| E (routing) | Legal and easy (`task` given). Route on the *faithful* proxies only (§5b): `qa-context` → tydiqa says plain 3-shot (38.94 vs adapter 19.53); `qa-oeg` → OEG says adapter (29.06 vs 3-shot 25.55). belebele and aya wins do not transfer and must not drive the choice. |
| F (LID gate) | Directly addresses observed bho drift; cheap. |
| G (3 submissions) | Unchanged: primary = distilled+routed, variant = 9B 3-shot safe bet, variant = aggressive. |
