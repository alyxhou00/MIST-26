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
- Prompts are effectively single-line: only 2 of 10,999 qa rows contain a real newline
  (and 2 contain a literal `\n` escape). No unescaping needed.
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
- `qa-oeg`: word budgets are routine ("in 120–150 words", "150 शब्दन में"), in every
  language **including Bhojpuri**.

Implications: (a) the test prompt already tells the model the output language, so our
lang-hint system turn is redundant-at-best there (job 3859058 measures what dropping it
costs on dev); (b) instruction-following — especially non-English length control — is
directly scored, validating roadmap C.

## 5. No multiple-choice anywhere

Zero qa prompts match MC patterns (`(A)`, `A. … B. …`). The belebele-style format that
dominates our dev set (1,123/2,978 rows) and where the gold-SFT adapter got its biggest
win (chrF 52.70 → 85.82) **does not appear at test time**. Free-form extraction
(tydiqa-like) is the closest dev proxy for `qa-context` — and that is exactly where the
adapter collapsed (38.94 → 19.53). Dev overall chrF is **not** a faithful test predictor;
weight tydiqa + aya/OEG when comparing systems, and consider re-weighting or re-splitting
dev to mirror the test mix.

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
| A (test-format alignment) | Done — `run_test.py` + this analysis. Dev A/B without lang-hint = job 3859058. |
| B (distillation) | Unchanged priority. Teacher should see prompts in the **test's conversational style** where possible; OEG is 2,359 rows of mostly-unfixed headroom. |
| C (instruction following) | **Upgraded from "nice" to "scored"**: constraints are in every prompt; length control fails today. Augment SFT data with word-budget/format constraints + rewritten targets. |
| D (Bhojpuri kit) | **Confirmed essential** (drift is real). Only one surprise language — target bho specifically. |
| E (routing) | Legal and easy (`task` given). But belebele-format wins don't transfer; route on measured test-proxy performance (tydiqa/aya-like), not dev overall. |
| F (LID gate) | Directly addresses observed bho drift; cheap. |
| G (3 submissions) | Unchanged: primary = distilled+routed, variant = 9B 3-shot safe bet, variant = aggressive. |
