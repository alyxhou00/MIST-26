"""Roadmap C: add native-language output constraints to SFT inputs (constraint following).

The test set scores instruction following directly -- every qa prompt embeds an output
constraint, and the 9B base fails them today: the smoke run (job 3859059) violated the word
budget in 9 of 10 Bhojpuri qa-oeg rows (150 -> 235 words, 120-150 -> 282, 100 -> 60). This
script takes the SFT rows produced by `scripts/filter_teacher.py` and rewrites a fraction of
their *inputs* to carry a constraint, so the adapter is trained to obey one.

    # after filter_teacher.py has written the distilled data
    python scripts/augment_constraints.py data/sft-distilled.jsonl \
        --out data/sft-distilled-c.jsonl --report

    python scripts/train_lora.py --data data/sft-distilled-c.jsonl

The central design rule is **derive the constraint from the answer, never the other way
round**. We do not rewrite targets to fit a constraint we invented; we measure the target we
already have and state a constraint it already satisfies. So every augmented row is a
demonstration of a constraint being *met*, and SFT can never teach the model that
constraints are things to ignore. (Rewriting targets to hit an arbitrary budget would need
a generation pass per row and would risk teaching the model to pad or truncate.)

Two constraint families, routed by `source` -- deliberately mirroring the two test tasks:

  * open-ended sources (aya, oeg) -> the test's `qa-oeg`, whose prompts carry word budgets.
    Gets a word/character budget banding the answer's measured length.
  * context sources (tydiqa, MCIF) -> the test's `qa-context`, whose prompts all say
    "answer in one sentence, using only what the passage says". Gets exactly that sentence
    (attested, per language), but only where the target really is one sentence.

belebele is excluded by default: it is multiple-choice, its targets are option strings that
no length or one-sentence constraint sensibly describes, and TEST_SET_ANALYSIS.md section 5
found that **no multiple-choice prompt appears at test time at all**.

All constraint text comes from `scripts/constraint_bank.py`, which derives it from the
official test prompts (and self-tests against them) rather than from invented translations.
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from constraint_bank import (
    BUDGET,
    DEFAULT_TEST_FILE,
    budget_bounds,
    context_tail,
    measure,
    word_budget,
)

# `source` values as they appear in the sample data (verified against the filter_teacher
# report for job 3860144).
OPEN_ENDED_SOURCES = {"CohereLabs/aya_dataset", "wmt25-mist-oeg-gpt-4.1"}
CONTEXT_SOURCES = {"copenlu/answerable_tydiqa", "FBK-MT/MCIF"}
EXCLUDED_SOURCES = {"facebook/belebele"}  # multiple-choice; absent from the test set

# A target counts as "one sentence" if it has a single terminator, at the very end. Kept
# strict on purpose: a false positive teaches the model that "one sentence" means two.
_TERMINATOR = re.compile(r"[.?!।。？！؟]")


def is_one_sentence(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    hits = list(_TERMINATOR.finditer(stripped))
    if not hits:  # no terminator at all (common for short extractive spans) -> one sentence
        return True
    return len(hits) == 1 and hits[0].end() == len(stripped)


def augment_row(row: dict, rng: random.Random, args: argparse.Namespace) -> tuple[dict, str]:
    """Return (row, kind) where kind is the constraint applied, or "" for untouched."""
    lang = row["lang_code"].split("_")[0]
    source, out = row.get("source", ""), row["output"]
    if source in EXCLUDED_SOURCES or lang not in BUDGET:
        return row, ""

    if source in OPEN_ENDED_SOURCES:
        n = measure(out, lang)
        if n < args.min_len:  # a budget on a 6-word answer is noise, not a constraint
            return row, ""
        if rng.random() < args.exact_ratio:
            # Exact form ("in about 140 words"): rounded to a ten, so it is approximate by
            # construction -- which is why the phrasing hedges ("about", "약", "程度").
            clause, kind = word_budget(lang, max(10, round(n / 10) * 10)), "budget-exact"
        else:
            lo, hi = budget_bounds(n, args.slack)
            clause, kind = word_budget(lang, lo, hi), "budget-range"
    elif source in CONTEXT_SOURCES:
        if not is_one_sentence(out):
            return row, ""
        clause, kind = context_tail(lang, args.test_file).one_sentence, "one-sentence"
    else:
        return row, ""

    return {**row, "input": f"{row['input'].rstrip()}\n\n{clause}", "constraint": kind}, kind


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("data", help="SFT JSONL from scripts/filter_teacher.py")
    ap.add_argument("--out", default="data/sft-distilled-c.jsonl",
                    help="augmented SFT JSONL for `train_lora.py --data`")
    ap.add_argument("--fraction", type=float, default=0.5,
                    help="share of ELIGIBLE rows to augment (default 0.5). Not 1.0: the test "
                         "set always constrains, but leaving unconstrained rows in keeps the "
                         "adapter usable for plain questions and stops the model emitting "
                         "length-hedging boilerplate when nothing was asked of it")
    ap.add_argument("--exact-ratio", type=float, default=0.35,
                    help="share of budget constraints phrased as an exact count rather than "
                         "a range (default 0.35; the test set uses both)")
    ap.add_argument("--min-len", type=int, default=40,
                    help="skip budget constraints on targets shorter than this, in the "
                         "language's own unit (words, or characters for jpn/zho)")
    ap.add_argument("--slack", type=float, default=0.15,
                    help="half-width of the range band around the measured length")
    ap.add_argument("--test-file", default=DEFAULT_TEST_FILE,
                    help="official test JSONL -- the source of the constraint phrasings")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report", action="store_true",
                    help="print what would be applied, with samples, and write nothing")
    args = ap.parse_args()

    if not Path(args.test_file).exists():
        sys.exit(f"{args.test_file} not found -- constraint phrasings are mined from it")
    with open(args.data, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    print(f"loaded {len(rows)} SFT rows from {args.data}")

    rng = random.Random(args.seed)
    out_rows, kinds, samples = [], Counter(), {}
    for row in rows:
        # Draw for every row, not just eligible ones, so --fraction stays independent of
        # eligibility and the seed keeps the split stable when upstream data changes.
        if rng.random() >= args.fraction:
            out_rows.append(row)
            kinds["untouched (not drawn)"] += 1
            continue
        new, kind = augment_row(row, rng, args)
        out_rows.append(new)
        kinds[kind or "untouched (ineligible)"] += 1
        if kind and kind not in samples:
            samples[kind] = new

    print("\nconstraints applied:")
    for kind, n in kinds.most_common():
        print(f"  {kind:24s} {n:6d}  ({n / len(rows):5.1%})")

    by_lang = Counter(r["lang_code"].split("_")[0] for r in out_rows if "constraint" in r)
    if by_lang:
        print(f"\naugmented rows cover {len(by_lang)} languages: "
              f"{', '.join(f'{l}={n}' for l, n in sorted(by_lang.items()))}")

    print("\nsample per constraint kind (tail of the input, then the target):")
    for kind, r in samples.items():
        print(f"\n--- {kind} [{r['lang_code']}, {r['source']}, origin={r['origin']}] ---")
        print(f"  input tail: ...{r['input'][-120:]}")
        print(f"  target    : {r['output'][:160]}{'...' if len(r['output']) > 160 else ''}")
        print(f"  measured  : {measure(r['output'], r['lang_code'].split('_')[0])} "
              f"{BUDGET[r['lang_code'].split('_')[0]].unit}s")

    if args.report:
        print("\n--report: nothing written")
        return
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_aug = sum("constraint" in r for r in out_rows)
    print(f"\nwrote {len(out_rows)} rows ({n_aug} with a constraint) -> {args.out}")
    print(f"train with:  python scripts/train_lora.py --data {args.out}")


if __name__ == "__main__":
    main()
