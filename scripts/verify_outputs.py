"""Roadmap C+D acceptance checks on official-test-set outputs (dev cannot measure either).

The C+D adapter (job 3869129) adds two things the v2 dev split has no way to score:

  * C -- word budgets on `qa-oeg`. `data/dev_v2.jsonl` carries no budget text at all
    (the augmentation lives in `data/train_v2-cd.jsonl`), so the only prompts that state
    a budget are the official test's 465 qa-oeg rows (`constraint_bank.parse_budget`).
  * D -- the 8,009-row Bhojpuri pack. dev has **zero** `bho` rows; the test set has 100
    in qa-oeg. The failure mode being checked is drift into standard Hindi.

So both checks read a `run_test.py` output file ({"id", "output"} per line), joined back
to `data/tests.jsonl` on `id`:

    python scripts/verify_outputs.py runs/test-qaoeg-cd-3869129.jsonl \
        [runs/test-qaoeg-plain-3867139.jsonl ...]

Several files are reported side by side, which is the only way to read the result: a
compliance or LID number is meaningless on its own, it only means something against the
adapter that was trained without C+D.

Caveats worth keeping in mind when reading the output:
  * budget compliance counts a prediction as compliant when its `measure()` length (chars
    for jpn/zho, words elsewhere) falls inside the band the prompt states, with the same
    15% slack `budget_bounds` used at training time. Over- and under-shoot are reported
    separately: they are different failures (padding vs truncation).
  * `bho_lid.classify` abstains rather than guess, and is documented as unreliable on
    single short sentences -- abstentions are reported, never folded into either side.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bho_lid import classify
from constraint_bank import measure, parse_budget

SLACK = 0.15


def load_tests(path: str) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, f)}


def load_outputs(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r["output"] for r in map(json.loads, f)}


def band(lo: int, hi: int | None) -> tuple[int, int]:
    """The prompt's stated budget as a [lo, hi] acceptance band. A one-sided budget
    ("in at most 150 words") gets no floor beyond 1 word -- the prompt does not ask for one."""
    if hi is None:
        return 1, int(lo * (1 + SLACK))
    return int(lo * (1 - SLACK)), int(hi * (1 + SLACK))


def check_budgets(tests: dict, outs: dict) -> dict:
    stats = Counter()
    per_lang: dict[str, Counter] = {}
    for rid, row in tests.items():
        if row["task"] != "qa-oeg" or rid not in outs:
            continue
        parsed = parse_budget(row["prompt"], row["question_lang"])
        if parsed is None:
            continue
        lang = row["question_lang"]
        lo, hi = band(*parsed)
        n = measure(outs[rid], lang)
        bucket = "ok" if lo <= n <= hi else ("over" if n > hi else "under")
        stats[bucket] += 1
        stats["n"] += 1
        per_lang.setdefault(lang, Counter())[bucket] += 1
        per_lang[lang]["n"] += 1
        # Track the size of the miss, not just its direction: a 10% overshoot and a 4x
        # overshoot are the same "violation" but not the same problem.
        if bucket == "over":
            stats["over_excess"] += n - hi
    return {"totals": stats, "per_lang": per_lang}


def check_bho(tests: dict, outs: dict) -> Counter:
    stats = Counter()
    for rid, row in tests.items():
        if row["question_lang"] != "bho" or rid not in outs:
            continue
        text = outs[rid]
        stats["n"] += 1
        if not text.strip():
            stats["empty"] += 1
            continue
        label = classify(text)
        stats[label or "abstain"] += 1
    return stats


def report(path: str, tests: dict) -> None:
    outs = load_outputs(path)
    print(f"\n=== {path}  ({len(outs)} rows) ===")

    b = check_budgets(tests, outs)
    t = b["totals"]
    if t["n"]:
        pct = lambda k: 100.0 * t[k] / t["n"]
        print(f"C -- word-budget compliance on {t['n']} qa-oeg rows that state a budget:")
        print(f"    compliant {t['ok']:4d} ({pct('ok'):5.1f}%)   "
              f"over {t['over']:4d} ({pct('over'):5.1f}%)   "
              f"under {t['under']:4d} ({pct('under'):5.1f}%)")
        if t["over"]:
            print(f"    mean overshoot beyond the band: {t['over_excess'] / t['over']:.0f} units")
        worst = sorted(b["per_lang"].items(), key=lambda kv: kv[1]["ok"] / kv[1]["n"])
        shown = ", ".join(f"{lang} {100.0 * c['ok'] / c['n']:.0f}%" for lang, c in worst[:6])
        print(f"    worst languages: {shown}")
    else:
        print("C -- no budget-carrying rows found (wrong task slice?)")

    s = check_bho(tests, outs)
    if s["n"]:
        print(f"D -- bho_lid on {s['n']} bho rows "
              f"(function-word LID; abstains rather than guess):")
        order = ["bho", "hin", "mai", "npi", "abstain", "empty"]
        print("    " + "  ".join(f"{k}={s[k]} ({100.0 * s[k] / s['n']:.1f}%)"
                                 for k in order if s[k]))
    else:
        print("D -- no bho rows found (wrong task slice?)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outputs", nargs="+", help="run_test.py output JSONL files")
    ap.add_argument("--test-file", default="data/tests.jsonl")
    args = ap.parse_args()

    tests = load_tests(args.test_file)
    for path in args.outputs:
        report(path, tests)


if __name__ == "__main__":
    main()
