"""Roadmap C+D acceptance checks on official-test-set outputs (dev cannot measure either).

The C+D adapter (job 3869129) adds two things the v2 dev split has no way to score:

  * C -- word budgets on `qa-oeg`. `data/dev_v2.jsonl` carries no budget text at all
    (the augmentation lives in `data/train_v2-cd.jsonl`), so the only prompts that state
    a budget are the official test's 465 qa-oeg rows (`constraint_bank.parse_budget`).
  * D -- the 8,009-row Bhojpuri pack. dev has **zero** `bho` rows; the test set has 100
    in qa-oeg. The failure mode being checked is drift into standard Hindi.

So both checks read a `run_test.py` output file ({"id", "output"} per line), joined back
to `data/tests-07-20.jsonl` on `id`:

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

**qa-context needs a different toolkit, and `bho_lid` is not part of it.** The 360 bho
qa-context test rows differ from the 100 qa-oeg ones in three ways that change what can
honestly be measured:

  * **97% are cross-lingual** (the passage is in another language: eng 100, arb/spa/zho 25
    each, ...; only 10 rows have a bho passage). So there is a failure mode qa-oeg does not
    have -- answering in the *passage's* language rather than drifting to Hindi.
  * **The prompt asks for one sentence** ("एके वाक्य में जवाब दे सकीं"), so answers are
    ~10-25 words. `bho_lid.py` states in its own docstring that it is "usable on anything
    paragraph-sized, not to be trusted on single short sentences" -- so it is **skipped**
    on qa-context rather than run and quietly believed.
  * Answers are far shorter than even that suggests -- **median 3 words** -- and are often a
    noun phrase lifted from the passage, where "which language is this" is undecidable in
    principle. Those rows are counted as undecidable, never assigned.

What survives on one-sentence answers: **script** (Unicode range -- catches answering in
English/Chinese/Arabic with total reliability, though it cannot separate bho from Hindi,
both Devanagari), **refusal rate** (exact match against `constraint_bank`'s attested
phrase), **one-sentence compliance**, and -- the one that turned out to matter --
**contrastive function words**.

That last one is the lesson of this file. Script and sentence counts showed the C+D and
plain adapters as *identical* on these 360 rows (Devanagari 358/360, one sentence ~99%,
both), which reads as "D does nothing here". Reading pairs by hand said otherwise: C+D
writes `आ`/`खातिर`/`होखल`, plain writes `और`/`के लिए`/`की`. The aggregate was blind, not the
effect absent. Asking "which of these two words did it pick" needs only one word of output,
whereas `bho_lid`'s marker *density* needs a paragraph -- so on this data the contrastive
test resolves 39% of rows where `bho_lid` resolves almost none, and it separates the two
systems 99% vs 18% bho. Cf. the standing rule: read the data before believing an aggregate.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bho_lid import classify
from constraint_bank import context_tail, measure, parse_budget

SLACK = 0.15

# Unicode ranges, majority-vote over letters. Deliberately coarse: the question this answers
# is "did it answer in the passage's language instead of the asked one", which is a
# script-level failure. It cannot separate bho from Hindi -- both are Devanagari.
SCRIPTS = {
    "Deva": [(0x0900, 0x097F)],
    "Latn": [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)],
    "Arab": [(0x0600, 0x06FF), (0x0750, 0x077F)],
    "Han": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
    "Kana": [(0x3040, 0x30FF)],
    "Hang": [(0xAC00, 0xD7AF)],
    "Cyrl": [(0x0400, 0x04FF)],
    "Beng": [(0x0980, 0x09FF)],
}
# Sentence terminators, incl. the Devanagari danda. Abbreviations and decimals inflate the
# "." count, so a 2-sentence reading is soft; 3+ is a real violation.
TERMINATORS = "।॥.!?？！。"

# Minimal contrastive pairs: same grammatical function, different language. This is the
# instrument that works where `bho_lid` cannot -- it asks "which of these two specific words
# did it choose", which a 3-word answer can still answer, instead of "what is the marker
# density of this text", which needs a paragraph. Rows containing neither side are reported
# as undecidable rather than assigned, because a 2-word noun phrase lifted from the passage
# genuinely has no language beyond its script.
BHO_HIN_CONTRASTS = [
    ("'and'",        r"आ",                    r"और"),
    ("'for'",        r"खातिर",                r"के लिए"),
    ("copula",       r"बा|बाड़|होखल|होला",    r"है|हैं"),
    ("oblique",      r"के",                   r"की|को"),
    ("infinitive",   r"करे|करल",              r"करना|करता"),
]


def has_token(text: str, pattern: str) -> bool:
    """Whole-token match -- `के` must not fire inside `केवल`."""
    return bool(re.search(r"(?:^|[\s,।])(?:" + pattern + r")(?=[\s,।]|$)", f" {text} "))


def bho_vs_hin(text: str) -> str | None:
    """'bho', 'hin', or None when the text commits to neither."""
    b = sum(has_token(text, pair[1]) for pair in BHO_HIN_CONTRASTS)
    h = sum(has_token(text, pair[2]) for pair in BHO_HIN_CONTRASTS)
    if b == h:
        return None
    return "bho" if b > h else "hin"


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
    """bho_lid on the qa-oeg bho rows only -- see the module docstring for why qa-context
    is excluded rather than measured badly."""
    stats = Counter()
    for rid, row in tests.items():
        if row["question_lang"] != "bho" or row["task"] != "qa-oeg" or rid not in outs:
            continue
        text = outs[rid]
        stats["n"] += 1
        if not text.strip():
            stats["empty"] += 1
            continue
        label = classify(text)
        stats[label or "abstain"] += 1
    return stats


def script_of(text: str) -> str:
    """Majority script over the letters in `text`, or "none" if it has none."""
    counts: Counter = Counter()
    for ch in text:
        cp = ord(ch)
        for name, ranges in SCRIPTS.items():
            if any(lo <= cp <= hi for lo, hi in ranges):
                counts[name] += 1
                break
    return counts.most_common(1)[0][0] if counts else "none"


def context_lang_of(rid: str) -> str:
    """qa-context ids are qa-context_{n}_{question_lang}_{context_lang}."""
    return rid.rsplit("_", 1)[-1]


def check_context(tests: dict, outs: dict, lang: str) -> dict:
    """Script / refusal / one-sentence checks for one question language's qa-context rows."""
    stats = Counter()
    scripts: Counter = Counter()
    by_ctx: dict[str, Counter] = {}
    try:
        refusal = context_tail(lang).refusal_phrase
    except Exception:
        refusal = None

    for rid, row in tests.items():
        if row["task"] != "qa-context" or row["question_lang"] != lang or rid not in outs:
            continue
        text = outs[rid].strip()
        stats["n"] += 1
        if not text:
            stats["empty"] += 1
            continue

        sc = script_of(text)
        scripts[sc] += 1
        ctx = context_lang_of(rid)
        by_ctx.setdefault(ctx, Counter())
        by_ctx[ctx]["n"] += 1
        by_ctx[ctx][sc] += 1

        if refusal and refusal in text:
            stats["refusal"] += 1
            continue          # a fixed phrase says nothing about free-generation drift
        stats["answered"] += 1
        stats[f"lex_{bho_vs_hin(text) or 'undecidable'}"] += 1
        # One sentence was asked for. Trailing terminator doesn't count as a second.
        n_sent = sum(text.count(t) for t in TERMINATORS)
        if text and text[-1] in TERMINATORS:
            n_sent -= 1
        if n_sent <= 0:
            stats["one_sentence"] += 1
        elif n_sent == 1:
            stats["two_sentences"] += 1
        else:
            stats["many_sentences"] += 1
    return {"totals": stats, "scripts": scripts, "by_ctx": by_ctx}


def report(path: str, tests: dict, ctx_lang: str) -> None:
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
    elif not any(tests[r]["task"] == "qa-context" for r in outs if r in tests):
        print("D -- no bho rows found (wrong task slice?)")

    c = check_context(tests, outs, ctx_lang)
    t = c["totals"]
    if t["n"]:
        n = t["n"]
        pct = lambda k: 100.0 * t[k] / n
        print(f"D(ctx) -- {n} {ctx_lang} qa-context rows "
              f"(bho_lid NOT applicable at one-sentence length -- see docstring):")
        want = "Deva" if ctx_lang in ("bho", "hin", "mar", "npi", "mai") else "?"
        right = c["scripts"].get(want, 0)
        print(f"    script: {want} {right} ({100.0 * right / n:.1f}%)   "
              + "  ".join(f"{k}={v}" for k, v in c["scripts"].most_common() if k != want))
        wrong = [(ctx, cc) for ctx, cc in c["by_ctx"].items() if cc["n"] - cc[want] > 0]
        if wrong:
            wrong.sort(key=lambda kv: kv[1][want] / kv[1]["n"])
            print("    wrong-script rows by PASSAGE language (the drift-to-passage failure): "
                  + ", ".join(f"{ctx} {cc['n'] - cc[want]}/{cc['n']}" for ctx, cc in wrong[:8]))
        print(f"    one sentence {t['one_sentence']} ({pct('one_sentence'):.1f}%)   "
              f"two {t['two_sentences']} ({pct('two_sentences'):.1f}%)   "
              f"3+ {t['many_sentences']} ({pct('many_sentences'):.1f}%)")
        print(f"    used the attested refusal phrase: {t['refusal']} ({pct('refusal'):.1f}%)"
              f"   empty: {t['empty']}")
        dec = t["lex_bho"] + t["lex_hin"]
        if dec:
            print(f"    bho-vs-Hindi by contrastive function words, on the {t['answered']} "
                  f"non-refusal answers:")
            print(f"        bho-leaning {t['lex_bho']}   hin-leaning {t['lex_hin']}   "
                  f"undecidable {t['lex_undecidable']} (too short to commit)"
                  f"  ->  {100.0 * t['lex_bho'] / dec:.0f}% bho of the {dec} decidable")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outputs", nargs="+", help="run_test.py output JSONL files")
    ap.add_argument("--test-file", default="data/tests-07-20.jsonl")
    ap.add_argument("--ctx-lang", default="bho",
                    help="question language for the qa-context script/refusal/sentence "
                         "checks (default bho -- the only language D targets)")
    args = ap.parse_args()

    tests = load_tests(args.test_file)
    for path in args.outputs:
        report(path, tests, args.ctx_lang)


if __name__ == "__main__":
    main()
