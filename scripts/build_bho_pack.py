"""Roadmap D: build Bhojpuri SFT rows for the surprise language.

`bho` has **zero rows in the training data** and 460 rows in the test set, and the base 9B
cannot produce it: the smoke run (job 3859059) drifted all 10 Bhojpuri qa-oeg outputs into
standard Hindi, Nepali or Maithili. This script assembles Bhojpuri training rows from
ungated public corpora, in the schema `train_lora.py --data` consumes -- the same schema
`filter_teacher.py` emits, so the two files concatenate.

    # on the cluster LOGIN node: download + assemble (I/O only, no compute -- see CLAUDE rules)
    python scripts/build_bho_pack.py --out data/sft-bho.jsonl --report
    python scripts/build_bho_pack.py --out data/sft-bho.jsonl

    # then concatenate with the distilled data and train
    cat data/sft-distilled-c.jsonl data/sft-bho.jsonl > data/sft-final.jsonl
    python scripts/train_lora.py --data data/sft-final.jsonl

## Sources (both verified ungated, 2026-07-15)

`HuggingFaceFW/fineweb-2` config `bho_Deva` -- 18,666 web documents, 25.7MB. The only real
volume of native Bhojpuri prose available. Used for a **continuation** task: show the first
sentences, ask for the rest in Bhojpuri. This is the closest available match to the test's
`qa-oeg` bho rows, which ask for ~150 words of free Bhojpuri prose.

`CohereLabs/xP3x` config `bho_Deva` -- FLORES translated into Bhojpuri, wrapped in prompts.
Used for a **translate-into-Bhojpuri** task. Two caveats that shape how it is used:
  * it is entirely FLORES-derived, so the unique Bhojpuri content is only ~2k sentences,
    fanned out across 200+ source languages x 3 templates into 1.22M rows. Row count is not
    data. `--xp3x-source-langs` keeps only source languages that exist in the MIST test set
    (default: eng/hin), and rows are deduplicated on the Bhojpuri side.
  * hin->bho is the single most on-target slice available anywhere: it shows the model the
    two languages side by side, which is exactly the confusion it is making.

Note the roadmap named FLORES+ and Aya as the sources. Both turned out to be dead ends:
`openlanguagedata/flores_plus` is **also gated** (`gated: auto`) -- it was chosen precisely
because `facebook/flores` is gated -- and `CohereLabs/aya_collection_language_split` has 132
language configs with **no Bhojpuri at all**. xP3x reaches the same FLORES Bhojpuri content
ungated, and fineweb-2 replaces Aya as the volume source.

## Quality gate

Web-scraped "Bhojpuri" is not reliably Bhojpuri, and a Hindi document mislabelled as bho
would *teach* the very drift we are fixing. Every candidate is therefore checked with
`scripts/bho_lid.py`. `--report` prints what the gate accepted and rejected -- read it
before trusting the output.

How much this gate is really doing, measured rather than assumed: fineweb-2's bho_Deva is
**largely clean** -- ~96% of its documents classify as Bhojpuri and only ~1% as hin/npi/mai.
So the gate is cheap insurance and a way to abstain on mixed-register junk, not a filter
rescuing us from mass contamination. An earlier version of this docstring claimed it caught
"167 Hindi / 80 Nepali / 16 Maithili" documents; that was wrong -- those were mostly the
classifier's own false negatives on genuine Bhojpuri (see bho_lid.py's MIN_DENSITY).

Constraint sentences on the continuation rows come from `scripts/constraint_bank.py`, so
these rows also carry native Bhojpuri word budgets ("150 शब्दन में जवाब दीं।"): the bho test
rows fail *both* language and length control today, and one row can teach both.
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from bho_lid import classify, is_bhojpuri
from constraint_bank import budget_bounds, measure, word_budget

# Bhojpuri sentence enders (Danda, double Danda, and the ASCII period web text often uses).
_SENT = re.compile(r"(?<=[।॥.])\s+")

# Task framing. Kept in Bhojpuri where the model must answer in Bhojpuri: an English
# instruction wrapping a Bhojpuri answer is exactly the cross-lingual shape that lets the
# model slip back into Hindi. "इहाँ से आगे भोजपुरी में लिखीं" = "continue from here in Bhojpuri".
CONTINUE_INSTRUCTION = "एह लेख के आगे भोजपुरी में लिखीं:"
TRANSLATE_INSTRUCTION = "एह वाक्य के भोजपुरी में अनुवाद करीं:"

SOURCE_FINEWEB = "HuggingFaceFW/fineweb-2:bho_Deva"
SOURCE_XP3X = "CohereLabs/xP3x:bho_Deva"


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text.strip()) if s.strip()]


def fineweb_rows(args, stats: Counter) -> list[dict]:
    """Continuation rows from native Bhojpuri web documents."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb-2", "bho_Deva", split="train",
                      streaming=args.streaming)
    rng = random.Random(args.seed)
    out = []
    for i, row in enumerate(ds):
        if args.max_docs and i >= args.max_docs:
            break
        text = (row.get("text") or "").strip()
        stats["fineweb: seen"] += 1
        sents = sentences(text)
        if len(sents) < args.min_sentences:
            stats["fineweb: too few sentences"] += 1
            continue
        # The LID gate is precision-safe on documents, not on single sentences -- so judge
        # the whole document, and only then cut it into prompt/target.
        lang = classify(text)
        if lang != "bho":
            stats[f"fineweb: LID rejected ({lang})"] += 1
            continue
        lead_n = max(1, min(args.lead_sentences, len(sents) - 1))
        lead, rest = " ".join(sents[:lead_n]), " ".join(sents[lead_n:])
        n = measure(rest, "bho")
        if not (args.min_words <= n <= args.max_words):
            stats["fineweb: target length out of range"] += 1
            continue
        # Both halves must be Bhojpuri: a document can start in Bhojpuri and continue in
        # Hindi, and it is the TARGET half that becomes the training signal.
        if not is_bhojpuri(rest):
            stats["fineweb: target half not bho"] += 1
            continue
        lo, hi = budget_bounds(n)
        prompt = f"{CONTINUE_INSTRUCTION}\n\n{lead}\n\n{word_budget('bho', lo, hi)}"
        out.append({"source": SOURCE_FINEWEB, "lang_code": "bho_Deva",
                    "input": prompt, "output": rest, "origin": "bho-pack",
                    "constraint": "budget-range"})
        stats["fineweb: KEPT"] += 1
        if args.limit_fineweb and len(out) >= args.limit_fineweb:
            break
    rng.shuffle(out)
    return out


def xp3x_rows(args, stats: Counter) -> list[dict]:
    """Translate-into-Bhojpuri rows from FLORES via xP3x, one row per Bhojpuri sentence."""
    from datasets import load_dataset

    # Preference order, not just membership: position in --xp3x-source-langs breaks ties
    # when the same Bhojpuri sentence is reachable from several source languages.
    prefs = [s.strip() for s in args.xp3x_source_langs.split(",") if s.strip()]
    rank_of = {src: i for i, src in enumerate(prefs)}
    ds = load_dataset("CohereLabs/xP3x", "bho_Deva", split="train", streaming=args.streaming)

    # target -> (rank, row). Keeping ONE row per Bhojpuri target: the unique content is only
    # ~2k sentences, and repeating each of them per source language would just reweight them.
    best: dict[str, tuple[int, dict]] = {}
    for row in ds:
        stats["xp3x: seen"] += 1
        cfg = row.get("config") or ""            # e.g. "hin_Deva-bho_Deva"
        src = cfg.split("-")[0]
        if src not in rank_of:
            stats["xp3x: source language not wanted"] += 1
            continue
        if row.get("template") != args.xp3x_template:
            stats["xp3x: other template"] += 1
            continue
        target = (row.get("targets") or "").strip()
        source_text = (row.get("inputs") or "").strip()
        if not target or not source_text:
            stats["xp3x: empty"] += 1
            continue
        rank = rank_of[src]
        if target in best and best[target][0] <= rank:
            stats["xp3x: duplicate target (kept better source)"] += 1
            continue
        # xP3x bakes the task into `inputs`, differently per template:
        #   continuation-x-x  <text> | The previous text is in Hindi. Here is a translation
        #                     to Bhojpuri:
        #   command-x-x       <text> Give me the same text in Bhojpuri.
        #   question-x-x      A text in English: "<text>\nWhat's the text in Bhojpuri?
        # We keep only continuation-x-x (--xp3x-template) and strip its framing: its marker
        # is unambiguous, and question-x-x is malformed upstream anyway (it opens a quote it
        # never closes). Restricting to one template costs no data -- every FLORES sentence
        # appears under all three -- and it is what makes this strip reliable.
        stem = re.split(r"\s*\|?\s*The previous text is in\b", source_text)[0]
        stem = stem.strip().rstrip("|").strip()
        if not stem:
            stats["xp3x: no stem after stripping framing"] += 1
            continue
        # Belt and braces: if any English task framing survived, the row is malformed --
        # drop it rather than train on a half-stripped prompt.
        if re.search(r"(text in Bhojpuri|previous text is in|Give me the same text)", stem):
            stats["xp3x: framing survived stripping"] += 1
            continue
        if target in best:
            stats["xp3x: duplicate target (replaced, better source)"] += 1
        best[target] = (rank, {"source": SOURCE_XP3X, "lang_code": "bho_Deva",
                               "input": f"{TRANSLATE_INSTRUCTION}\n\n{stem}",
                               "output": target, "origin": "bho-pack",
                               "constraint": "", "xp3x_source": src})
    out = [r for _, r in best.values()]
    stats["xp3x: KEPT"] += len(out)
    for src, n in Counter(r["xp3x_source"] for r in out).items():
        stats[f"xp3x: kept from {src}"] = n
    return out[: args.limit_xp3x] if args.limit_xp3x else out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data/sft-bho.jsonl")
    ap.add_argument("--report", action="store_true",
                    help="print the gate's accept/reject breakdown and samples, write nothing")
    ap.add_argument("--no-fineweb", action="store_true")
    ap.add_argument("--no-xp3x", action="store_true")
    ap.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True,
                    help="stream instead of downloading whole configs (default: on -- xP3x's "
                         "bho_Deva config is 302MB/1.22M rows and we keep a few thousand)")
    ap.add_argument("--max-docs", type=int, default=0,
                    help="stop after this many fineweb docs (0 = all 18,666)")
    ap.add_argument("--limit-fineweb", type=int, default=6000)
    ap.add_argument("--limit-xp3x", type=int, default=4000)
    ap.add_argument("--xp3x-template", default="continuation-x-x",
                    choices=["continuation-x-x", "command-x-x", "question-x-x"],
                    help="xP3x wraps each pair in three prompt templates; keep just one so "
                         "the framing can be stripped reliably (default: continuation-x-x). "
                         "Costs no data -- every sentence appears under all three")
    ap.add_argument("--xp3x-source-langs", default="hin_Deva,eng_Latn",
                    help="xP3x source languages to keep, MOST WANTED FIRST -- order breaks "
                         "ties when one Bhojpuri sentence is reachable from several sources. "
                         "Default hin,eng: both are in the test set, and hin->bho is first "
                         "because it shows the model Hindi and Bhojpuri side by side, which "
                         "is the exact contrast it fails. 200+ others exist but are off-task")
    ap.add_argument("--min-sentences", type=int, default=4,
                    help="skip short fineweb docs: the LID gate needs document length to be "
                         "precision-safe, and we split the doc into lead + target")
    ap.add_argument("--lead-sentences", type=int, default=2)
    ap.add_argument("--min-words", type=int, default=40)
    ap.add_argument("--max-words", type=int, default=300,
                    help="the test's bho qa-oeg budgets top out around 180 words")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    stats: Counter = Counter()
    rows: list[dict] = []
    try:
        if not args.no_fineweb:
            print("fineweb-2 bho_Deva: streaming ...", flush=True)
            rows += fineweb_rows(args, stats)
        if not args.no_xp3x:
            print("xP3x bho_Deva: streaming ...", flush=True)
            rows += xp3x_rows(args, stats)
    except ImportError:
        sys.exit("needs `datasets` -- run this in $WORK/mist-venv on the cluster login node")

    print("\ngate breakdown:")
    for k, v in sorted(stats.items()):
        print(f"  {k:44s} {v:7d}")

    by_source = Counter(r["source"] for r in rows)
    print(f"\nassembled {len(rows)} bho rows: "
          + ", ".join(f"{s}={n}" for s, n in by_source.items()))
    if not rows:
        sys.exit("no rows survived the gate -- inspect the breakdown above before proceeding")

    print("\nsamples:")
    for src in by_source:
        r = next(x for x in rows if x["source"] == src)
        print(f"\n--- {src} ---")
        print(f"  input : {r['input'][:220]}")
        print(f"  output: {r['output'][:220]}")
        print(f"  measured: {measure(r['output'], 'bho')} words")

    if args.report:
        print("\n--report: nothing written")
        return
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            # qa_idx is bookkeeping only (train_lora.py reads input/lang_code/output); these
            # rows have no counterpart in the sample data's index, so number them separately
            # and keep them negative to guarantee they never collide with a real qa_idx.
            f.write(json.dumps({"qa_idx": -(i + 1), **r}, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(rows)} rows -> {args.out}")
    print(f"concatenate with the distilled data, then: "
          f"python scripts/train_lora.py --data data/sft-final.jsonl")


if __name__ == "__main__":
    main()
