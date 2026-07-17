"""Build the item-split, test-format train/dev set (v2) from the sample + upstream data.

Replaces the old row-level 80/20 split of samples.jsonl (leaky: DATA_AUDIT.md section 2)
with a dataset that (a) matches the official test-set format and (b) is split BY ITEM, so
no dev item has a train twin in any language. Outputs `data/train_v2.jsonl` and
`data/dev_v2.jsonl` with the schema

    task           qa-context | qa-oeg | sum-sum   (exact test task names, README table)
    question_lang  bare 3-letter code -- the question AND answer language (test invariant)
    context_lang   bare code of the passage/document language; null for qa-oeg and for
                   CrossSum (whose per-row article language the sample does not record)
    source         as in samples.jsonl
    input / output the SFT pair
    item_group     opaque group id; ALL rows of one underlying item (every language
                   version, both MCIF portions of a talk, answerable + unanswerable
                   pairings, and any later C-augmented variant) share it, and the
                   dev/train side is a pure function of it -- keep it grouped in any
                   further resampling.

Per-source treatment (decisions of 2026-07-17, DATA_AUDIT section 6 + user):

* belebele -> qa-context, REBUILT from upstream facebook/belebele (the sample's 19
  per-language item subsets only partially overlap and lack eng). 900 parallel items x
  24 languages, minus the ~32% whose question references the dropped MC options ("which
  of the following ..." -- unanswerable without them, and phrased like nothing in the
  test set; detected on the parallel English question). Per kept item, 10 question
  languages are drawn from the 23 test question languages belebele covers (all but bho),
  each row wrapped in the attested test layout
  `<lead-in>\\n\\n<passage>\\n\\n<question-intro><q>\\n\\n<tail>` with the boilerplate in
  the question language (constraint_bank mines it from tests.jsonl). The separator is the
  test file's LITERAL backslash-n pair -- matching what run_test.py feeds the model by
  default, per the train/infer-format-must-match finding (EXPERIMENTS.md 3861409).
  Context language: 4% monolingual (the test's 340/8640), otherwise drawn from the test's
  own qa-context context-language marginal (bho excluded: belebele has no Bhojpuri).
  Gold = the correct option's text in the question language (verbatim-in-passage on all
  sample rows; the MC scaffolding is dropped entirely -- the test has no MC). 7% of rows
  are made UNANSWERABLE by swapping in a same-language passage of a different item on the
  same split side; their gold is the language's exact attested refusal phrase.
  Item group = the flores passage URL (questions about one passage stay together).

* tydiqa -> qa-context, rebuilt from upstream copenlu/answerable_tydiqa (train+validation
  pools), which the sample drew only answerable rows from: per kept language 240
  answerable + 60 unanswerable rows (the test's "no answer" escape had ZERO training
  signal in the sample), same test-layout wrap, monolingual (context_lang =
  question_lang). tel/swh/tha are dropped (absent from the test set even as passage
  languages). Passages filtered to 20-200 words (chars/4 bounds for jpn) around the
  test's p50 63. Item group = union-find over (question text, document URL).

* MCIF -> qa portion becomes qa-context (context_lang eng), sum portion sum-sum; rows
  kept verbatim (its embedded instructions are already test-style). Item group = the
  TALK (transcript hash), shared across both portions -- all 21 QA talks are among the
  100 sum talks, so a dev talk is dev for both tasks.

* aya -> qa-oeg. The ~38 script-detectable cross-lingual rows (English question ->
  e.g. Hindi gold; they violate the test invariant answer-lang == question-lang) are
  dropped. Item group = exact input text (169 duplicated inputs, mostly multi-reference,
  stay together).

* OEG -> qa-oeg. Item group = the canonical item id from scripts/oeg_alignment.py
  (46 localized-parallel prompts x 10 languages; row order differs per language, so this
  needed a manual cross-language alignment).

* CrossSum / wiki_lingua -> sum-sum, schema-mapped only (the sum task belongs to the
  teammate). Item group = article-text hash; NOTE this cannot catch CrossSum's
  translated-parallel articles across languages -- flagged in DATA_AUDIT, needs the
  upstream pairing metadata if the sum split is ever load-bearing.

Split: dev fraction 0.20 of item groups. Hash-based (md5 of the group id) for the big
sources; explicit seeded draws for the two small ones where hash variance bites
(OEG: 9 of 46 items; MCIF: 4 of the 21 QA talks + 16 of the 79 sum-only talks).
Everything is seeded (--seed, default 42) and iteration order is sorted, so the build
is reproducible byte-for-byte.

    python scripts/build_dataset.py            # writes data/{train,dev}_v2.jsonl + report
    python scripts/build_dataset.py --selfcheck-only

Needs data/samples.jsonl and data/tests.jsonl (download commands in the README) plus the
two upstream pulls:

    for l in arb_Arab ben_Beng ces_Latn ckb_Arab deu_Latn eng_Latn fin_Latn fra_Latn \
             hat_Latn hin_Deva ind_Latn ita_Latn jpn_Jpan kor_Hang mar_Deva pes_Arab \
             por_Latn rus_Cyrl slk_Latn spa_Latn tur_Latn vie_Latn yor_Latn zho_Hans; do
      curl -sL "https://huggingface.co/datasets/facebook/belebele/resolve/main/data/$l.jsonl" \
           -o "data/upstream/belebele/$l.jsonl"
    done
    curl -sL "https://huggingface.co/datasets/copenlu/answerable_tydiqa/resolve/main/data/train-00000-of-00001-af2f3eaa87d1aa8b.parquet" \
         -o data/upstream/tydiqa/train.parquet
    curl -sL "https://huggingface.co/datasets/copenlu/answerable_tydiqa/resolve/main/data/validation-00000-of-00001-1f04eb244a33fa1b.parquet" \
         -o data/upstream/tydiqa/validation.parquet
"""

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import constraint_bank as cb

SEP = cb.SEP  # the test file's literal backslash-n backslash-n separator

BELEBELE_LANG = {
    "arb": "arb_Arab", "ben": "ben_Beng", "ces": "ces_Latn", "ckb": "ckb_Arab",
    "deu": "deu_Latn", "eng": "eng_Latn", "fin": "fin_Latn", "fra": "fra_Latn",
    "hat": "hat_Latn", "hin": "hin_Deva", "ind": "ind_Latn", "ita": "ita_Latn",
    "jpn": "jpn_Jpan", "kor": "kor_Hang", "mar": "mar_Deva", "pes": "pes_Arab",
    "por": "por_Latn", "rus": "rus_Cyrl", "slk": "slk_Latn", "spa": "spa_Latn",
    "tur": "tur_Latn", "vie": "vie_Latn", "yor": "yor_Latn", "zho": "zho_Hans",
}
BELEBELE_Q_LANGS = sorted(set(BELEBELE_LANG) - {"fra"})  # fra is passage-only at test time

TYDIQA_LANG = {"english": "eng", "arabic": "arb", "bengali": "ben", "finnish": "fin",
               "indonesian": "ind", "japanese": "jpn", "korean": "kor", "russian": "rus"}

DEV_FRAC = 0.20
BELEBELE_LANGS_PER_ITEM = 10     # -> ~270 rows per question language after the MC filter
BELEBELE_UNANSWERABLE = 0.07
BELEBELE_MONOLINGUAL = 0.04      # the test's 340/8640
TYDIQA_PER_LANG = (240, 60)      # (answerable, unanswerable)

# Script detection for the aya cross-lingual filter (drop English-question rows whose
# lang_code says the gold is e.g. Hindi). Latin-script languages are undetectable this
# way and pass through -- the audit's residual.
_SCRIPT_OF = {"Arab": "ARABIC", "Deva": "DEVANAGARI", "Hang": "HANGUL", "Jpan": None,
              "Hans": "CJK", "Cyrl": "CYRILLIC", "Beng": "BENGALI", "Latn": "LATIN"}


def bare(lang_code: str) -> str:
    return lang_code.split("_")[0]


def side_of(group: str) -> str:
    h = int(hashlib.md5(group.encode()).hexdigest(), 16) / 2**128
    return "dev" if h < DEV_FRAC else "train"


def wrap_qa_context(qlang: str, passage: str, question: str, test_file: str) -> str:
    """The attested test qa-context layout, boilerplate in the question language."""
    return SEP.join([cb.lead_in(qlang, test_file), passage.strip(),
                     cb.question_intro(qlang, test_file) + question.strip(),
                     cb.context_tail(qlang, test_file).full])


def _script_frac(text: str, script: str | None) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 1.0
    if script == "CJK":
        hit = sum(1 for c in letters if "CJK" in unicodedata.name(c, ""))
    elif script is None:  # jpn: kana or CJK
        hit = sum(1 for c in letters
                  if any(s in unicodedata.name(c, "") for s in ("CJK", "HIRAGANA", "KATAKANA")))
    else:
        hit = sum(1 for c in letters if unicodedata.name(c, "").startswith(script))
    return hit / len(letters)


# ---------------------------------------------------------------------------- belebele

def build_belebele(upstream_dir: Path, tests_file: str, rng: random.Random) -> list[dict]:
    items: dict[tuple, dict] = {}
    for code, fl in BELEBELE_LANG.items():
        for line in open(upstream_dir / f"{fl}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            items.setdefault((r["link"], r["question_number"]), {})[code] = r
    assert len(items) == 900, f"expected 900 belebele items, got {len(items)}"
    short = [k for k, v in items.items() if len(v) != len(BELEBELE_LANG)]
    assert not short, f"{len(short)} items missing languages, e.g. {short[:3]}"

    # Drop items whose question references the (dropped) MC options -- "which of the
    # following ...". Detected on the English question, which every language's version
    # parallels; ~32% of items. Without the options many are unanswerable (all the
    # "which would NOT ..." ones) and the phrasing never occurs at test time.
    mc_ref = re.compile(r"of the following|of these|following is|below", re.I)
    n0 = len(items)
    items = {k: v for k, v in items.items() if not mc_ref.search(v["eng"]["question"])}
    print(f"belebele: dropped {n0 - len(items)} of {n0} items with option-referencing "
          f"questions; {len(items)} kept")

    # context-language marginal measured from the test file itself (bho unavailable here)
    tests = [json.loads(l) for l in open(tests_file, encoding="utf-8")]
    marg = Counter(r["id"].split("_")[-1] for r in tests if r["task"] == "qa-context")
    marg.pop("bho", None)
    assert set(marg) <= set(BELEBELE_LANG), f"unknown test context langs: {set(marg) - set(BELEBELE_LANG)}"

    # explicit 20% draw at the passage (link) level -- hash-splitting 600-odd links
    # over-shoots the dev fraction by several points, and belebele is the biggest
    # qa-context source, so the waste is worth avoiding
    links = sorted({k[0] for k in items})
    dev_links = set(rng.sample(links, round(len(links) * DEV_FRAC)))
    by_side: dict[str, list] = {"dev": [], "train": []}
    for k in sorted(items):
        by_side["dev" if k[0] in dev_links else "train"].append(k)

    rows = []
    for k in sorted(items):
        link, _ = k
        group = f"belebele:{link}"
        side = "dev" if link in dev_links else "train"
        item = items[k]
        for q in rng.sample(BELEBELE_Q_LANGS, BELEBELE_LANGS_PER_ITEM):
            if rng.random() < BELEBELE_MONOLINGUAL:
                ctx = q
            else:
                cands = [(l, n) for l, n in sorted(marg.items()) if l != q]
                ctx = rng.choices([l for l, _ in cands], weights=[n for _, n in cands])[0]
            if rng.random() < BELEBELE_UNANSWERABLE:
                # a same-side passage of a DIFFERENT flores article makes the question
                # unanswerable; gold is the language's exact attested refusal phrase
                while True:
                    k2 = by_side[side][rng.randrange(len(by_side[side]))]
                    if k2[0] != link:
                        break
                passage = items[k2][ctx]["flores_passage"]
                out = cb.context_tail(q, tests_file).refusal_phrase
            else:
                passage = item[ctx]["flores_passage"]
                out = item[q][f"mc_answer{item[q]['correct_answer_num']}"]
            rows.append({
                "task": "qa-context", "question_lang": q, "context_lang": ctx,
                "source": "facebook/belebele",
                "input": wrap_qa_context(q, passage, item[q]["question"], tests_file),
                "output": out, "item_group": group, "_side": side,
            })
    return rows


# ------------------------------------------------------------------------------ tydiqa

def build_tydiqa(upstream_dir: Path, tests_file: str, rng: random.Random) -> list[dict]:
    import pandas as pd
    df = pd.concat([pd.read_parquet(upstream_dir / "train.parquet"),
                    pd.read_parquet(upstream_dir / "validation.parquet")], ignore_index=True)
    df = df[df["language"].isin(TYDIQA_LANG)].copy()
    df["lang"] = df["language"].map(TYDIQA_LANG)
    df["start"] = df["annotations"].map(lambda a: int(a["answer_start"][0]))
    df["answer"] = df["annotations"].map(lambda a: str(a["answer_text"][0]))
    df = df.drop_duplicates(subset=["question_text", "document_plaintext"])

    def passage_ok(row) -> bool:
        p = row["document_plaintext"].strip()
        if row["lang"] == "jpn":  # no spaces; test-passage-equivalent band in characters
            return 40 <= cb.measure(p, "jpn") <= 500
        return 20 <= len(p.split()) <= 200

    df = df[df.apply(passage_ok, axis=1)]

    # union-find: rows sharing a question OR a document URL are one item group
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str):
        parent[find(a)] = find(b)

    for _, r in df.iterrows():
        union(f"q:{r['question_text']}", f"d:{r['document_url']}")

    rows = []
    for lang in sorted(set(TYDIQA_LANG.values())):
        sub = df[df["lang"] == lang]
        for pool, n_want, answerable in (
                (sub[sub["start"] >= 0], TYDIQA_PER_LANG[0], True),
                (sub[sub["start"] < 0], TYDIQA_PER_LANG[1], False)):
            recs = sorted(pool.to_dict("records"),
                          key=lambda r: hashlib.md5(
                              (r["question_text"] + r["document_url"]).encode()).hexdigest())
            for r in rng.sample(recs, min(n_want, len(recs))):
                group = "tydiqa:" + hashlib.md5(find(f"q:{r['question_text']}").encode()).hexdigest()[:16]
                rows.append({
                    "task": "qa-context", "question_lang": lang, "context_lang": lang,
                    "source": "copenlu/answerable_tydiqa",
                    "input": wrap_qa_context(lang, r["document_plaintext"],
                                             r["question_text"], tests_file),
                    "output": r["answer"] if answerable
                              else cb.context_tail(lang, tests_file).refusal_phrase,
                    "item_group": group,
                })
    return rows


# ---------------------------------------------------------------- sample-derived sources

def talk_key(input_text: str) -> str:
    """MCIF talk id: hash of the transcript (the text after the first blank line)."""
    return hashlib.md5(input_text.split("\n\n", 1)[-1][:300].encode()).hexdigest()[:16]


def build_from_samples(samples_file: str, rng: random.Random) -> tuple[list[dict], dict]:
    samples = [json.loads(l) for l in open(samples_file, encoding="utf-8")]
    rows, dropped = [], Counter()

    # --- MCIF: explicit talk-level split so the 21 QA talks are fairly represented
    mcif = [r for r in samples if r["source"] == "FBK-MT/MCIF"]
    qa_talks = sorted({talk_key(r["input"]) for r in mcif if r["task"] == "qa"})
    sum_only = sorted({talk_key(r["input"]) for r in mcif if r["task"] == "sum"} - set(qa_talks))
    dev_talks = set(rng.sample(qa_talks, round(len(qa_talks) * DEV_FRAC))) | \
                set(rng.sample(sum_only, round(len(sum_only) * DEV_FRAC)))
    for r in mcif:
        tk = talk_key(r["input"])
        rows.append({
            "task": "qa-context" if r["task"] == "qa" else "sum-sum",
            "question_lang": bare(r["lang_code"]),
            "context_lang": "eng",
            "source": r["source"], "input": r["input"], "output": r["output"],
            "item_group": f"mcif:{tk}",
            "_side": "dev" if tk in dev_talks else "train",
        })

    # --- aya: drop script-detectable cross-lingual rows; group by exact input
    for r in (r for r in samples if r["source"] == "CohereLabs/aya_dataset"):
        script = _SCRIPT_OF[r["lang_code"].split("_")[1]]
        if script != "LATIN" and _script_frac(r["input"], script) < 0.3 \
                and _script_frac(r["output"], script) > 0.5:
            dropped[f"aya cross-lingual ({r['lang_code']})"] += 1
            continue
        group = "aya:" + hashlib.md5(r["input"].strip().encode()).hexdigest()[:16]
        rows.append({"task": "qa-oeg", "question_lang": bare(r["lang_code"]),
                     "context_lang": None, "source": r["source"],
                     "input": r["input"], "output": r["output"], "item_group": group})

    # --- OEG: canonical item ids from the manual alignment; explicit 9/46 dev items
    from oeg_alignment import ALIGNMENT
    dev_items = set(rng.sample(range(46), round(46 * DEV_FRAC)))
    idx_within = Counter()
    for r in (r for r in samples if r["source"] == "wmt25-mist-oeg-gpt-4.1"):
        item = ALIGNMENT[r["lang_code"]][idx_within[r["lang_code"]]]
        idx_within[r["lang_code"]] += 1
        rows.append({"task": "qa-oeg", "question_lang": bare(r["lang_code"]),
                     "context_lang": None, "source": r["source"],
                     "input": r["input"], "output": r["output"],
                     "item_group": f"oeg:{item:02d}",
                     "_side": "dev" if item in dev_items else "train"})

    # --- sum-only sources: schema-mapped, article-hash groups
    for r in (r for r in samples if r["source"] == "csebuetnlp/CrossSum"):
        group = "crosssum:" + hashlib.md5(
            r["input"].split("\n\n", 1)[-1].strip().encode()).hexdigest()[:16]
        rows.append({"task": "sum-sum", "question_lang": bare(r["lang_code"]),
                     "context_lang": None, "source": r["source"],
                     "input": r["input"], "output": r["output"], "item_group": group})
    for r in (r for r in samples if r["source"] == "esdurmus/wiki_lingua"):
        group = "wikilingua:" + hashlib.md5(r["input"].strip().encode()).hexdigest()[:16]
        rows.append({"task": "sum-sum", "question_lang": bare(r["lang_code"]),
                     "context_lang": bare(r["lang_code"]), "source": r["source"],
                     "input": r["input"], "output": r["output"], "item_group": group})

    return rows, dropped


# ------------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--samples", default="data/samples.jsonl")
    ap.add_argument("--tests", default="data/tests.jsonl")
    ap.add_argument("--upstream", default="data/upstream")
    ap.add_argument("--out-train", default="data/train_v2.jsonl")
    ap.add_argument("--out-dev", default="data/dev_v2.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selfcheck-only", action="store_true",
                    help="build in memory, run the checks and report, write nothing")
    args = ap.parse_args()
    for f in (args.samples, args.tests):
        if not Path(f).exists():
            sys.exit(f"{f} not found (see README for the download command)")

    cb.DEFAULT_TEST_FILE = args.tests
    rng = random.Random(args.seed)
    rows = build_belebele(Path(args.upstream) / "belebele", args.tests, rng)
    rows += build_tydiqa(Path(args.upstream) / "tydiqa", args.tests, rng)
    sample_rows, dropped = build_from_samples(args.samples, rng)
    rows += sample_rows

    for r in rows:  # hash-split everything that has no explicit side yet
        r.setdefault("_side", side_of(r["item_group"]))

    # ---- self-checks ----------------------------------------------------------------
    sides_of_group = defaultdict(set)
    for r in rows:
        sides_of_group[r["item_group"]].add(r["_side"])
    leaks = {g for g, s in sides_of_group.items() if len(s) > 1}
    assert not leaks, f"item groups on both sides: {sorted(leaks)[:5]}"

    for r in rows:  # every synthesized qa-context row must parse like a test prompt
        if r["source"] in ("facebook/belebele", "copenlu/answerable_tydiqa"):
            assert len(r["input"].split(SEP)) == 4, f"bad layout: {r['input'][:80]}"
            assert "\n" not in r["input"].split(SEP)[0]
    print(f"self-checks passed: {len(sides_of_group)} item groups, no dev/train overlap; "
          f"synthesized qa-context rows parse as 4 literal-\\n\\n segments")

    # ---- report ---------------------------------------------------------------------
    for what, n in sorted(dropped.items()):
        print(f"dropped: {what}: {n}")
    tab = Counter((r["task"], r["source"], r["_side"]) for r in rows)
    print(f"\n{'task':11s} {'source':28s} {'train':>6s} {'dev':>5s}")
    for task, src in sorted({(t, s) for t, s, _ in tab}):
        print(f"{task:11s} {src:28s} {tab[(task, src, 'train')]:6d} {tab[(task, src, 'dev')]:5d}")
    n_dev = sum(1 for r in rows if r["_side"] == "dev")
    print(f"{'TOTAL':40s} {len(rows) - n_dev:6d} {n_dev:5d}")
    qlangs = Counter((r["task"], r["question_lang"]) for r in rows)
    for task in ("qa-context", "qa-oeg", "sum-sum"):
        langs = sorted((l, n) for (t, l), n in qlangs.items() if t == task)
        print(f"\n{task} question langs ({len(langs)}): "
              + " ".join(f"{l}={n}" for l, n in langs))

    if args.selfcheck_only:
        print("\n--selfcheck-only: nothing written")
        return
    for path, side in ((args.out_train, "train"), (args.out_dev, "dev")):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                if r["_side"] == side:
                    f.write(json.dumps({k: v for k, v in r.items() if k != "_side"},
                                       ensure_ascii=False) + "\n")
        print(f"wrote {sum(1 for r in rows if r['_side'] == side)} rows -> {path}")


if __name__ == "__main__":
    main()
