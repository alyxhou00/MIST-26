"""Bhojpuri-vs-neighbours language ID, by function-word markers.

The base 9B does not produce Bhojpuri: the smoke run (job 3859059) drifted 10/10 `bho`
qa-oeg rows into standard Hindi, with one output in Nepali and one in Maithili. Every part
of the Bhojpuri kit (roadmap D) therefore needs to answer "is this string actually
Bhojpuri?" -- to filter scraped training text (`build_bho_pack.py`), and later to gate
generation (roadmap F).

The four languages are close relatives sharing the Devanagari script, so script detection is
useless and character n-grams are weak. Function words separate them cleanly instead:

    bho  बा / नइखे / एगो / रउआ / बाकिर / खातिर      ("is" = बा)
    hin  है / हैं / और / नहीं / हुआ                  ("is" = है)
    mai  अछि / छलाह / करैत / मुदा / सँ               ("is" = अछि)
    npi  छ / छन् / गर्छ / भएको / तर                  ("is" = छ)

`classify` scores marker density per language and requires the winner to beat every rival by
`margin`x, returning None when nothing wins clearly -- abstaining is the right move for a
filter whose failure mode (Hindi labelled as Bhojpuri) would actively *teach* the drift we
are trying to remove.

Measured with `--eval` on sib200 (FLORES-derived, 400 parallel sentences per language), by
document length:

    1 sentence    bho recall 65.0%   precision  98.9%   4-way acc 71.1%
    3 sentences   bho recall 91.0%   precision 100.0%   4-way acc 94.2%
    5 sentences   bho recall 95.0%   precision 100.0%   4-way acc 98.1%

So it is precision-safe on anything paragraph-sized (fineweb-2 documents), and should not be
trusted on single short sentences. Recall costs us some genuine Bhojpuri; that trade is
correct here, because we are selecting from a corpus rather than trying to keep every row.

This is deliberately a stopgap: **GlotLID** (`cis-lmu/glotlid`, covers `bho_Deva` properly)
is the real tool and is what roadmap F should gate generation with. This module exists
because it needs no model download, no new dependency and no GPU, and because a table of
function words is auditable in a way a 900MB classifier is not.

    python scripts/bho_lid.py --eval        # reproduce the table above (needs network)
"""

import argparse
import json
import re
import sys
import urllib.request
from collections import Counter

# Function words that are distinctive of ONE of the four languages. Chosen to be words a
# speaker of the language uses constantly (copulas, conjunctions, pronouns, participles) and
# that the other three do not share -- a marker that appears in two of these languages is
# worse than no marker, because it pushes `classify` toward abstaining rather than deciding.
MARKERS = {
    "bho": ["बा", "बाड़", "बाड़ी", "बाड़े", "नइखे", "करेला", "करेलें", "रहल", "एगो", "रउआ",
            "रउरा", "हमनी", "जवन", "बाकिर", "होखे", "भइल", "कइल", "सकेला", "चलेला", "दिहल",
            "गइल", "लागल", "करत", "खातिर", "ओकर", "इहाँ", "केहू", "काहे"],
    "hin": ["है", "हैं", "था", "थे", "थी", "और", "नहीं", "करता", "करते", "हुआ", "गया",
            "रहा", "रही", "किया", "लिए", "साथ", "बहुत", "यह", "वह", "कोई", "क्यों"],
    "mai": ["अछि", "छल", "छलाह", "छथि", "करैत", "मुदा", "एवं", "सँ", "केर", "भेल", "गेल",
            "रहैत", "अपन", "ओ", "एहि", "कऽ"],
    "npi": ["छ", "छन्", "गर्छ", "गर्थे", "भएको", "गर्ने", "तर", "मा", "हुन्", "गरे",
            "भने", "यो", "त्यो", "हो", "गरेको", "पनि", "लाई"],
}

# Devanagari runs. Splitting on non-Devanagari keeps Latin/digits out of the denominator, so
# marker density is measured over the actual Indic text.
_TOK = re.compile(r"[ऀ-ॿ]+")

DEFAULT_MARGIN = 1.25
MIN_TOKENS = 12  # below this, densities are too noisy to be worth trusting (see --eval)


def densities(text: str) -> dict[str, float]:
    """Marker density per language: share of Devanagari tokens that are that language's
    markers. Densities do not sum to 1 -- most tokens are content words matching nothing."""
    toks = _TOK.findall(text)
    if not toks:
        return {}
    return {lang: sum(t in set(ms) for t in toks) / len(toks) for lang, ms in MARKERS.items()}


def classify(text: str, margin: float = DEFAULT_MARGIN,
             min_tokens: int = MIN_TOKENS) -> str | None:
    """Best-matching language, or None when no language wins by `margin`x (or the text is
    too short to judge)."""
    if len(_TOK.findall(text)) < min_tokens:
        return None
    d = densities(text)
    if not d or max(d.values()) == 0:
        return None
    best = max(d, key=d.get)
    rival = max(v for k, v in d.items() if k != best)
    return best if d[best] >= margin * rival else None


def is_bhojpuri(text: str, margin: float = DEFAULT_MARGIN,
                min_tokens: int = MIN_TOKENS) -> bool:
    return classify(text, margin, min_tokens) == "bho"


# --------------------------------------------------------------------------------------
# evaluation against sib200 (Davlan/sib200): 400 parallel sentences in each of the 4 langs
# --------------------------------------------------------------------------------------

def _sib200(config: str, n: int = 400) -> list[str]:
    out, off = [], 0
    while len(out) < n:
        url = ("https://datasets-server.huggingface.co/rows?dataset=Davlan/sib200"
               f"&config={config}&split=train&offset={off}&length=100")
        with urllib.request.urlopen(url) as r:
            got = json.loads(r.read().decode("utf-8")).get("rows", [])
        if not got:
            break
        out += [g["row"]["text"] for g in got]
        off += len(got)
    return out[:n]


def evaluate(margin: float) -> int:
    langs = ["bho", "hin", "mai", "npi"]
    try:
        corpus = {l: _sib200(f"{l}_Deva") for l in langs}
    except Exception as e:  # noqa: BLE001
        print(f"could not fetch sib200: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    print(f"sib200 sentences: { {l: len(t) for l, t in corpus.items()} } | margin={margin}")

    for doc_sents in (1, 3, 5):
        conf = Counter()
        for true in langs:
            texts = corpus[true]
            docs = [" ".join(texts[i:i + doc_sents]) for i in range(0, len(texts), doc_sents)]
            for d in docs:
                conf[(true, classify(d, margin))] += 1
        cols = langs + [None]
        tot = sum(conf.values())
        acc = sum(conf[(l, l)] for l in langs) / tot
        tp = conf[("bho", "bho")]
        fp = sum(conf[(t, "bho")] for t in langs if t != "bho")
        n_bho = sum(conf[("bho", c)] for c in cols)
        print(f"\n=== documents of {doc_sents} sentence(s) === 4-way accuracy {acc:.1%}")
        print("      " + "".join(f"{str(c):>6s}" for c in cols) + "   (rows=true)")
        for t in langs:
            print(f"{t:6s}" + "".join(f"{conf[(t, c)]:>6d}" for c in cols))
        print(f"  bho recall {tp / n_bho:.1%}  precision "
              f"{tp / (tp + fp) if tp + fp else 1:.1%}  (false positives {fp})")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eval", action="store_true",
                    help="score against sib200 bho/hin/mai/npi and print confusion matrices")
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    ap.add_argument("--text", help="classify one string and exit")
    a = ap.parse_args()
    if a.text:
        print(f"{classify(a.text, a.margin)}  {densities(a.text)}")
    elif a.eval:
        sys.exit(evaluate(a.margin))
    else:
        ap.error("pass --eval or --text")
