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

    1 sentence    bho recall  73.2%   precision  98.3%   4-way acc 73.0%
    3 sentences   bho recall  94.0%   precision  99.2%   4-way acc 94.2%
    5 sentences   bho recall 100.0%   precision 100.0%   4-way acc 97.8%

So: usable on anything paragraph-sized, not to be trusted on single short sentences.

**Read those numbers narrowly.** sib200 is FLORES news prose, and they do NOT transfer to
other registers -- an earlier version of this module scored 91%/100% there while confidently
mislabelling genuine Bhojpuri *web* text as Nepali, because the marker list had been built
from that same clean register and had no coverage of the everyday -ela/-ala verb forms
(होला, जाला, होवेला). Treat --eval as a regression test, not as a general accuracy claim;
judge any new corpus by sampling its rejects (`--text`), as `build_bho_pack.py --report`
invites you to.

On the corpus that actually matters, fineweb-2's bho_Deva (2,929 documents >=300 chars):
96.2% classify as bho, 2.8% abstain, and only ~1% come back as hin/npi/mai -- i.e. that
subset is largely clean, and this gate earns its place mainly by abstaining on the margins
rather than by catching mass contamination.

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
    # The -ela/-ala habitual verb endings (होला "is", जाला "goes", करेला "does") are the
    # single most reliable Bhojpuri signal in running text, and were missing from the first
    # version of this list -- which was built only from sib200's FLORES news register. On
    # fineweb's web register that gap left genuinely Bhojpuri documents scoring bho=0.0, so
    # they were classified off whatever stray Hindi word they happened to contain.
    "bho": ["बा", "बाड़", "बाड़ी", "बाड़े", "नइखे", "करेला", "करेलें", "रहल", "एगो", "रउआ",
            "रउरा", "हमनी", "जवन", "बाकिर", "होखे", "भइल", "कइल", "सकेला", "चलेला", "दिहल",
            "गइल", "लागल", "करत", "खातिर", "ओकर", "इहाँ", "केहू", "काहे",
            "होला", "जाला", "होखेला", "होवेला", "रहेला", "देला", "लेला", "मिलेला",
            "जायेला", "लागेला", "फरेला", "कहल", "मनावल", "पावल", "होइल", "कहेला",
            "बानी", "बाटे", "हवे", "हउवे", "जवना", "जौन", "अउरी", "बड़हन"],
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

# A `margin` alone is not enough: "best >= 1.25 * rival" is VACUOUSLY true whenever the
# rivals score 0.0, so a single stray token used to win outright. On fineweb web text that
# produced confident nonsense -- genuine Bhojpuri ("आम ... के पेड़ पर फरेला") came back as
# `npi` off one spurious marker, with densities {bho 0.0, hin 0.0, mai 0.0, npi 0.016}.
# So the winner must also clear an ABSOLUTE floor. Calibrated on fineweb: documents that are
# really Bhojpuri carry a marker density of p05=0.021 / p50=0.058, while noise verdicts sit
# near 0.01, so 0.015 separates them without touching genuine matches.
MIN_DENSITY = 0.015


def densities(text: str) -> dict[str, float]:
    """Marker density per language: share of Devanagari tokens that are that language's
    markers. Densities do not sum to 1 -- most tokens are content words matching nothing."""
    toks = _TOK.findall(text)
    if not toks:
        return {}
    return {lang: sum(t in set(ms) for t in toks) / len(toks) for lang, ms in MARKERS.items()}


def classify(text: str, margin: float = DEFAULT_MARGIN, min_tokens: int = MIN_TOKENS,
             min_density: float = MIN_DENSITY) -> str | None:
    """Best-matching language, or None when the evidence is too thin or too close.

    Abstains (returns None) unless the winner both clears `min_density` in absolute terms
    AND beats every rival by `margin`x. Both conditions matter: the margin catches genuine
    ambiguity between two languages, the floor catches the far more common case of no real
    evidence at all (see MIN_DENSITY).
    """
    if len(_TOK.findall(text)) < min_tokens:
        return None
    d = densities(text)
    if not d:
        return None
    best = max(d, key=d.get)
    if d[best] < min_density:
        return None
    rival = max(v for k, v in d.items() if k != best)
    return best if d[best] >= margin * rival else None


def is_bhojpuri(text: str, margin: float = DEFAULT_MARGIN, min_tokens: int = MIN_TOKENS,
                min_density: float = MIN_DENSITY) -> bool:
    return classify(text, margin, min_tokens, min_density) == "bho"


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
