"""Cross-language item alignment for the OEG source (wmt25-mist-oeg-gpt-4.1).

OEG is a parallel corpus -- the same 46 prompts localized into 10 languages -- but the
row ORDER differs per language (verified 2026-07-17: index 30 is "holiday" in eng,
"school system" in deu, "time zones" in zho), so grouping by row index would leave the
dev/train leakage the item-split rebuild exists to fix. There is no upstream id to join
on and no digit signature that survives localization (the prompts localize entities,
digits and even the numbers themselves: the 13-line poem is 10 verses in Czech, the
50-word story is 125 characters in Japanese).

The mapping below was therefore produced by READING all 460 prompts and aligning each
language's 46 rows to the English list (session of 2026-07-17). `ALIGNMENT[lang][i]` is
the canonical item id (= English row index) of that language's i-th OEG row, where rows
are numbered in samples.jsonl order within the language. Every entry is asserted to be a
permutation of 0..45 at import time, and `--selftest` cross-checks the two prompts whose
word budgets survive localization in every language: item 32 (the "300-word article on
political parties") must contain a rendered 300 and item 43 (the "under-200-words job
posting") a rendered 200, in the language's own digit family.
"""

import argparse
import json
import sys

# {sample lang_code: canonical item id of the language's i-th row}. Canonical id =
# English row index (eng is the identity). Localization anchors used during alignment,
# for the reader's spot-checking: item 7 (river monologue) is the Vltava in ces, the
# Volga in rus, the Citarum in ind and the Kamo (鴨川) in jpn; item 10 (five fun facts)
# is the Grand Canyon in eng, the Schwarzwald in deu, Karel IV in ces, the matryoshka in
# rus, the Taj Mahal in hin, the Giza pyramids in arb, Borobudur in ind, sushi in jpn,
# the Great Wall in zho and panta ilish in ben.
ALIGNMENT = {
    "eng_Latn": list(range(46)),
    "deu_Latn": [20, 44, 36, 43, 25, 9, 12, 41, 38, 32, 23, 28, 39, 13, 2, 6, 17, 4,
                 37, 18, 8, 21, 11, 30, 24, 40, 31, 0, 27, 15, 34, 19, 5, 35, 42, 16,
                 22, 10, 33, 45, 3, 1, 29, 26, 7, 14],
    "ces_Latn": [32, 30, 43, 14, 26, 21, 20, 44, 45, 39, 5, 3, 35, 17, 27, 33, 42, 11,
                 38, 41, 37, 24, 16, 31, 40, 12, 36, 13, 2, 23, 18, 19, 29, 9, 6, 34,
                 8, 1, 25, 10, 15, 28, 4, 7, 22, 0],
    "zho_Hans": [34, 20, 16, 5, 18, 21, 2, 32, 28, 31, 30, 23, 17, 8, 37, 26, 39, 7,
                 33, 3, 13, 10, 12, 36, 45, 6, 1, 15, 27, 44, 22, 9, 4, 14, 11, 43,
                 38, 40, 19, 25, 24, 0, 41, 35, 42, 29],
    "rus_Cyrl": [13, 31, 34, 23, 22, 37, 43, 41, 26, 40, 16, 30, 8, 1, 9, 12, 4, 19,
                 32, 38, 45, 10, 18, 24, 36, 33, 29, 0, 14, 28, 3, 15, 25, 17, 21, 27,
                 2, 20, 44, 7, 11, 35, 6, 5, 39, 42],
    "arb_Arab": [2, 42, 43, 40, 41, 23, 19, 22, 37, 35, 44, 32, 5, 14, 6, 25, 31, 28,
                 20, 27, 11, 9, 15, 38, 36, 21, 45, 8, 24, 4, 34, 30, 7, 0, 12, 1,
                 16, 39, 26, 17, 13, 33, 10, 18, 29, 3],
    "ben_Beng": [4, 3, 6, 42, 25, 29, 19, 31, 33, 38, 41, 23, 5, 17, 13, 45, 36, 1,
                 35, 2, 8, 15, 28, 24, 44, 26, 21, 37, 7, 12, 16, 40, 14, 20, 34, 43,
                 30, 18, 9, 0, 39, 10, 27, 32, 22, 11],
    "hin_Deva": [22, 4, 8, 29, 15, 37, 20, 44, 41, 1, 33, 18, 21, 10, 2, 9, 38, 27,
                 19, 13, 3, 43, 39, 32, 26, 34, 23, 12, 11, 36, 5, 0, 45, 6, 35, 14,
                 16, 42, 28, 7, 30, 17, 24, 40, 31, 25],
    "ind_Latn": [35, 3, 15, 10, 39, 27, 19, 28, 7, 21, 45, 42, 4, 17, 41, 12, 30, 43,
                 14, 0, 25, 36, 34, 9, 40, 6, 29, 33, 1, 8, 32, 44, 37, 31, 13, 38,
                 11, 22, 26, 2, 5, 24, 23, 16, 20, 18],
    "jpn_Jpan": [45, 23, 5, 2, 11, 17, 6, 28, 24, 16, 9, 18, 3, 27, 33, 14, 32, 44,
                 21, 1, 42, 25, 39, 12, 19, 30, 41, 0, 40, 15, 29, 37, 7, 35, 4, 34,
                 22, 38, 10, 26, 13, 36, 20, 8, 43, 31],
}

for _lang, _m in ALIGNMENT.items():
    assert sorted(_m) == list(range(46)), f"{_lang}: not a permutation of 0..45"


def item_id(lang_code: str, row_index: int) -> int:
    """Canonical OEG item id for the `row_index`-th row of `lang_code` (samples order)."""
    return ALIGNMENT[lang_code][row_index]


def selftest(samples_file: str) -> int:
    from collections import defaultdict
    rows = [json.loads(l) for l in open(samples_file, encoding="utf-8")]
    oeg = defaultdict(list)
    for r in rows:
        if r["source"] == "wmt25-mist-oeg-gpt-4.1":
            oeg[r["lang_code"]].append(r)
    fails = 0
    if set(oeg) != set(ALIGNMENT):
        print(f"FAIL: language sets differ: {sorted(set(oeg) ^ set(ALIGNMENT))}")
        return 1
    # Budgets that survive localization everywhere: 300 (item 32), 200 (item 43) --
    # rendered in the language's own digit family (Bengali writes ৩০০).
    digit_forms = {300: ["300", "৩০০"], 200: ["200", "২০০"]}
    for item, n in ((32, 300), (43, 200)):
        for lang, rs in sorted(oeg.items()):
            idx = ALIGNMENT[lang].index(item)
            hit = any(f in rs[idx]["input"] for f in digit_forms[n])
            fails += not hit
            print(f"  {'ok  ' if hit else 'FAIL'} {lang} row {idx:2d} = item {item}: "
                  f"expects a rendered {n}{'' if hit else '  <-- not found'}")
    print("FAILED" if fails else "all checks passed")
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--samples-file", default="data/samples.jsonl")
    a = ap.parse_args()
    if not a.selftest:
        ap.error("nothing to do; pass --selftest")
    sys.exit(selftest(a.samples_file))
