"""Lightweight error analysis for a predictions CSV from benchmark.py.

Flags two cheap-to-detect failure modes that disproportionately hurt chrF:
  - script mismatch: gold is in a non-Latin script but prediction is mostly Latin/ASCII
  - length mismatch: prediction is much longer than gold (rambling/explaining instead
    of giving the short extractive-style answer)

Prints per-language rates plus a few example rows per language so you can eyeball
the actual failure mode, not just the chrF number.

    python scripts/error_analysis.py predictions/predictions-<jobid>.csv
"""

import argparse
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

NON_LATIN_RANGES = [
    (0x0600, 0x06FF),  # Arabic
    (0x0900, 0x097F),  # Devanagari
    (0x0980, 0x09FF),  # Bengali
    (0x0E00, 0x0E7F),  # Thai
    (0x0C00, 0x0C7F),  # Telugu
    (0x3040, 0x30FF),  # Hiragana/Katakana
    (0x3400, 0x9FFF),  # CJK
    (0xAC00, 0xD7A3),  # Hangul
]


def script_of(ch: str) -> str:
    cp = ord(ch)
    for lo, hi in NON_LATIN_RANGES:
        if lo <= cp <= hi:
            return "non_latin"
    if ch.isalpha():
        return "latin" if cp < 0x0250 else "other"
    return "other"


def non_latin_frac(text: str) -> float:
    letters = [c for c in str(text) if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if script_of(c) == "non_latin") / len(letters)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_csv")
    ap.add_argument("--n-examples", type=int, default=3, help="example rows to print per language")
    args = ap.parse_args()

    df = pd.read_csv(args.pred_csv, encoding="utf-8-sig")
    df["prediction"] = df["prediction"].fillna("")
    df["gold"] = df["gold"].fillna("")

    df["gold_non_latin"] = df["gold"].apply(non_latin_frac)
    df["pred_non_latin"] = df["prediction"].apply(non_latin_frac)
    # gold expects non-Latin script but prediction came back mostly Latin/ASCII
    df["script_mismatch"] = (df["gold_non_latin"] > 0.5) & (df["pred_non_latin"] < 0.2)

    df["gold_len"] = df["gold"].str.len()
    df["pred_len"] = df["prediction"].str.len()
    # prediction much longer than gold (rambling instead of a short extractive answer)
    df["len_mismatch"] = (df["pred_len"] > 3 * df["gold_len"] + 20)

    print(f"file: {args.pred_csv}\nn = {len(df)}\n")
    print(f"{'lang':12s} {'n':>5s} {'script_mismatch%':>16s} {'len_mismatch%':>14s}")
    rows = []
    for lang, g in df.groupby("lang_code"):
        sm = g["script_mismatch"].mean() * 100
        lm = g["len_mismatch"].mean() * 100
        rows.append((lang, len(g), sm, lm))
    for lang, n, sm, lm in sorted(rows, key=lambda r: -(r[2] + r[3])):
        print(f"{lang:12s} {n:5d} {sm:16.1f} {lm:14.1f}")

    print("\n--- sample rows: top offenders ---")
    worst = df.sort_values(["script_mismatch", "len_mismatch"], ascending=False)
    shown = 0
    for _, row in worst.iterrows():
        if not (row["script_mismatch"] or row["len_mismatch"]):
            break
        if shown >= args.n_examples * 5:
            break
        tag = []
        if row["script_mismatch"]:
            tag.append("SCRIPT")
        if row["len_mismatch"]:
            tag.append("LEN")
        print(f"\n[{'/'.join(tag)}] lang={row['lang_code']} source={row['source']}")
        print(f"  gold: {row['gold'][:200]!r}")
        print(f"  pred: {row['prediction'][:200]!r}")
        shown += 1


if __name__ == "__main__":
    main()
