"""Summarise a predictions CSV from benchmark.py: overall chrF, plus per-source and
per-language breakdowns.

Cheap (CPU, seconds) and independent of generation -- run it on the login node, re-run it to
add metrics, or point it at a *partial* CSV to score an interrupted job without re-generating.

    python evaluate.py predictions.csv
"""

import argparse

import pandas as pd
import sacrebleu


def chrf(preds, refs) -> float:
    """Corpus chrF (0-100) for parallel lists of predictions and references."""
    return sacrebleu.corpus_chrf(list(preds), [list(refs)]).score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_csv", help="CSV from benchmark.py "
                                     "(columns: source, lang_code, input, gold, prediction)")
    args = ap.parse_args()

    df = pd.read_csv(args.pred_csv, encoding="utf-8-sig")  # strip the BOM benchmark.py writes
    # Failed/empty generations come back as NaN from the CSV; treat them as empty strings.
    df["prediction"] = df["prediction"].fillna("")
    df["gold"] = df["gold"].fillna("")

    print(f"file: {args.pred_csv}")
    print(f"n = {len(df)}")
    print(f"overall chrF = {chrf(df['prediction'], df['gold']):.2f}\n")

    print("chrF by source:")
    for source, g in df.groupby("source"):
        print(f"  {source:35s} n={len(g):5d}  chrF={chrf(g['prediction'], g['gold']):6.2f}")

    print("\nchrF by language (best to worst):")
    rows = [(lang, len(g), chrf(g["prediction"], g["gold"]))
            for lang, g in df.groupby("lang_code")]
    for lang, n, score in sorted(rows, key=lambda r: r[2], reverse=True):
        print(f"  {lang:12s} n={n:5d}  chrF={score:6.2f}")


if __name__ == "__main__":
    main()
