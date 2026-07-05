"""Summarise a predictions CSV from benchmark.py: overall chrF/BERTScore/ROUGE-L, plus
per-source and per-language breakdowns.

chrF is cheap (CPU, seconds). BERTScore additionally needs a transformer forward pass over
every row (a GPU helps but CPU works for dev-set sizes); it's loaded once and reused for the
overall and per-group numbers. Independent of generation -- run it on the login node, re-run
it to add metrics, or point it at a *partial* CSV to score an interrupted job without
re-generating.

    python scripts/evaluate.py predictions/predictions-<jobid>.csv
"""

import argparse

import pandas as pd
import sacrebleu
from bert_score import BERTScorer
from rouge_score import rouge_scorer


def chrf(preds, refs) -> float:
    """Corpus chrF (0-100) for parallel lists of predictions and references."""
    return sacrebleu.corpus_chrf(list(preds), [list(refs)]).score


def rouge_l(preds, refs) -> pd.Series:
    """Per-row ROUGE-L F-measure (0-100). No stemmer: text spans many non-English scripts."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return pd.Series(
        [scorer.score(ref, pred)["rougeL"].fmeasure * 100 for pred, ref in zip(preds, refs)]
    )


def bertscore_f1(scorer: BERTScorer, preds, refs) -> pd.Series:
    """Per-row BERTScore F1 (0-100). Empty predictions (failed generations) score 0 without
    going through bert_score's own scoring path, which crashes on empty strings under the
    transformers version this repo pins (its `build_inputs_with_special_tokens` fallback for
    empty input was removed in transformers 5)."""
    preds, refs = pd.Series(preds).reset_index(drop=True), pd.Series(refs).reset_index(drop=True)
    non_empty = preds.str.strip() != ""
    result = pd.Series(0.0, index=preds.index)
    if non_empty.any():
        _, _, f1 = scorer.score(preds[non_empty].tolist(), refs[non_empty].tolist())
        result[non_empty] = f1.numpy() * 100
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_csv", help="CSV from benchmark.py "
                                     "(columns: source, lang_code, input, gold, prediction)")
    ap.add_argument("--bertscore-model", default="bert-base-multilingual-cased",
                     help="BERTScore backbone; must cover all 27 task languages")
    args = ap.parse_args()

    df = pd.read_csv(args.pred_csv, encoding="utf-8-sig")  # strip the BOM benchmark.py writes
    # Failed/empty generations come back as NaN from the CSV; treat them as empty strings.
    df["prediction"] = df["prediction"].fillna("")
    df["gold"] = df["gold"].fillna("")

    # Per-row BERTScore/ROUGE-L computed once up front, then averaged per group below --
    # cheaper than re-scoring each subset (chrF is corpus-level, so it's recomputed per group).
    scorer = BERTScorer(model_type=args.bertscore_model)
    df["bertscore_f1"] = bertscore_f1(scorer, df["prediction"], df["gold"])
    df["rouge_l_f1"] = rouge_l(df["prediction"], df["gold"])

    print(f"file: {args.pred_csv}")
    print(f"n = {len(df)}")
    print(f"overall chrF      = {chrf(df['prediction'], df['gold']):.2f}")
    print(f"overall BERTScore = {df['bertscore_f1'].mean():.2f}")
    print(f"overall ROUGE-L   = {df['rouge_l_f1'].mean():.2f}\n")

    print("by source:")
    for source, g in df.groupby("source"):
        print(f"  {source:35s} n={len(g):5d}  chrF={chrf(g['prediction'], g['gold']):6.2f}"
              f"  BERTScore={g['bertscore_f1'].mean():6.2f}  ROUGE-L={g['rouge_l_f1'].mean():6.2f}")

    print("\nby language (best to worst chrF):")
    rows = [(lang, len(g), chrf(g["prediction"], g["gold"]),
              g["bertscore_f1"].mean(), g["rouge_l_f1"].mean())
            for lang, g in df.groupby("lang_code")]
    for lang, n, score, bs, rl in sorted(rows, key=lambda r: r[2], reverse=True):
        print(f"  {lang:12s} n={n:5d}  chrF={score:6.2f}  BERTScore={bs:6.2f}  ROUGE-L={rl:6.2f}")


if __name__ == "__main__":
    main()
