"""Filter teacher generations against the golds and emit SFT-ready training data.

Takes the JSONL shards written by scripts/teacher_generate.py, scores every teacher answer
against its gold (per-row sentence chrF + BERTScore F1 -- the same metric stack as
scripts/evaluate.py), and writes a training file for `train_lora.py --data`. The teacher
smoke run (job 3859176) showed why this step is load-bearing: the 35B teacher is fluent but
hallucinates facts (wrong quiz answers, invented geography), and those rows must not become
training targets.

    # 1. look at the score distributions before choosing thresholds
    python scripts/filter_teacher.py runs/teacher-s*of3.jsonl --report

    # 2. then write the training file
    python scripts/filter_teacher.py runs/teacher-s*of3.jsonl \
        --chrf-min 30 --bertscore-min 70 --out data/sft-distilled.jsonl

A teacher answer is KEPT when `chrF >= --chrf-min` **or** `BERTScore >= --bertscore-min`
(pass `--require-both` for AND). Rationale: chrF alone punishes answers that are verbose
but semantically right (common for open-ended aya/OEG rows scored against short golds) --
BERTScore rescues those; BERTScore alone is too lenient on fluent hallucinations that stay
on-topic -- the OR of two calibrated thresholds is a better trade-off than either metric
alone, and `--report` prints the per-source grid to calibrate them on real data.

Mix policy (--mix):
    replace  (default) one example per train row: the teacher's answer where it passed,
             the gold otherwise. Keeps the dataset the same size/rows as the gold-SFT run
             (adapter 3822375), so "did distilled targets help" stays a one-variable A/B.
    both     every gold row, plus a second copy with the teacher's answer where it passed.
    teacher  only rows where the teacher passed, teacher answers only.

Output rows are {qa_idx, source, lang_code, input, output, origin} -- the same column names
train_lora.py consumes, with `origin` ("teacher"/"gold") kept for bookkeeping.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import sacrebleu

from evaluate import bertscore_f1  # scripts/evaluate.py, not the HF `evaluate` package


def load_shards(paths: list[str]) -> pd.DataFrame:
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f)
    df = pd.DataFrame(rows)
    dupes = df["qa_idx"].duplicated()
    if dupes.any():
        print(f"WARNING: dropping {dupes.sum()} duplicate qa_idx rows "
              f"(overlapping shards?)", file=sys.stderr)
        df = df[~dupes]
    return df.reset_index(drop=True)


def sentence_chrf(pred: str, ref: str) -> float:
    return sacrebleu.sentence_chrf(pred, [ref]).score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+",
                    help="teacher JSONL file(s) from scripts/teacher_generate.py")
    ap.add_argument("--report", action="store_true",
                    help="print score distributions and a threshold grid, write nothing")
    ap.add_argument("--chrf-min", type=float, default=30.0)
    ap.add_argument("--bertscore-min", type=float, default=70.0)
    ap.add_argument("--require-both", action="store_true",
                    help="keep only rows passing BOTH thresholds (default: either)")
    ap.add_argument("--mix", choices=["replace", "both", "teacher"], default="replace",
                    help="see module docstring (default: replace)")
    ap.add_argument("--bertscore-model", default="bert-base-multilingual-cased",
                    help="BERTScore backbone; must cover all task languages")
    ap.add_argument("--out", default="data/sft-distilled.jsonl",
                    help="SFT-ready JSONL for `train_lora.py --data`")
    args = ap.parse_args()

    df = load_shards(args.shards)
    n_empty = (df["teacher"].str.strip() == "").sum()
    print(f"loaded {len(df)} teacher rows from {len(args.shards)} file(s)"
          + (f" ({n_empty} with empty teacher answer -> auto-fail)" if n_empty else ""))

    # --- per-row scores (empty teacher answers score 0 in both metrics) ---
    from bert_score import BERTScorer  # deferred: slow import, needs the model download
    print("scoring: per-row sentence chrF + BERTScore F1 ...", flush=True)
    df["chrf"] = [sentence_chrf(t, g) for t, g in zip(df["teacher"], df["gold"])]
    df["bertscore"] = bertscore_f1(BERTScorer(model_type=args.bertscore_model),
                                   df["teacher"], df["gold"])

    if args.report:
        qtiles = [0.05, 0.25, 0.50, 0.75, 0.95]
        print("\nscore distributions (per source):")
        for name, g in [("ALL", df)] + list(df.groupby("source")):
            cq = g["chrf"].quantile(qtiles).round(1).tolist()
            bq = g["bertscore"].quantile(qtiles).round(1).tolist()
            print(f"  {name:35s} n={len(g):6d}  chrF p5/p25/p50/p75/p95 = {cq}")
            print(f"  {'':35s} {'':9s}  BERT p5/p25/p50/p75/p95 = {bq}")
        print("\nkept fraction (chrF >= C or BERTScore >= B):")
        header = "  C\\B   " + "".join(f"{b:>8.0f}" for b in (60, 65, 70, 75, 80))
        print(header)
        for c in (20, 30, 40, 50, 60):
            cells = []
            for b in (60, 65, 70, 75, 80):
                kept = ((df["chrf"] >= c) | (df["bertscore"] >= b)).mean()
                cells.append(f"{kept:8.1%}")
            print(f"  {c:<6.0f}" + "".join(cells))
        return

    passed = ((df["chrf"] >= args.chrf_min) & (df["bertscore"] >= args.bertscore_min)
              if args.require_both else
              (df["chrf"] >= args.chrf_min) | (df["bertscore"] >= args.bertscore_min))
    passed &= df["teacher"].str.strip() != ""
    print(f"\npass rate ({'AND' if args.require_both else 'OR'}, "
          f"chrF>={args.chrf_min}, BERTScore>={args.bertscore_min}): "
          f"{passed.mean():.1%} overall")
    for source, g in df.groupby("source"):
        print(f"  {source:35s} {passed[g.index].mean():6.1%} of n={len(g)}")

    def rows(frame, col, origin):
        for r in frame.itertuples(index=False):
            yield {"qa_idx": int(r.qa_idx), "source": r.source, "lang_code": r.lang_code,
                   "input": r.input, "output": getattr(r, col), "origin": origin}

    if args.mix == "replace":
        out_rows = list(rows(df[passed], "teacher", "teacher")) \
                 + list(rows(df[~passed], "gold", "gold"))
    elif args.mix == "both":
        out_rows = list(rows(df, "gold", "gold")) \
                 + list(rows(df[passed], "teacher", "teacher"))
    else:  # teacher
        out_rows = list(rows(df[passed], "teacher", "teacher"))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in sorted(out_rows, key=lambda r: r["qa_idx"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_teacher = sum(r["origin"] == "teacher" for r in out_rows)
    print(f"wrote {len(out_rows)} rows ({n_teacher} teacher / {len(out_rows) - n_teacher} gold) "
          f"-> {args.out}")
    print(f"train with:  python scripts/train_lora.py --data {args.out}")


if __name__ == "__main__":
    main()
