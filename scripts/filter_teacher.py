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

Combining teachers: the 122B run covers aya+OEG and the 35B shards cover the whole corpus,
so those qa_idx arrive twice. --prefer names the winner and is REQUIRED once inputs overlap
(the rows carry no model field, so the file path is the only signal, and argument order is
too easy to get backwards to be allowed to decide):

    python scripts/filter_teacher.py runs/teacher122b-aya-oeg.jsonl runs/teacher-s*of3.jsonl \
        --prefer 122b --report

A teacher answer is KEPT when `chrF >= --chrf-min` **or** `BERTScore >= --bertscore-min`
(pass `--require-both` for AND). Rationale: chrF alone punishes answers that are verbose
but semantically right (common for open-ended aya/OEG rows scored against short golds) --
BERTScore rescues those; BERTScore alone is too lenient on fluent hallucinations that stay
on-topic -- the OR of two calibrated thresholds is a better trade-off than either metric
alone, and `--report` prints the per-source grid to calibrate them on real data.

**One global threshold is not defensible** -- it does something different, and something
wrong, on each source. Measured pass rates at 30/70 (job 3861614, real 35B rows): OEG 94.4%,
MCIF 62.6%, aya 44.6%, belebele 33.3%, tydiqa 31.5%. So:

    --gold-only belebele        # never take a teacher answer for this source
    --source-min oeg=20,65      # looser threshold for this source

belebele in particular MUST be `--gold-only`: a passing row replaces a "2: <option>" gold
with prose, which wrecks the one format gold-SFT nailed (85.82 chrF) and buys nothing --
belebele does not reach the test set (no multiple choice there). The old claim that belebele
"always fails the filter anyway" was false; it only held for the chrF half of the OR. See
IMPLEMENTATION_NOTES 5.2/5.5. The per-source pass table is printed on every run so a bad
policy is visible rather than silent.

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


def load_shards(paths: list[str], prefer: str | None = None) -> pd.DataFrame:
    """Concatenate teacher JSONL files, resolving qa_idx collisions explicitly.

    Different teachers overlap: the 35B shards cover the whole corpus, the 122B run covers
    aya+OEG, so every aya/OEG qa_idx arrives twice. The rows carry no model field (both
    generators write the same schema), so the *file* is the only thing that says which
    teacher produced an answer -- and picking by argument order alone is a silent trap: both
    orders run fine and quietly train on a different teacher. So when files collide, --prefer
    must name the winner; otherwise we refuse rather than guess.
    """
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                r["src_file"] = p
                rows.append(r)
    df = pd.DataFrame(rows)

    collided = df["qa_idx"].duplicated(keep=False)
    if not collided.any():
        return df.reset_index(drop=True)

    files = sorted(df.loc[collided, "src_file"].unique())
    n_idx = df.loc[collided, "qa_idx"].nunique()
    if prefer is None:
        sys.exit(
            f"error: {n_idx} qa_idx appear in more than one input file, so these files "
            f"disagree about the teacher answer:\n"
            + "".join(f"    {f}\n" for f in files)
            + "Pass --prefer SUBSTRING to say which file's answers win (e.g. --prefer 122b).\n"
              "Refusing to pick by argument order: it would silently change the training "
              "targets depending on how the files were typed."
        )

    match = df["src_file"].str.contains(prefer, regex=False)
    if not match.any():
        sys.exit(f"error: --prefer {prefer!r} matched none of the input files:\n"
                 + "".join(f"    {f}\n" for f in sorted(df['src_file'].unique())))
    ambiguous = sorted(df.loc[match & collided, "src_file"].unique())
    if len(ambiguous) > 1:
        sys.exit(f"error: --prefer {prefer!r} matched several colliding files, so it still "
                 f"does not say which wins:\n" + "".join(f"    {f}\n" for f in ambiguous)
                 + "Use a substring unique to one file.")

    # Stable sort puts preferred rows first within each qa_idx; keep-first then resolves
    # every collision the same way regardless of the order the files were passed in.
    df = df.assign(_rank=(~match).astype(int)).sort_values("_rank", kind="stable")
    df = df[~df["qa_idx"].duplicated()].drop(columns="_rank").sort_index()
    n_won = df["src_file"].str.contains(prefer, regex=False).sum()
    print(f"resolved {n_idx} overlapping qa_idx in favour of --prefer {prefer!r} "
          f"({n_won} of {len(df)} kept rows come from the preferred file)", file=sys.stderr)
    return df.reset_index(drop=True)


def sentence_chrf(pred: str, ref: str) -> float:
    return sacrebleu.sentence_chrf(pred, [ref]).score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+",
                    help="teacher JSONL file(s) from scripts/teacher_generate.py")
    ap.add_argument("--report", action="store_true",
                    help="print score distributions and a threshold grid, write nothing")
    ap.add_argument("--prefer", metavar="SUBSTRING",
                    help="when the same qa_idx appears in several inputs (e.g. the 122B "
                         "aya+oeg run vs the 35B whole-corpus shards), keep the answer from "
                         "the file whose path contains SUBSTRING. Required whenever inputs "
                         "overlap; argument order never decides.")
    ap.add_argument("--chrf-min", type=float, default=30.0)
    ap.add_argument("--bertscore-min", type=float, default=70.0)
    ap.add_argument("--require-both", action="store_true",
                    help="keep only rows passing BOTH thresholds (default: either)")
    ap.add_argument("--gold-only", metavar="SUBSTR[,SUBSTR]",
                    help="sources whose name contains any of these substrings NEVER take a "
                         "teacher answer, whatever it scores (e.g. 'belebele'). See "
                         "IMPLEMENTATION_NOTES 5.5: belebele teacher rows pass 33%% of the "
                         "time and every pass replaces a '2: <option>' gold with prose.")
    ap.add_argument("--source-min", action="append", default=[], metavar="SUBSTR=CHRF,BERT",
                    help="per-source threshold override, repeatable "
                         "(e.g. --source-min oeg=20,65). Sources not named use the global "
                         "--chrf-min/--bertscore-min.")
    ap.add_argument("--mix", choices=["replace", "both", "teacher"], default="replace",
                    help="see module docstring (default: replace)")
    ap.add_argument("--bertscore-model", default="bert-base-multilingual-cased",
                    help="BERTScore backbone; must cover all task languages")
    ap.add_argument("--out", default="data/sft-distilled.jsonl",
                    help="SFT-ready JSONL for `train_lora.py --data`")
    args = ap.parse_args()

    df = load_shards(args.shards, args.prefer)
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

    def gate(frame, chrf_min, bert_min):
        if args.require_both:
            return (frame["chrf"] >= chrf_min) & (frame["bertscore"] >= bert_min)
        return (frame["chrf"] >= chrf_min) | (frame["bertscore"] >= bert_min)

    def match(substr):
        m = df["source"].str.contains(substr.strip(), case=False, regex=False)
        if not m.any():
            sys.exit(f"error: no source matches {substr.strip()!r}. Sources present:\n"
                     + "".join(f"    {s}\n" for s in sorted(df["source"].unique())))
        return m

    # Per-source policy. Default is the global threshold; --source-min overrides it for the
    # named sources; --gold-only vetoes them outright. Applied in that order so a source
    # named in both ends up gold-only.
    passed = gate(df, args.chrf_min, args.bertscore_min)
    policy = pd.Series(f"{args.chrf_min:g}/{args.bertscore_min:g}", index=df.index)
    for spec in args.source_min:
        substr, sep, thr = spec.partition("=")
        if not sep or thr.count(",") != 1:
            sys.exit(f"error: --source-min wants SUBSTR=CHRF,BERT (got {spec!r})")
        try:
            c, b = (float(x) for x in thr.split(","))
        except ValueError:
            sys.exit(f"error: --source-min thresholds must be numbers (got {thr!r})")
        m = match(substr)
        passed.loc[m] = gate(df[m], c, b).to_numpy()
        policy.loc[m] = f"{c:g}/{b:g}"
    for substr in (args.gold_only.split(",") if args.gold_only else []):
        m = match(substr)
        passed.loc[m] = False
        policy.loc[m] = "GOLD-ONLY"

    passed &= df["teacher"].str.strip() != ""
    print(f"\npass rate ({'AND' if args.require_both else 'OR'}; default "
          f"chrF>={args.chrf_min:g} / BERTScore>={args.bertscore_min:g}): "
          f"{passed.mean():.1%} overall")
    print(f"  {'source':35s} {'policy':>11s} {'pass':>7s}   n")
    for source, g in df.groupby("source"):
        pol = policy[g.index].unique()
        print(f"  {source:35s} {'|'.join(pol):>11s} {passed[g.index].mean():6.1%}  {len(g)}")

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
