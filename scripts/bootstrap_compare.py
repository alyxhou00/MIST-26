"""Paired bootstrap over rows: is a COMBINED gap between two systems bigger than the column?

Written 2026-07-23, for a specific problem. The dev qa-oeg aggregate is
`0.87*OEG + 0.13*aya`, so 87% of the number that has been ranking the C/D adapters comes
from `wmt25-mist-oeg-gpt-4.1` -- **90 rows**. Across the four adapters the aya column
(n=944) moves 0.28 points end to end while the OEG column moves 2.8, which is what a
90-row column does whether or not anything real changed. Ranking systems 1.4 points apart
on it is a claim about the sample, not about the systems, until someone resamples it.

So this script does the resampling. It is the same paired bootstrap sacrebleu uses for BLEU,
applied to `evaluate.combined()`:

  * **Paired.** One set of resampled row indices per iteration, shared by every system --
    that is what makes the difference distribution tight enough to say anything at n=90.
    Scoring each system on its own independent resample would measure both systems' sampling
    noise instead of the noise in their *gap*.
  * **Recomputed, not averaged.** chrF is a corpus statistic (character n-gram counts pooled
    over rows, not a mean of per-row scores), so each iteration recomputes it from the sampled
    rows. BERTScore and ROUGE-L are per-row means and are averaged over the same indices;
    both are computed once up front, since the rows do not change, only which ones are drawn.
  * **Report the gap, not two intervals.** Per-system CIs overlapping does not mean the
    difference is insignificant -- the systems are scored on identical rows, so their errors
    are correlated. `p` below is the fraction of iterations where the gap changes sign.

Every file must be a `benchmark.py` CSV over the *same* dev rows (columns: source, lang_code,
input, gold, prediction); rows are aligned on `input` and the script refuses to run if the
sets differ, since an unpaired "paired" bootstrap silently reports nonsense.

    python scripts/bootstrap_compare.py runs/predictions-lora-A.csv runs/predictions-lora-B.csv
    python scripts/bootstrap_compare.py --source CohereLabs/aya_dataset A.csv B.csv C.csv

The first file is the reference; every other file is reported as a delta against it. This
answers "is the gap real", not "which is better" -- a gap that survives resampling can still
be an artifact of a proxy that only stands in for the test task (see evaluate.py's header).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from bert_score import BERTScorer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import bertscore_f1, chrf, combined, rouge_l  # noqa: E402


def load(path: str, source: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")  # strip the BOM benchmark.py writes
    df["prediction"] = df["prediction"].fillna("")
    df["gold"] = df["gold"].fillna("")
    df = df[df["source"] == source].reset_index(drop=True)
    if df.empty:
        sys.exit(f"{path}: no rows with source={source!r}")
    return df


def align(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Put every system's rows in one order, keyed on the prompt. Bails on any mismatch.

    One row = one question = one gold + one prediction per system. Every CSV carries its own
    copy of the `gold` column (benchmark.py writes it out per run), and they are only *supposed*
    to be the same text, read from the same dev file -- so this checks it instead of assuming.
    A gold that differs between two files means they were scored against different dev
    revisions, which would make the comparison meaningless in a way nothing downstream would
    show."""
    ref_name, ref = next(iter(frames.items()))
    keys = list(ref["input"])
    if len(set(keys)) != len(keys):
        sys.exit(f"{ref_name}: duplicate prompts in this source -- cannot align on `input`")
    out = {}
    for name, df in frames.items():
        if set(df["input"]) != set(keys):
            sys.exit(f"{name}: scored a different row set than {ref_name} "
                     f"({len(df)} vs {len(keys)} rows) -- these are not paired")
        out[name] = df.set_index("input").loc[keys].reset_index()

    ref_gold = out[ref_name]["gold"]
    for name, df in out.items():
        differing = (df["gold"] != ref_gold).sum()
        if differing:
            sys.exit(f"{name}: {differing} of {len(df)} rows have a different `gold` than "
                     f"{ref_name} -- these files were scored against different dev data")
    return out


def score(df: pd.DataFrame, idx: np.ndarray) -> float:
    """COMBINED over one resample: chrF recomputed on the drawn rows, the other two averaged."""
    preds, golds = df["prediction"].to_numpy()[idx], df["gold"].to_numpy()[idx]
    return combined(chrf(preds, golds),
                    df["bertscore_f1"].to_numpy()[idx].mean(),
                    df["rouge_l_f1"].to_numpy()[idx].mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pred_csvs", nargs="+", help="benchmark.py CSVs; the first is the reference")
    ap.add_argument("--source", default="wmt25-mist-oeg-gpt-4.1",
                    help="dev source column to compare on (default: the 90-row OEG column "
                         "that carries 87%% of the qa-oeg aggregate)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bertscore-model", default="bert-base-multilingual-cased")
    args = ap.parse_args()

    frames = align({p: load(p, args.source) for p in args.pred_csvs})
    n = len(next(iter(frames.values())))
    print(f"source: {args.source}   n = {n} rows   systems = {len(frames)}   "
          f"bootstrap = {args.n_boot} resamples (seed {args.seed})\n")

    scorer = BERTScorer(model_type=args.bertscore_model)
    for df in frames.values():
        df["bertscore_f1"] = bertscore_f1(scorer, df["prediction"], df["gold"])
        df["rouge_l_f1"] = rouge_l(df["prediction"], df["gold"])

    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, n, size=(args.n_boot, n))          # shared across systems: paired
    point = {name: score(df, np.arange(n)) for name, df in frames.items()}
    boot = {name: np.array([score(df, idx) for idx in draws]) for name, df in frames.items()}

    ref_name = args.pred_csvs[0]
    print(f"reference: {ref_name}   COMBINED = {point[ref_name]:.2f}\n")
    for name in args.pred_csvs[1:]:
        d = boot[name] - boot[ref_name]
        lo, hi = np.percentile(d, [2.5, 97.5])
        # Two-sided: how often the sign flips, doubled. A gap that survives has p small AND
        # a CI clear of zero; the two disagree only when the difference distribution is skewed.
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        verdict = "distinguishable" if lo > 0 or hi < 0 else "NOT distinguishable from noise"
        print(f"{name}")
        print(f"    COMBINED = {point[name]:.2f}   delta = {point[name] - point[ref_name]:+.2f}"
              f"   95% CI [{lo:+.2f}, {hi:+.2f}]   p = {p:.3f}   -> {verdict}")


if __name__ == "__main__":
    main()
