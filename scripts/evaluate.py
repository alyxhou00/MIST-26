"""Summarise a predictions CSV from benchmark.py: per-task metrics first, then the legacy
overall/per-source/per-language chrF/BERTScore/ROUGE-L breakdowns.

**Read the per-task block, not the overall line.** Dev overall mixes sources that predict
opposite things, and one (belebele, 38% of the rows) that predicts nothing at all -- the test
set has no multiple choice. This script groups the rest by the test sub-task they proxy and
applies the metric each one deserves (TEST_SET_ANALYSIS 5b):

  qa-context             SPLIT IN TWO (2026-07-16), and the split matters more than the metric:
                         the test sub-task is 96% cross-lingual, so **MCIF is the faithful proxy
                         and tydiqa (79% of the pooled rows) is not** -- it is monolingual, worth
                         ~4% of the sub-task. Pooling them, as this script did until 2026-07-16,
                         reported mostly the wrong task.
                         Metric per half: EM + token F1 are SQuAD-style and only resolve when the
                         golds are short enough to hit exactly. Measured: tydiqa's golds are 63%
                         1-2 words (EM works, chrF has almost no resolution -- "1650" vs "around
                         1650" moves it about as much as right-vs-wrong does), but **MCIF's are
                         median 6 words, 42% are 8+ (EM ~0 -- judge it on chrF/BERTScore)**.
                         So EM is only informative on the proxy that doesn't resemble the test
                         set. The script prints the gold-length evidence next to EM either way.
  qa-oeg (long-form)     OEG: chrF/BERTScore against 175-word golds, + word-budget compliance,
                         which is scored at test time and which nothing else here can see.
  qa-oeg (short-answer)  aya: the same test task's short tail -- ~13 of qa-oeg's 100 unique
                         prompts are trivia and lists. Reported separately from long-form on
                         purpose: they are opposite ends of one spectrum and averaging them
                         describes neither.

chrF is cheap (CPU, seconds). BERTScore additionally needs a transformer forward pass over
every row (a GPU helps but CPU works for dev-set sizes); it's loaded once and reused for the
overall and per-group numbers. Independent of generation -- run it on the login node, re-run
it to add metrics, or point it at a *partial* CSV to score an interrupted job without
re-generating.

    python scripts/evaluate.py predictions/predictions-<jobid>.csv
"""

import argparse
import re
import string
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
import sacrebleu
from bert_score import BERTScorer
from rouge_score import rouge_scorer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from constraint_bank import BUDGET, measure, parse_budget  # noqa: E402

# Which dev sources stand in for which test task (TEST_SET_ANALYSIS 5b). qa-oeg is a spectrum,
# not one regime: of its 100 unique prompts ~20 carry a 120-300 word budget, ~65 are unbounded
# open-ended, and ~13 are short-answer trivia/lists ("name a country with no vowels in its
# name"). OEG proxies the long end, aya the short end -- they are reported as SEPARATE columns
# and must never be averaged, because dev's weighting is inverted against the test mix (aya has
# 978 rows for ~13% of the task, OEG 97 for ~87%).
#
# qa-context is split for the same reason, measured 2026-07-16 (EXPERIMENTS.md): the test
# sub-task is 96% CROSS-LINGUAL (passage in one language, question in another), and of our two
# proxies only MCIF is. tydiqa is monolingual -- 79% of the pooled rows standing in for ~4% of
# the real sub-task -- so pooling them, as this script used to, reports mostly the wrong task.
TASK_PROXY = {
    "qa-context (cross-lingual)": ["FBK-MT/MCIF"],
    "qa-context (monolingual)": ["copenlu/answerable_tydiqa"],
    "qa-oeg (long-form)": ["wmt25-mist-oeg-gpt-4.1"],
    "qa-oeg (short-answer)": ["CohereLabs/aya_dataset"],
}
# Which qa-context proxy actually resembles the test set. Printed on every run so the faithful
# one can't quietly be read as the minor column just because it has fewer rows (n=165 vs 615).
PROXY_FIDELITY = {
    "qa-context (cross-lingual)": "✅ FAITHFUL -- matches the 96% of test qa-context that is cross-lingual",
    "qa-context (monolingual)": "❌ UNFAITHFUL -- monolingual; ~4% of the test sub-task. Do not route on this.",
}
UNSCORED = {"facebook/belebele": "multiple choice; the test set has none at all"}


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


def _norm(text: str) -> list[str]:
    """SQuAD-style normalisation, generalised past English: casefold, strip punctuation
    (Unicode-aware -- `string.punctuation` misses Arabic/Devanagari/CJK marks) and collapse
    whitespace. No article stripping: "the/a/an" is English-only and 23 of our 24 languages
    would be unaffected or harmed by it."""
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    text = "".join(" " if (c in string.punctuation or unicodedata.category(c).startswith("P"))
                   else c for c in text)
    return text.split()


def exact_match(preds, refs) -> float:
    """Fraction of rows whose normalised prediction equals the normalised gold (0-100)."""
    return 100.0 * sum(_norm(p) == _norm(r) for p, r in zip(preds, refs)) / max(len(preds), 1)


def token_f1(preds, refs) -> float:
    """Mean per-row SQuAD token F1 (0-100): overlap between predicted and gold tokens.
    Partial credit where EM gives none -- "in 399" vs "399" scores 0 EM but 0.67 F1."""
    scores = []
    for pred, ref in zip(preds, refs):
        p, r = _norm(pred), _norm(ref)
        if not p or not r:
            scores.append(100.0 * (p == r))
            continue
        # Multiset intersection: Counter's & takes the min of each token's count, which is
        # what SQuAD's F1 counts. (pandas' & on two value_counts is a *logical* and, not an
        # elementwise min -- it silently misaligns indices and then dies on int & NaN.)
        common = sum((Counter(p) & Counter(r)).values())
        if common == 0:
            scores.append(0.0)
            continue
        precision, recall = common / len(p), common / len(r)
        scores.append(100.0 * 2 * precision * recall / (precision + recall))
    return sum(scores) / max(len(scores), 1)


def budget_compliance(inputs, preds, langs) -> tuple[int, float, float]:
    """(rows carrying a budget, % of those the prediction satisfies, mean signed overshoot %).

    Needs no gold -- it reads the constraint out of the prompt and measures the output, so it
    works identically on dev, on C-augmented data, and on official test outputs. Returns
    n=0 on the plain dev split, which has essentially no budgeted rows: that is the honest
    answer, not a failure (TEST_SET_ANALYSIS 4).
    """
    hits, overs = [], []
    for prompt, pred, lang in zip(inputs, preds, langs):
        lang = str(lang).split("_")[0]
        if lang not in BUDGET:
            continue
        band = parse_budget(str(prompt), lang)
        if band is None:
            continue
        lo, hi = band
        n = measure(str(pred), lang)
        hits.append(lo <= n <= hi)
        mid = (lo + hi) / 2
        overs.append(100.0 * (n - mid) / mid)
    if not hits:
        return 0, float("nan"), float("nan")
    return len(hits), 100.0 * sum(hits) / len(hits), sum(overs) / len(overs)


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

    # ---- per-task block: the numbers that actually predict test performance ----
    print("\n=== BY TEST SUB-TASK (use these to compare systems) ===")
    for task, sources in TASK_PROXY.items():
        g = df[df["source"].isin(sources)]
        if g.empty:
            print(f"\n{task}: no proxy rows in this file")
            continue
        print(f"\n{task}  (proxy: {', '.join(s.split('/')[-1] for s in sources)}; n={len(g)})")
        if task in PROXY_FIDELITY:
            print(f"  {PROXY_FIDELITY[task]}")
        if task.startswith("qa-context"):
            em = exact_match(g["prediction"], g["gold"])
            f1 = token_f1(g["prediction"], g["gold"])
            c = chrf(g["prediction"], g["gold"])
            # EM only means something where the golds are short enough to hit exactly. That is
            # true of tydiqa (63% of golds are 1-2 words) and NOT of MCIF (19%; median 6 words,
            # 42% are 8+) -- so print the evidence next to the number instead of letting the
            # reader assume the docstring's "2-word extractions" holds for both.
            short = (g["gold"].astype(str).str.split().str.len() <= 2).mean() * 100
            print(f"  Exact Match = {em:6.2f}   token F1 = {f1:6.2f}   chrF = {c:6.2f}   "
                  f"BERTScore = {g['bertscore_f1'].mean():6.2f}")
            print(f"  golds that are 1-2 words: {short:.0f}%  -> EM is "
                  f"{'meaningful here' if short >= 50 else 'NEARLY USELESS here (golds are too long to hit exactly)'}")
            # sqrt(EM x chrF): the team's hedge for the unknown official metric (user, 2026-07-16
            # -- we are not asking the organisers). Only honest where EM is; a geometric mean
            # with a ~0 factor is ~0 no matter what chrF says.
            if short >= 50:
                print(f"  sqrt(EM x chrF) = {(em * c) ** 0.5:6.2f}   <- the selection rule")
            else:
                print("  sqrt(EM x chrF): NOT REPORTED -- EM is ~0 here, so the geometric mean "
                      "would rank noise. Judge this proxy on chrF/BERTScore.")
        else:
            print(f"  chrF = {chrf(g['prediction'], g['gold']):6.2f}   BERTScore = "
                  f"{g['bertscore_f1'].mean():6.2f}   ROUGE-L = {g['rouge_l_f1'].mean():6.2f}")
        if "input" in g.columns:
            n_b, ok, over = budget_compliance(g["input"], g["prediction"], g["lang_code"])
            if n_b:
                print(f"  word budget: {ok:.1f}% of {n_b} budgeted rows in band "
                      f"(mean overshoot {over:+.0f}%)")
            else:
                print("  word budget: no budgeted rows here (expected on dev; "
                      "the metric is for test outputs / C-augmented data)")

    unscored = df[df["source"].isin(UNSCORED)]
    if not unscored.empty:
        print(f"\nexcluded from the above ({len(unscored)} rows, {len(unscored)/len(df):.0%} "
              f"of this file) -- these do not predict test performance:")
        for src, why in UNSCORED.items():
            n = (df["source"] == src).sum()
            if n:
                print(f"  {src:35s} n={n:5d}  {why}")

    # ---- legacy overall: kept for continuity with older rows, NOT for judging ----
    print("\n=== overall (LEGACY -- 71% noise, do not compare systems on this) ===")
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
