"""Roadmap D, second cut: subsample the 8,009-row Bhojpuri pack to a proportionate size.

The full pack (`data/sft-bho.jsonl`, built by the D pipeline) is **40.7% of the C+D
training mix** (8,009 of 19,683 examples) but bho is only **4.2% of the qa test set**
(460 of 10,999 rows). That is a ~10x over-allocation, and it is the leading suspect for
the dev regression C+D showed (job 3869130: qa-oeg agg -1.42, OEG -1.67) -- a cost paid
on the ~96% of rows that are not bho. Job 3876434 (C-only) prices that suspicion; this
script builds the follow-up that keeps D's benefit at a proportionate training cost.

    python scripts/shrink_bho_pack.py data/sft-bho.jsonl \
        --out data/sft-bho-small.jsonl --report      # inspect the selection
    python scripts/shrink_bho_pack.py data/sft-bho.jsonl --out data/sft-bho-small.jsonl

    python scripts/augment_constraints.py data/train_v2.jsonl \
        --out data/train_v2-cd-small.jsonl --append-bho data/sft-bho-small.jsonl

**Keep both sources -- they fix different failure modes.** The instinct was to keep the
xP3x translation half (cleaner supervision: real parallel text) and drop the fineweb
continuation half (raw web scrape). Measured output lengths say that is backwards for
this task:

    HuggingFaceFW/fineweb-2:bho_Deva   n=6000  median=120 words  p10=55  p90=237
    CohereLabs/xP3x:bho_Deva           n=2009  median= 24 words  p10=15  p90= 37

qa-oeg wants ~150-word answers, and one of the three flaws found by hand-reading the
3875151 outputs was bho answers of 5-10 words. **fineweb is the only half that
demonstrates paragraph-length bho at all**; an xP3x-only pack would train the terseness
in. xP3x earns its place on the other flaw -- it is hin->bho parallel text, i.e. a direct
demonstration of *not* falling back to Hindi, which is the 36%-drift failure. So the
shrunk pack keeps a majority of fineweb, sampled toward the target length band, plus
enough xP3x to hold the anti-drift signal.

Sampling is deterministic (seed 42) so the file is reproducible, and it never crosses the
item-group split rule: the pack is training-only with no dev counterpart (`origin` =
`bho-pack`, `qa_idx` < 0), so no group can be split by subsampling it.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

FINEWEB = "HuggingFaceFW/fineweb-2:bho_Deva"
XP3X = "CohereLabs/xP3x:bho_Deva"

# fineweb rows whose answer sits in this band are the ones that look like a qa-oeg answer
# (the test's stated budgets run 30-250 words). Rows outside it are eligible only as
# filler if the band cannot fill the quota.
TARGET_LO, TARGET_HI = 80, 250


def load(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pick(rows: list[dict], quota: int, rng: random.Random,
         prefer_band: bool) -> list[dict]:
    """Take `quota` rows, preferring the target length band when asked."""
    if quota >= len(rows):
        return list(rows)
    if not prefer_band:
        return rng.sample(rows, quota)
    band = [r for r in rows if TARGET_LO <= len(r["output"].split()) <= TARGET_HI]
    rest = [r for r in rows if not (TARGET_LO <= len(r["output"].split()) <= TARGET_HI)]
    if len(band) >= quota:
        return rng.sample(band, quota)
    return band + rng.sample(rest, quota - len(band))


def report(name: str, rows: list[dict]) -> None:
    lens = sorted(len(r["output"].split()) for r in rows)
    by_src = collections.Counter(r.get("source") for r in rows)
    print(f"  {name}: {len(rows)} rows, median {lens[len(lens) // 2]} words "
          f"(p10 {lens[len(lens) // 10]}, p90 {lens[9 * len(lens) // 10]})")
    for src, n in sorted(by_src.items()):
        print(f"      {src}: {n}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", help="the full bho pack (data/sft-bho.jsonl)")
    ap.add_argument("--out", default="data/sft-bho-small.jsonl")
    ap.add_argument("--fineweb", type=int, default=1400,
                    help="rows to keep from the fineweb continuation half "
                         "(long-form fluency; default 1400)")
    ap.add_argument("--xp3x", type=int, default=1000,
                    help="rows to keep from the xP3x hin->bho half "
                         "(anti-Hindi-drift; default 1000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report", action="store_true",
                    help="print the selection and write nothing")
    args = ap.parse_args()

    rows = load(Path(args.data))
    by_src = collections.defaultdict(list)
    for r in rows:
        by_src[r.get("source")].append(r)

    unknown = set(by_src) - {FINEWEB, XP3X}
    if unknown:
        sys.exit(f"unexpected source(s) in {args.data}: {sorted(unknown)} -- this script "
                 f"encodes per-source quotas and cannot guess one for a new source")

    rng = random.Random(args.seed)
    kept = (pick(by_src[FINEWEB], args.fineweb, rng, prefer_band=True)
            + pick(by_src[XP3X], args.xp3x, rng, prefer_band=False))
    kept.sort(key=lambda r: int(r["qa_idx"]), reverse=True)

    print(f"input:  {len(rows)} rows")
    report("input ", rows)
    print(f"kept:   {len(kept)} rows ({100 * len(kept) / len(rows):.1f}% of the pack)")
    report("kept  ", kept)
    mix = 11674 + len(kept)
    print(f"\nprojected training mix: {mix} examples, bho = {100 * len(kept) / mix:.1f}% "
          f"(was 40.7%); bho is 4.2% of the qa test set")

    if args.report:
        print("\n--report: nothing written")
        return

    out = Path(args.out)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(kept)} rows -> {out}")
    print(f"next:  python scripts/augment_constraints.py data/train_v2.jsonl \\\n"
          f"           --out data/train_v2-cd-small.jsonl --append-bho {out}")


if __name__ == "__main__":
    main()
