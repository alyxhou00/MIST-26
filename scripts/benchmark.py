"""Minimal QA benchmark: Qwen3.5-2B on the WMT26 MIST `qa` task, zero- or few-shot.

Splits the `qa` examples 80/20 (train/dev, seed 42), runs the model on the dev half via its
chat template, and writes a predictions CSV. Scoring is a separate step -- run
`scripts/evaluate.py` on the CSV for chrF/BERTScore/ROUGE-L (single source of truth).

    python scripts/benchmark.py                       # full dev split; zero-shot, lang-hint ON
    python scripts/benchmark.py --shots 3             # few-shot: 3 train-split demonstrations
    python scripts/benchmark.py --limit 50            # quick check on the first 50 dev rows
    python scripts/benchmark.py --no-lang-hint        # raw zero-shot: no target-language instruction
    python scripts/benchmark.py --source aya --lang hin_Deva   # only the cross-lingual aya rows
    python scripts/benchmark.py --out runs/my-run.csv          # choose where predictions are written

Flags: --shots (few-shot demonstrations per example, default 0 = zero-shot) · --limit (cap dev
rows) · --source (substring on `source`) · --lang (exact lang_code) · --[no-]lang-hint
(target-language system turn, default on) · --out (predictions CSV path) · --model ·
--max-new-tokens · --seed. Filters combine and apply within the dev split.
"""

import argparse
import csv
import zlib
from pathlib import Path

from datasets import load_dataset

from prompt_template import build_messages


def make_shot_picker(train, k: int, seed: int):
    """Return a function picking k few-shot demonstrations from the train split for one dev row.

    Selection is matched to the dev row: prefer train rows with the same (source, lang_code) --
    they demonstrate both the task format (e.g. belebele's multiple-choice answers) and the
    target language -- falling back to same source only, then to the whole train pool, whenever
    a stratum has fewer than k rows.

    The sample is seeded per dev example from a hash of its input text (not from a shared RNG
    consumed in iteration order), so a given dev example always gets the same shots regardless
    of --limit/--source/--lang filters or row order. That keeps A/B runs comparable and any
    single prediction exactly reproducible in isolation.

    Rows whose input is byte-identical to the dev row's are excluded from every tier: a few
    aya prompts repeat verbatim across the 80/20 split (53 of 2978 dev rows, 4 with the same
    gold), and sampling one of those as a demonstration would hand the model its own answer.
    """
    by_src_lang = {key: grp for key, grp in train.groupby(["source", "lang_code"])}
    by_src = {key: grp for key, grp in train.groupby("source")}
    empty = train.iloc[0:0]

    def pick(source: str, lang_code: str, input_text: str) -> list[tuple[str, str]]:
        def usable(pool):  # drop verbatim copies of the dev row's own question
            return pool[pool["input"] != input_text]

        pool = usable(by_src_lang.get((source, lang_code), empty))
        if len(pool) < k:
            pool = usable(by_src.get(source, train))
        if len(pool) < k:
            pool = usable(train)
        picked = pool.sample(
            n=min(k, len(pool)),
            random_state=zlib.crc32(f"{seed}:{input_text}".encode("utf-8")),
        )
        return list(zip(picked["input"], picked["output"]))

    return pick


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--limit", type=int, default=0, help="0 = whole dev split")
    ap.add_argument("--source", default=None,
                    help="only dev rows whose `source` contains this substring, case-insensitive "
                         "(e.g. 'aya', 'belebele'). Useful for a targeted smoke test.")
    ap.add_argument("--lang", default=None,
                    help="only dev rows with this exact lang_code (e.g. 'hin_Deva')")
    ap.add_argument("--shots", type=int, default=0,
                    help="few-shot demonstrations per example, drawn from the train split "
                         "(never dev) and matched on (source, lang_code) where possible. "
                         "Inserted as completed user/assistant turns before the question. "
                         "Default 0 = zero-shot.")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/predictions.csv",
                    help="CSV of per-example source/lang/input/gold/prediction")
    ap.add_argument("--lang-hint", action=argparse.BooleanOptionalAction, default=True,
                    help="add a system turn 'Respond in <language>.' derived from lang_code "
                         "(default: ON). Disambiguates examples where the input's language "
                         "differs from the expected output language (e.g. aya_dataset: English "
                         "question, non-English answer). Pass --no-lang-hint to disable and "
                         "reproduce the raw zero-shot baseline. Same template is reused for SFT "
                         "(prompt_template.py).")
    args = ap.parse_args()

    # --- data: qa subset, 80/20 split, evaluate on dev ---
    df = load_dataset("pinzhenchen/wmt26-mist-sample")["train"].to_pandas()
    qa = df[df["task"] == "qa"].sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    dev = qa.iloc[: int(len(qa) * 0.2)]
    # Few-shot demonstrations come from the *other* 80% (the train split), so a dev example
    # can never appear among its own shots and the dev metric stays honest.
    pick_shots = None
    if args.shots:
        train = qa.iloc[int(len(qa) * 0.2):]
        pick_shots = make_shot_picker(train, args.shots, args.seed)
    # Optional filters, applied within the dev split so we still only ever touch dev rows.
    # --limit is applied last, so it caps whatever the filters leave.
    if args.source:
        dev = dev[dev["source"].str.contains(args.source, case=False, na=False)]
    if args.lang:
        dev = dev[dev["lang_code"] == args.lang]
    if args.limit:
        dev = dev.head(args.limit)
    print(f"dev examples: {len(dev)}  shots={args.shots}  "
          f"(source={args.source}, lang={args.lang}, limit={args.limit or 'none'})")
    if len(dev) == 0:
        print("no rows match the filters; nothing to do.")
        return

    # --- model (Qwen3.5-2B is multimodal; we use it text-only) ---
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    # --- generation ---
    # Each prediction is written to the CSV and flushed immediately, so an interrupted run
    # (time limit, node failure, OOM) keeps every example completed so far instead of losing all.
    # Fix the seed so the (random) sampling below is reproducible across runs.
    set_seed(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: opens in Excel
        writer = csv.writer(f)
        writer.writerow(["source", "lang_code", "input", "gold", "prediction"])
        for i, row in enumerate(dev.itertuples(index=False), 1):
            try:
                shots = (pick_shots(row.source, row.lang_code, row.input)
                         if pick_shots else None)
                messages = build_messages(row.input, row.lang_code,
                                          lang_hint=args.lang_hint, examples=shots)
                prompt = tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,  # 2B is prone to thinking loops; keep non-thinking (card)
                )
                inputs = tok(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    # Card's "non-thinking, text task" sampling. (presence_penalty=2.0 from the
                    # card has no model.generate equivalent, so it's omitted.)
                    out = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=1.0,
                        top_p=1.0,
                        top_k=20,
                    )
                pred = tok.decode(
                    out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()
            except Exception as e:  # noqa: BLE001 - keep going so one bad example can't lose the run
                print(f"  [{i}] FAILED: {type(e).__name__}: {e}", flush=True)
                pred = ""
            writer.writerow([row.source, row.lang_code, row.input, row.output, pred])
            f.flush()  # ensure the row is on disk before moving on
            n_written += 1
            print(f"  [{i}/{len(dev)}]", flush=True)
    print(f"wrote {n_written} rows -> {args.out}")
    print(f"score it with:  python scripts/evaluate.py {args.out}")


if __name__ == "__main__":
    main()
