"""Minimal zero-shot QA benchmark: Qwen3.5-2B on the WMT26 MIST `qa` task.

Splits the `qa` examples 80/20 (train/dev, seed 42), runs the model zero-shot on the dev
half via its chat template, and reports chrF.

    python scripts/benchmark.py --limit 50     # quick check on 50 examples
    python scripts/benchmark.py                # full dev split
"""

import argparse
import csv
from pathlib import Path

from datasets import load_dataset

# lang_code -> human-readable name, for the optional --lang-hint prompt prefix.
# Covers the 27 languages listed in the WMT26 MIST task page; the 2 "surprise"
# test languages aren't known yet.
LANG_NAMES = {
    "arb_Arab": "Arabic", "ben_Beng": "Bengali", "ces_Latn": "Czech",
    "ckb_Arab": "Central Kurdish", "deu_Latn": "German", "eng_Latn": "English",
    "fin_Latn": "Finnish", "fra_Latn": "French", "hat_Latn": "Haitian Creole",
    "hin_Deva": "Hindi", "ind_Latn": "Indonesian", "ita_Latn": "Italian",
    "jpn_Jpan": "Japanese", "kor_Hang": "Korean", "mar_Deva": "Marathi",
    "pes_Arab": "Persian", "por_Latn": "Portuguese", "rus_Cyrl": "Russian",
    "slk_Latn": "Slovak", "spa_Latn": "Spanish", "swh_Latn": "Swahili",
    "tel_Telu": "Telugu", "tha_Thai": "Thai", "tur_Latn": "Turkish",
    "vie_Latn": "Vietnamese", "yor_Latn": "Yoruba", "zho_Hans": "Chinese",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--limit", type=int, default=0, help="0 = whole dev split")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/predictions.csv",
                    help="CSV of per-example source/lang/input/gold/prediction")
    ap.add_argument("--lang-hint", action="store_true",
                    help="prefix each prompt with 'Please respond in <language>.', "
                         "derived from lang_code -- disambiguates examples where the "
                         "input's language differs from the expected output language "
                         "(e.g. aya_dataset: English question, non-English answer)")
    args = ap.parse_args()

    # --- data: qa subset, 80/20 split, evaluate on dev ---
    df = load_dataset("pinzhenchen/wmt26-mist-sample")["train"].to_pandas()
    qa = df[df["task"] == "qa"].sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    dev = qa.iloc[: int(len(qa) * 0.2)]
    if args.limit:
        dev = dev.head(args.limit)
    print(f"dev examples: {len(dev)}")

    # --- model (Qwen3.5-2B is multimodal; we use it text-only) ---
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    # --- zero-shot generation ---
    # Each prediction is written to the CSV and flushed immediately, so an interrupted run
    # (time limit, node failure, OOM) keeps every example completed so far instead of losing all.
    # Fix the seed so the (random) sampling below is reproducible across runs.
    set_seed(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    preds, golds = [], []
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: opens in Excel
        writer = csv.writer(f)
        writer.writerow(["source", "lang_code", "input", "gold", "prediction"])
        for i, row in enumerate(dev.itertuples(index=False), 1):
            try:
                content = row.input
                if args.lang_hint:
                    lang_name = LANG_NAMES.get(row.lang_code, row.lang_code)
                    content = f"Please respond in {lang_name}.\n\n{content}"
                prompt = tok.apply_chat_template(
                    [{"role": "user", "content": content}],
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
            preds.append(pred)
            golds.append(row.output)
            print(f"  [{i}/{len(dev)}]", flush=True)
    print(f"wrote {len(preds)} rows -> {args.out}")

    # --- score (also recoverable from the CSV above if the run was interrupted) ---
    import sacrebleu

    chrf = sacrebleu.corpus_chrf(preds, [golds]).score
    print(f"\nchrF = {chrf:.2f}  (n={len(preds)}, model={args.model})")


if __name__ == "__main__":
    main()
