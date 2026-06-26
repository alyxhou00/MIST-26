"""Minimal zero-shot QA benchmark: Qwen3.5-2B on the WMT26 MIST `qa` task.

Splits the `qa` examples 80/20 (train/dev, seed 42), runs the model zero-shot on the dev
half via its chat template, and reports chrF.

    python benchmark.py --limit 50     # quick check on 50 examples
    python benchmark.py                # full dev split
"""

import argparse

from datasets import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-2B")
    ap.add_argument("--limit", type=int, default=0, help="0 = whole dev split")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
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
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    # --- zero-shot generation ---
    preds = []
    for i, text in enumerate(dev["input"], 1):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        pred = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        preds.append(pred)
        print(f"  [{i}/{len(dev)}]", flush=True)

    # --- score ---
    import sacrebleu

    chrf = sacrebleu.corpus_chrf(preds, [dev["output"].tolist()]).score
    print(f"\nchrF = {chrf:.2f}  (n={len(dev)}, model={args.model})")


if __name__ == "__main__":
    main()
