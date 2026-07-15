"""Teacher generation for sequence-level distillation (EXPERIMENTS.md "LoRA SFT on
distilled data").

Runs a large teacher model over the qa TRAIN split -- the 80% side of benchmark.py's
80/20 seed-42 split, so dev stays honest -- and writes one JSON object per row with every
input column plus the teacher's answer. Downstream steps (separate scripts) then
quality-filter the teacher answers against the golds (chrF/BERTScore) and LoRA-SFT the 9B
student on the filtered teacher+gold mix.

    python scripts/teacher_generate.py --limit 15                # smoke
    python scripts/teacher_generate.py --shard 1/2 --out runs/teacher-s1of2.jsonl

Teacher default is Qwen/Qwen3.5-35B-A3B (bf16): the largest family member that fits one
A100 80GB, and MoE (3B active), so generation is fast enough for ~11.9K rows in one job.
There is no Qwen3.5-32B (the family is 27B dense / 35B-A3B MoE); the quantized variants
would fit an A40 but need packages we cannot install while the atuin venv is read-only
(README "Temporary layout"). The teacher is never deployed, so its parameters don't count
against the shared task's 10B limit -- only the student does.

The teacher sees our raw train `input` with the lang-hint system turn ON by default: its
job is to produce the best-quality answer in the right output language (aya rows ask an
English question expecting a local-language answer -- without the hint those generations
would be wasted). The *student's* prompt format is a separate decision made at SFT time
(the output file keeps the raw input, so it can be re-wrapped, e.g. in test-style
conversational shells -- see TEST_SET_ANALYSIS.md sec. 8).

Resume: rows are keyed by `qa_idx` (position in the seed-42 shuffled qa frame, stable
across runs); rerunning with the same --out skips ids already present, so a job that hits
the wall clock is resubmitted as-is. --shard i/n splits the train rows into n contiguous
chunks for parallel jobs (separate --out per shard, concatenate afterwards).
"""

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset

from prompt_template import build_messages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="process the i-th of n contiguous chunks of the train split "
                         "(1-based, e.g. '2/3'); applied before --limit")
    ap.add_argument("--limit", type=int, default=0, help="0 = whole train split")
    ap.add_argument("--lang-hint", action=argparse.BooleanOptionalAction, default=True,
                    help="give the teacher the 'Respond in <language>.' system turn "
                         "(default ON -- see module docstring)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="sampling temperature (Qwen3.5 instruct card recommendation)")
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--gptq-backend", default=None,
                    help="force the gptqmodel kernel backend ('torch', 'triton', "
                         "'exllama_v2', ...) when loading a GPTQ checkpoint. The default "
                         "Marlin kernel rejects Qwen3.5-122B-A10B-GPTQ-Int4's out_features=1 "
                         "layer (job 3859381); 'torch' accepts any shape at some speed cost.")
    ap.add_argument("--seed", type=int, default=42,
                    help="split seed AND sampling seed; must stay 42 to match "
                         "benchmark.py's 80/20 split or the dev set leaks into training")
    ap.add_argument("--out", default="runs/teacher.jsonl",
                    help="JSONL of {qa_idx, source, lang_code, input, gold, teacher}; "
                         "existing qa_idx values in it are skipped (resume)")
    args = ap.parse_args()

    # --- data: identical split recipe to benchmark.py, but we take the TRAIN side ---
    df = load_dataset("pinzhenchen/wmt26-mist-sample")["train"].to_pandas()
    qa = df[df["task"] == "qa"].sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    qa["qa_idx"] = qa.index  # stable row key, shared with any future script using this split
    train = qa.iloc[int(len(qa) * 0.2):]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        if not 1 <= i <= n:
            sys.exit(f"bad --shard {args.shard!r}: need 1 <= i <= n")
        train = train.iloc[(len(train) * (i - 1)) // n: (len(train) * i) // n]
    if args.limit:
        train = train.head(args.limit)

    done: set[int] = set()
    out_path = Path(args.out)
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(line)["qa_idx"] for line in f}
        print(f"resuming: {len(done)} rows already in {args.out}")
    todo = train[~train["qa_idx"].isin(done)]
    print(f"train rows: {len(train)} after shard={args.shard}, limit={args.limit or 'none'}; "
          f"{len(todo)} to generate")
    if len(todo) == 0:
        print("nothing to do.")
        return

    # --- model: same stack as benchmark.py; device_map=auto so the MoE can spread over
    # multiple GPUs if the job requests them (one a100_80 fits it, two a100_40 also work) ---
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed

    tok = AutoTokenizer.from_pretrained(args.model)
    extra = {}
    if args.gptq_backend:
        from transformers import GPTQConfig
        extra["quantization_config"] = GPTQConfig(bits=4, backend=args.gptq_backend)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", **extra
    ).eval()

    # --- generation: same sampling recipe as benchmark.py, flushed per row ---
    set_seed(args.seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        for i, row in enumerate(todo.itertuples(index=False), 1):
            try:
                messages = build_messages(row.input, row.lang_code, lang_hint=args.lang_hint)
                prompt = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
                inputs = tok(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=20,
                    )
                answer = tok.decode(
                    out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()
            except Exception as e:  # noqa: BLE001 - one bad row must not lose the run
                print(f"  [{i}] qa_idx={row.qa_idx} FAILED: {type(e).__name__}: {e}", flush=True)
                answer = ""
            f.write(json.dumps(
                {"qa_idx": int(row.qa_idx), "source": row.source, "lang_code": row.lang_code,
                 "input": row.input, "gold": row.output, "teacher": answer},
                ensure_ascii=False) + "\n")
            f.flush()
            print(f"  [{i}/{len(todo)}] qa_idx={row.qa_idx}", flush=True)
    print(f"appended {len(todo)} rows -> {args.out}")


if __name__ == "__main__":
    main()
