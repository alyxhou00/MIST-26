"""vLLM variant of scripts/teacher_generate.py -- same split, prompts, resume semantics
and output schema; batched generation instead of row-by-row.

Exists because the Qwen3.5-122B-A10B-GPTQ-Int4 teacher cannot be loaded through
transformers+gptqmodel: the default Marlin kernel rejects its out_features=1 layer
(job 3859381) and the torch fallback backend hits CUDA illegal memory accesses
(job 3859398). vLLM ships mature GPTQ-MoE kernels for the Qwen family, and its batched
inference is a large throughput win over the one-row-at-a-time transformers loop anyway.

    python scripts/teacher_generate_vllm.py --limit 15          # smoke (same 15 rows)
    python scripts/teacher_generate_vllm.py --shard 1/2 --out runs/teacher122b-s1of2.jsonl

Rows are generated in chunks of --chunk (default 128) and appended to --out after each
chunk, so an interrupted job loses at most one chunk and resumes via qa_idx like the
transformers variant.
"""

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset

from prompt_template import build_messages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-122B-A10B-GPTQ-Int4")
    ap.add_argument("--tensor-parallel", type=int, default=2,
                    help="GPUs to shard over (the int4 122B needs 2x a100_80)")
    ap.add_argument("--shard", default=None, metavar="I/N")
    ap.add_argument("--limit", type=int, default=0, help="0 = whole train split")
    ap.add_argument("--chunk", type=int, default=128,
                    help="rows per generate/write cycle (resume granularity)")
    ap.add_argument("--lang-hint", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=8192,
                    help="vLLM context window; prompts longer than this minus "
                         "--max-new-tokens would be rejected, 8192 covers the long MCIF rows")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42,
                    help="split seed AND sampling seed; must stay 42 to match "
                         "benchmark.py's 80/20 split or the dev set leaks into training")
    ap.add_argument("--out", default="runs/teacher-vllm.jsonl")
    args = ap.parse_args()

    # --- data: identical to teacher_generate.py ---
    df = load_dataset("pinzhenchen/wmt26-mist-sample")["train"].to_pandas()
    qa = df[df["task"] == "qa"].sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    qa["qa_idx"] = qa.index
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

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel,
              max_model_len=args.max_model_len, seed=args.seed)
    sampling = SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=20,
                              max_tokens=args.max_new_tokens, seed=args.seed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(todo.itertuples(index=False))
    with open(out_path, "a", encoding="utf-8") as f:
        for start in range(0, len(rows), args.chunk):
            batch = rows[start:start + args.chunk]
            conversations = [build_messages(r.input, r.lang_code, lang_hint=args.lang_hint)
                             for r in batch]
            outs = llm.chat(conversations, sampling,
                            chat_template_kwargs={"enable_thinking": False})
            for r, o in zip(batch, outs):
                answer = o.outputs[0].text.strip() if o.outputs else ""
                f.write(json.dumps(
                    {"qa_idx": int(r.qa_idx), "source": r.source, "lang_code": r.lang_code,
                     "input": r.input, "gold": r.output, "teacher": answer},
                    ensure_ascii=False) + "\n")
            f.flush()
            print(f"  [{min(start + args.chunk, len(rows))}/{len(rows)}]", flush=True)
    print(f"appended {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
