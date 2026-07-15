"""Inference on the official WMT26 MIST test set -> submission-format JSONL.

The test prompts (pinzhenchen/wmt26-mist-test, data/tests.jsonl -- see README for the
download command) are SELF-CONTAINED: context, constraints and question in one string,
often with embedded format instructions ("answer in one sentence", "in 150 words", ...).
So unlike benchmark.py, which builds prompts from the sample data with our template, this
script feeds each prompt verbatim as a single user turn. No system turn by default:
--lang-hint adds the same "Respond in <language>." turn used everywhere else (legal at
test time -- question_lang is given -- but it can fight instructions embedded in the
prompt, so it's opt-in pending a dev A/B).

    python scripts/run_test.py                                  # all qa rows, base model
    python scripts/run_test.py --task qa-oeg --lang bho --limit 10   # smoke: surprise language
    python scripts/run_test.py --lora adapters/... --shard 2/3      # middle third, SFT'd model

Output is one {"id": ..., "output": ...} JSON object per line (the submission format).
Rows already present in --out are skipped on restart, so a run that hits the wall clock
can be resubmitted with the same --out and it resumes where it stopped. --shard i/n cuts
the (filtered) rows into n contiguous chunks for parallel jobs over multiple nodes; give
each shard its own --out and concatenate afterwards.

Known test-set quirks (v. 14 July 2026), both reportable to the organizers:
  * the 100 qa-oeg English rows (qa-oeg_1..100_eng_eng) have an EMPTY prompt string.
    Generating from an empty prompt would produce unrelated text, so those rows get output ""
    and a warning; re-run once the organizers fix the data.
  * every qa-context prompt is double-escaped and carries LITERAL backslash-n at its section
    boundaries -- see --unescape and TEST_SET_ANALYSIS.md section 2.
"""

import argparse
import json
import sys
from pathlib import Path

from prompt_template import TEST_LANG_NAMES, system_turn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-file", default="data/tests.jsonl",
                    help="official test JSONL ({id, prompt, task, question_lang} per line)")
    ap.add_argument("--task", default="qa-context,qa-oeg",
                    help="comma-separated task values to keep (default: both qa tasks; "
                         "sum-sum is the teammate's subtask)")
    ap.add_argument("--lang", default=None,
                    help="only rows with this question_lang (bare 3-letter test code, e.g. 'bho')")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="process the i-th of n contiguous chunks of the filtered rows "
                         "(1-based, e.g. '2/3'); applied before --limit")
    ap.add_argument("--limit", type=int, default=0, help="0 = all filtered rows")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--lora", default=None,
                    help="path to a trained LoRA adapter (from scripts/train_lora.py) to load "
                         "on top of --model before generation. Omit for the base model.")
    ap.add_argument("--unescape", action="store_true",
                    help="turn literal backslash-n in the prompt into real newlines. The "
                         "official file is double-escaped: ALL 8,640 qa-context prompts carry "
                         "the two characters '\\' 'n' at their section boundaries (passage / "
                         "question / instructions), so by default the model reads '\\n\\n' as "
                         "text in 79%% of qa rows. Off by default because it edits the "
                         "official input and there is no dev proxy to A/B it on -- see "
                         "TEST_SET_ANALYSIS.md section 2")
    ap.add_argument("--lang-hint", action=argparse.BooleanOptionalAction, default=False,
                    help="prepend the shared 'Respond in <language>.' system turn derived from "
                         "question_lang (default: OFF -- the test prompt is self-contained)")
    ap.add_argument("--max-new-tokens", type=int, default=512,
                    help="default 512, not benchmark.py's 256: qa-oeg prompts routinely ask "
                         "for 120-180 words, which needs more than 256 tokens in most "
                         "non-Latin scripts")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="sampling temperature (default matches Qwen3.5-9B's card)")
    ap.add_argument("--top-p", type=float, default=0.8,
                    help="nucleus sampling top-p (default matches Qwen3.5-9B's card)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/test-outputs.jsonl",
                    help="submission-format JSONL; existing ids in it are skipped (resume)")
    args = ap.parse_args()

    # --- rows: filter -> shard -> limit, in that order ---
    tasks = {t.strip() for t in args.task.split(",") if t.strip()}
    with open(args.test_file, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    rows = [r for r in rows if r["task"] in tasks]
    if args.lang:
        rows = [r for r in rows if r["question_lang"] == args.lang]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        if not 1 <= i <= n:
            sys.exit(f"bad --shard {args.shard!r}: need 1 <= i <= n")
        rows = rows[(len(rows) * (i - 1)) // n: (len(rows) * i) // n]
    if args.limit:
        rows = rows[: args.limit]

    # Resume: skip ids already answered in --out (a resubmitted job appends the rest).
    done: set[str] = set()
    out_path = Path(args.out)
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f}
        print(f"resuming: {len(done)} ids already in {args.out}")
    todo = [r for r in rows if r["id"] not in done]
    print(f"test rows: {len(rows)} after filters (task={sorted(tasks)}, lang={args.lang}, "
          f"shard={args.shard}, limit={args.limit or 'none'}), {len(todo)} to generate")
    if not todo:
        print("nothing to do.")
        return

    # --- model: identical stack to benchmark.py (Qwen3.5 is multimodal; text-only use) ---
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer, set_seed

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    if args.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora)
    model = model.eval()

    # --- generation: same sampling recipe as benchmark.py, flushed per row ---
    set_seed(args.seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_empty = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for i, row in enumerate(todo, 1):
            if not row["prompt"].strip():  # the 100 empty eng qa-oeg rows (see module docstring)
                n_empty += 1
                f.write(json.dumps({"id": row["id"], "output": ""}, ensure_ascii=False) + "\n")
                f.flush()
                continue
            try:
                prompt_text = row["prompt"]
                if args.unescape:
                    prompt_text = prompt_text.replace(chr(92) + "n", "\n")
                messages = []
                if args.lang_hint:
                    name = TEST_LANG_NAMES.get(row["question_lang"], row["question_lang"])
                    messages.append(system_turn(name))
                messages.append({"role": "user", "content": prompt_text})
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
                pred = tok.decode(
                    out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip()
            except Exception as e:  # noqa: BLE001 - one bad row must not lose the run
                print(f"  [{i}] {row['id']} FAILED: {type(e).__name__}: {e}", flush=True)
                pred = ""
            f.write(json.dumps({"id": row["id"], "output": pred}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  [{i}/{len(todo)}] {row['id']}", flush=True)
    if n_empty:
        print(f"WARNING: {n_empty} rows had an empty prompt -> wrote empty outputs "
              f"(known test-set issue, see docstring)", file=sys.stderr)
    print(f"appended {len(todo)} rows -> {args.out}")


if __name__ == "__main__":
    main()
