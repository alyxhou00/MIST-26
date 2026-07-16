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
    python scripts/run_test.py --task qa-context --shots 3          # the routed qa-context arm

--shots N prepends N demonstrations from the sample data as completed user/assistant turns,
matched to the row on task->source and question_lang (see make_shot_picker). The demos are in
the *sample* format while the test prompt is self-contained prose, so they demonstrate answer
shape and language, not prompt format. Per TEST_SET_ANALYSIS 8/E this is the qa-context arm of
the routed submission (dev tydiqa, the faithful proxy: 3-shot 38.94 vs adapter 19.53), while
qa-oeg wants --lora at 0 shots -- demos and the adapter do not stack (dev 21.64 vs 27.64).

Output is one {"id": ..., "output": ...} JSON object per line (the submission format).
Rows already present in --out are skipped on restart, so a run that hits the wall clock
can be resubmitted with the same --out and it resumes where it stopped. --shard i/n cuts
the (filtered) rows into n contiguous chunks for parallel jobs over multiple nodes; give
each shard its own --out and concatenate afterwards.

Known test-set quirks (data revision `5950311`, re-downloaded 2026-07-16):
  * 8 English qa-oeg rows (qa-oeg_93..100_eng_eng) ship UNSUBSTITUTED template placeholders --
    "the national sport in {country}", "diminutives in {language}" -- where every other
    language has a real value filled in. Passed through verbatim with a warning: substituting
    a guess would invent an input the organizers did not write. TEST_SET_ANALYSIS.md section 6.
  * every qa-context prompt is double-escaped and carries LITERAL backslash-n at its section
    boundaries -- see --unescape and TEST_SET_ANALYSIS.md section 2.
  * the 100 empty English qa-oeg prompts are FIXED upstream as of `5950311`. The empty-prompt
    guard below is kept as a safety net (it costs nothing and a future revision could regress),
    but it is dead code on this revision.
"""

import argparse
import json
import re
import sys
import zlib
from pathlib import Path

from prompt_template import TEST_LANG_NAMES, TEST_TASK_SOURCES, system_turn

# Unsubstituted template slots in the official prompts -- `{country}`, `{language}`. Only the
# 8 English qa-oeg rows qa-oeg_93..100 have them (TEST_SET_ANALYSIS 6); detected rather than
# repaired, so a run reports the damage instead of inventing an input.
_PLACEHOLDER = re.compile(r"\{(?:country|language)\}")


def make_shot_picker(pool, k: int, seed: int):
    """Return a function picking k few-shot demonstrations from the sample data for a test row.

    The test set has no golds, so demonstrations can only come from the sample data
    (`pinzhenchen/wmt26-mist-sample`, qa split) -- a different distribution from the test
    prompts (templated `context + question` vs self-contained conversational prose). That
    mismatch is inherent to few-shot here and is the main caveat on this flag: the demos
    teach answer *shape and language*, not prompt format. See the module docstring.

    Unlike benchmark.py's picker there is no dev/train split to respect -- the official test
    rows are disjoint from the sample data, so the whole qa split is drawn from.

    Matching, per TEST_SET_ANALYSIS 5b:
      * `task` -> `source` via TEST_TASK_SOURCES (belebele excluded: multiple choice, absent
        from the test set, and the wrong format to demonstrate);
      * `question_lang` (bare 'hin') -> `lang_code` ('hin_Deva') by comparing the prefix.
        Both name the *output* language on their own side, so this is a like-for-like match.

    Tiers: (task-sources, same language) -> (task-sources, any language) -> whole pool. The
    language fallback is reported, not silent: it is reached only by a test language with no
    sample rows -- in the final file that means **bho** -- and demonstrating an answer in the
    wrong language to a model already documented to drift bho->hin (TEST_SET_ANALYSIS 7.2) is
    a real risk, not a neutral default. Use --shots-require-lang to force zero-shot instead.

    Seeded per row from a hash of its `id` (not a shared RNG consumed in iteration order), so
    a row gets the same shots regardless of --shard/--limit/--lang or row order -- the same
    property benchmark.py gets by hashing the input text. `id` rather than the prompt because
    it is stable under --unescape, which rewrites the prompt but must not move the shots.
    """
    pool = pool.assign(lang=pool["lang_code"].str.split("_").str[0])
    fell_back: set[str] = set()

    def pick(task: str, question_lang: str, row_id: str,
             require_lang: bool = False) -> list[tuple[str, str]]:
        sources = TEST_TASK_SOURCES.get(task)
        cand = pool[pool["source"].isin(sources)] if sources else pool
        same_lang = cand[cand["lang"] == question_lang]
        if len(same_lang) >= k:
            chosen = same_lang
        else:
            if require_lang:
                return []
            if question_lang not in fell_back:
                fell_back.add(question_lang)
                print(f"WARNING: only {len(same_lang)} sample rows in {question_lang!r} for "
                      f"task {task!r} (need {k}); falling back to demonstrations in OTHER "
                      f"languages. They may pull the output language away from "
                      f"{question_lang!r} -- see TEST_SET_ANALYSIS 7.2 (bho->hin drift). "
                      f"--shots-require-lang runs these rows zero-shot instead.",
                      file=sys.stderr, flush=True)
            chosen = cand if len(cand) >= k else pool
        picked = chosen.sample(
            n=min(k, len(chosen)),
            random_state=zlib.crc32(f"{seed}:{row_id}".encode("utf-8")),
        )
        return list(zip(picked["input"], picked["output"]))

    return pick


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
    ap.add_argument("--shots", type=int, default=0,
                    help="few-shot demonstrations per row, drawn from the sample data "
                         "(--shots-file) and matched on task->source + question_lang. "
                         "Inserted as completed user/assistant turns before the prompt. "
                         "Default 0 = zero-shot. NOTE the demos are sample-format while the "
                         "test prompt is self-contained prose -- they teach answer shape and "
                         "language, not prompt format. Per TEST_SET_ANALYSIS 8/E this is the "
                         "intended setting for qa-context (dev tydiqa: 3-shot 38.94 vs "
                         "adapter 19.53); qa-oeg prefers --lora with 0 shots.")
    ap.add_argument("--shots-file", default=None,
                    help="local qa sample data for --shots (CSV/JSONL with source, lang_code, "
                         "input, output). Default: load pinzhenchen/wmt26-mist-sample from the "
                         "Hub, as benchmark.py does.")
    ap.add_argument("--shots-require-lang", action="store_true",
                    help="with --shots, run a row zero-shot rather than demonstrate in a "
                         "language other than its question_lang (affects bho, which has no "
                         "sample rows at all -- see make_shot_picker)")
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

    # --- few-shot pool (only loaded when asked; keeps the zero-shot path dependency-free) ---
    pick_shots = None
    if args.shots:
        if args.lora:
            print("WARNING: --shots with --lora. Measured on dev, demonstrations and the SFT "
                  "adapter do NOT stack (21.64 vs 27.64 for 3-shot base): the adapter was "
                  "trained on zero-shot-formatted inputs, so demos put it off-distribution. "
                  "Route instead -- adapter for qa-oeg, 3-shot base for qa-context "
                  "(TEST_SET_ANALYSIS 8/E).", file=sys.stderr, flush=True)
        if args.shots_file:
            import pandas as pd
            sample = (pd.read_json(args.shots_file, lines=True)
                      if args.shots_file.endswith(".jsonl")
                      else pd.read_csv(args.shots_file))
        else:
            from datasets import load_dataset
            sample = load_dataset("pinzhenchen/wmt26-mist-sample")["train"].to_pandas()
        sample = sample[sample["task"] == "qa"] if "task" in sample.columns else sample
        missing = {"source", "lang_code", "input", "output"} - set(sample.columns)
        if missing:
            sys.exit(f"--shots pool is missing column(s) {sorted(missing)}")
        pick_shots = make_shot_picker(sample, args.shots, args.seed)
        print(f"few-shot: {args.shots} demos/row from {len(sample)} qa sample rows "
              f"(sources used: {sorted(set().union(*TEST_TASK_SOURCES.values()))})")

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
    n_placeholder = 0
    with open(out_path, "a", encoding="utf-8") as f:
        for i, row in enumerate(todo, 1):
            if _PLACEHOLDER.search(row["prompt"]):  # 8 eng qa-oeg rows (see module docstring)
                n_placeholder += 1
            if not row["prompt"].strip():  # fixed upstream in `5950311`; guard kept as a net
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
                if pick_shots:
                    for ex_input, ex_output in pick_shots(
                        row["task"], row["question_lang"], row["id"],
                        require_lang=args.shots_require_lang,
                    ):
                        messages.append({"role": "user", "content": ex_input})
                        messages.append({"role": "assistant", "content": ex_output})
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
        print(f"WARNING: {n_empty} rows had an empty prompt -> wrote empty outputs. This was "
              f"fixed upstream in `5950311`; seeing it again means --test-file is a stale "
              f"revision (re-download per the README) or the organizers regressed the data.",
              file=sys.stderr)
    if n_placeholder:
        print(f"WARNING: {n_placeholder} rows contain an unsubstituted template placeholder "
              f"({{country}}/{{language}}) and were sent to the model verbatim -- their outputs "
              f"will be about a literal '{{country}}'. Known upstream bug in the English qa-oeg "
              f"block, TEST_SET_ANALYSIS.md section 6.", file=sys.stderr)
    print(f"appended {len(todo)} rows -> {args.out}")


if __name__ == "__main__":
    main()
