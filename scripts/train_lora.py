"""LoRA SFT for the WMT26 MIST `qa` task on a Qwen3.5 causal LM.

Trains on the exact train 80% split `benchmark.py` evaluates against (same seed-42 split, same
`prompt_template.build_messages` prompt used at zero-shot inference time), so the resulting
adapter is scored with `benchmark.py --lora <this --out> --shots 0` on the identical held-out
dev 20% -- directly comparable to the 0-shot/few-shot baselines. No demonstrations are baked
into training prompts: SFT's job is to make the zero-shot behavior good directly, so "did
fine-tuning help" and "did few-shot help" stay isolated, independently comparable to the
0-shot baseline.

    python scripts/train_lora.py                          # full train split, Qwen3.5-9B
    python scripts/train_lora.py --limit 200 --epochs 1    # quick smoke test
    python scripts/train_lora.py --model Qwen/Qwen3.5-2B   # a smaller base
    python scripts/train_lora.py --data data/sft-distilled.jsonl   # distilled targets
                                       # (from scripts/filter_teacher.py; rows built from the
                                       # same train split, so the dev 20% stays held out)

Then evaluate the same way as every other experiment:
    python scripts/benchmark.py --lora <--out dir> --out runs/predictions-lora.csv
    python scripts/evaluate.py runs/predictions-lora.csv

Flags: --model · --data (training rows as JSONL with input/lang_code/output columns instead
of the HF train split) · --limit (0 = whole train split) · --epochs · --lr · --batch-size (per-device)
· --grad-accum · --max-length (tokens; longer rows are left-truncated, see
QAExampleDataset) · --r/--alpha/--dropout (LoRA hyperparameters) · --target-modules (regex,
see DEFAULT_TARGET_MODULES) · --seed · --out (adapter output dir) ·
--no-gradient-checkpointing · --save-steps (checkpoint interval, for resuming an interrupted
run).
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from prompt_template import build_messages

# peft accepts a regex string for target_modules, matched (re.fullmatch) against each module's
# full dotted name. Scoped to the text decoder only: every attention layer's q/k/v/o_proj and
# every layer's MLP gate/up/down_proj (present regardless of attention-vs-delta-net layer type
# -- Qwen3.5 hybridizes full self-attention with gated linear-attention layers per decoder
# layer). Deliberately excludes model.visual.* (vision tower, unused -- we're text-only) and
# the Gated DeltaNet layers' own specialized projections (in_proj_qkv/in_proj_z/in_proj_b/
# in_proj_a/out_proj -- distinct names, so they never collide with this regex; out of scope for
# a first LoRA pass, the shared per-layer MLP LoRA still adapts those layers). Verified against
# transformers/models/qwen3_5/modeling_qwen3_5.py: the text decoder lives at
# `<top-level model>.model.language_model.*`.
DEFAULT_TARGET_MODULES = (
    r"^model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)


class QAExampleDataset(Dataset):
    """Tokenized (input_ids, labels) pairs, one per training row.

    Each row becomes the same chat messages `benchmark.py` builds at inference time
    (`build_messages(..., lang_hint=True)`, zero-shot -- no demonstrations), plus the gold
    `output` as the final assistant turn. Labels are `-100` (ignored by the loss) over the
    prompt span and real token ids over the answer span, so the model is only ever trained to
    predict the answer, not to reproduce the instruction. The boundary is found by tokenizing
    the prompt-only prefix (`add_generation_prompt=True`) and the prompt+answer full text
    separately and taking the prefix's token length -- verified against Qwen3.5's tokenizer
    that the prefix's tokenization is an exact prefix of the full text's tokenization (ChatML
    template, special-token turn boundaries don't re-merge across the split point), so this
    doesn't depend on the chat template exposing an assistant-token-mask helper.

    Rows longer than `max_length` tokens are left-truncated (the earliest tokens are dropped),
    which always keeps the entire answer intact (it's the last chunk of the full sequence) --
    the only effect is that a truncated row's visible "prompt" starts mid-passage instead of at
    the system turn. This is a deliberate, documented lossy edge case for the rare very-long
    rows (e.g. FBK-MT/MCIF passages) rather than something engineered around.
    """

    def __init__(self, rows, tok, max_length: int):
        self.examples = []
        n_truncated = 0
        for row in rows:
            messages = build_messages(row.input, row.lang_code, lang_hint=True)
            prefix_text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            full_text = tok.apply_chat_template(
                messages + [{"role": "assistant", "content": row.output}],
                tokenize=False, add_generation_prompt=False, enable_thinking=False,
            )
            prefix_ids = tok(prefix_text, add_special_tokens=False)["input_ids"]
            full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
            prefix_len = len(prefix_ids)
            if len(full_ids) > max_length:
                cut = len(full_ids) - max_length
                full_ids = full_ids[cut:]
                prefix_len = max(0, prefix_len - cut)
                n_truncated += 1
            labels = [-100] * prefix_len + full_ids[prefix_len:]
            self.examples.append((full_ids, labels))
        if n_truncated:
            print(f"  {n_truncated}/{len(rows)} rows truncated to {max_length} tokens "
                  f"(kept the tail, i.e. the answer -- see QAExampleDataset docstring)")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids, labels = self.examples[idx]
        return {"input_ids": input_ids, "labels": labels}


def collate(features, pad_token_id: int) -> dict:
    """Pad a batch of variable-length (input_ids, labels) to the batch's own max length --
    labels padded with -100 (ignored), input_ids padded with pad_token_id, attention_mask
    marking real vs pad tokens."""
    max_len = max(len(f["input_ids"]) for f in features)
    input_ids, attention_mask, labels = [], [], []
    for f in features:
        pad = max_len - len(f["input_ids"])
        input_ids.append(f["input_ids"] + [pad_token_id] * pad)
        attention_mask.append([1] * len(f["input_ids"]) + [0] * pad)
        labels.append(f["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--data", default=None,
                    help="JSONL of training rows (columns: input, lang_code, output -- e.g. "
                         "from scripts/filter_teacher.py) used INSTEAD of the HF train split. "
                         "The file must be built from train-split rows only; nothing here "
                         "re-checks the 80/20 boundary.")
    ap.add_argument("--limit", type=int, default=0, help="0 = whole train split")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=2, help="per-device batch size")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=2048,
                    help="tokens; longer rows are left-truncated (kept: the tail, i.e. the "
                         "gold answer -- see QAExampleDataset docstring)")
    ap.add_argument("--r", type=int, default=16, help="LoRA rank")
    ap.add_argument("--alpha", type=int, default=32, help="LoRA alpha")
    ap.add_argument("--dropout", type=float, default=0.05, help="LoRA dropout")
    ap.add_argument("--target-modules", default=DEFAULT_TARGET_MODULES,
                    help="regex passed to peft.LoraConfig(target_modules=...); default scopes "
                         "to the text decoder's attention/MLP projections only, see "
                         "DEFAULT_TARGET_MODULES")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="adapters/qwen3.5-9b-qa-lora",
                    help="directory the trained adapter (+tokenizer) is saved to; also where "
                         "Trainer writes checkpoints, so re-running with the same --out "
                         "resumes an interrupted run instead of restarting")
    ap.add_argument("--no-gradient-checkpointing", action="store_true",
                    help="disable gradient checkpointing (default on -- trades compute for the "
                         "activation-memory headroom the 9B model needs on a single a40)")
    ap.add_argument("--save-steps", type=int, default=200,
                    help="checkpoint every N steps, so an interrupted run (time limit, node "
                         "failure) can resume instead of losing everything -- same principle "
                         "benchmark.py applies by flushing every prediction row immediately")
    args = ap.parse_args()

    # --- data: qa subset, 80/20 split -- identical to benchmark.py's, so LoRA only ever trains
    # on the train 80% and is evaluated on the same held-out dev 20% as every other experiment.
    # With --data, the rows come from a prepared JSONL instead (same column names; built from
    # train-split rows upstream, e.g. by scripts/filter_teacher.py).
    if args.data:
        import pandas as pd
        train = pd.read_json(args.data, lines=True)
    else:
        df = load_dataset("pinzhenchen/wmt26-mist-sample")["train"].to_pandas()
        qa = df[df["task"] == "qa"].sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        train = qa.iloc[int(len(qa) * 0.2):]
    if args.limit:
        train = train.head(args.limit)
    print(f"train examples: {len(train)}  model={args.model}  data={args.data or 'HF train split'}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dataset = QAExampleDataset(list(train.itertuples(index=False)), tok, args.max_length)

    set_seed(args.seed)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # required for grad-checkpointing through a frozen base
    model = get_peft_model(model, LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout,
        target_modules=args.target_modules, task_type="CAUSAL_LM", bias="none",
    ))
    model.print_trainable_parameters()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    resume = any(out_dir.glob("checkpoint-*"))

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to=[],
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda features: collate(features, tok.pad_token_id),
    )
    trainer.train(resume_from_checkpoint=resume)

    model.save_pretrained(str(out_dir))  # adapter-only (PeftModel.save_pretrained), not the base model
    tok.save_pretrained(str(out_dir))
    print(f"adapter saved -> {out_dir}")
    print(f"evaluate it with:  python scripts/benchmark.py --lora {out_dir} "
          f"--out runs/predictions-lora.csv  &&  "
          f"python scripts/evaluate.py runs/predictions-lora.csv")


if __name__ == "__main__":
    main()
