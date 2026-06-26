# WMT26 MIST

Working notes for the WMT26 MIST shared task, using the sample dataset
[`pinzhenchen/wmt26-mist-sample`](https://huggingface.co/datasets/pinzhenchen/wmt26-mist-sample).

## Tasks

The shared task has three tasks:

**Sub-task 1.** **Context-based question answering** — given a document in language X, answer questions about it, also in language X. <br>
**Sub-task 2.** **Summarization** — given a document in language X, produce a summary in language Y.<br>
**Task 3.** **Open-ended generation** — given an open-ended question, produce a helpful, natural response.

## Columns: `task` and `source`

Two columns describe each example:

- **`task`** — has only **two** values in the data: `qa` and `sum`. It does *not* have a
  separate label for open-ended generation; those examples are tagged `qa`.
- **`source`** — the upstream dataset the example comes from.

## Source dataset → task mapping

Counts are from the `train` split (`df.groupby(["task", "source"]).size()`):

| Conceptual sub-task                | `task` label | `source`                              | rows |
|------------------------------------|--------------|---------------------------------------|------|
| **Sub-task 1 — Context-based QA**  | `qa`         | `facebook/belebele`                   | 5700 |
|                                    | `qa`         | `copenlu/answerable_tydiqa`           | 3112 |
|                                    | `qa`         | `FBK-MT/MCIF` (QA portion)            | 880  |
| **Sub-task 2 — Summarization**     | `sum`        | `csebuetnlp/CrossSum`                 | 7026 |
|                                    | `sum`        | `esdurmus/wiki_lingua`                | 1600 |
|                                    | `sum`        | `FBK-MT/MCIF` (summarization portion) | 400  |
| **Task 3 — Open-ended generation** | `qa`         | `CohereLabs/aya_dataset`              | 4741 |
|                                    | `qa`         | `wmt25-mist-oeg-gpt-4.1`              | 460  |

Totals: `qa` = 14,893 rows, `sum` = 9,026 rows.

## Load Dataset

```bash
pip install datasets
```

```python
from datasets import load_dataset

ds = load_dataset("pinzhenchen/wmt26-mist-sample")
print(ds["train"])   # the only split -- train
```

## 0. Zero-shot QA benchmark (`Qwen/Qwen3.5-2B`)

[`benchmark.py`](benchmark.py) is a minimal zero-shot benchmark of the `qa` sub-task: it splits
the examples **80/20 train/dev** (seed 42), runs the model on the dev half via its chat template,
and reports **chrF**. `Qwen/Qwen3.5-2B` is multimodal but used here text-only (each `input` is one
user turn); it needs a recent `transformers` (see `requirements.txt`).

```bash
python benchmark.py --limit 50     # quick check
python benchmark.py                # full dev split
```

On the Alex cluster (NHR@FAU), set up a venv once on the login node, then `sbatch job.sbatch`
(see the comments in [`job.sbatch`](job.sbatch)).
