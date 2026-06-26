# WMT26 MIST

Working notes for the WMT26 MIST shared task, using the sample dataset
[`pinzhenchen/wmt26-mist-sample`](https://huggingface.co/datasets/pinzhenchen/wmt26-mist-sample).

## Tasks

The shared task has three sub-tasks:

1. **Context-based question answering** — given a document in language X, answer questions about it, also in language X.
2. **Summarization** — given a document in language X, produce a summary in language Y.
3. **Open-ended generation** — given an open-ended question, produce a helpful, natural response.

## Columns: `task` and `source`

Two columns describe each example:

- **`task`** — has only **two** values in the data: `qa` and `sum`. It does *not* have a
  separate label for open-ended generation; those examples are tagged `qa`.
- **`source`** — the upstream dataset the example comes from.

To recover the **three conceptual sub-tasks** you need `source`, because the `task` column
folds Sub-task 1 (context-based QA) and Task 3 (open-ended generation) together under `qa`.

## Source dataset → task mapping

Counts are from the `train` split (`df.groupby(["task", "source"]).size()`):

| Conceptual sub-task | `task` label | `source` | rows |
|------|------|------|------|
| **Sub-task 1 — Context-based QA** | `qa` | `facebook/belebele` | 5700 |
| | `qa` | `copenlu/answerable_tydiqa` | 3112 |
| | `qa` | `FBK-MT/MCIF` (QA portion) | 880 |
| **Sub-task 2 — Summarization** | `sum` | `csebuetnlp/CrossSum` | 7026 |
| | `sum` | `esdurmus/wiki_lingua` | 1600 |
| | `sum` | `FBK-MT/MCIF` (summarization portion) | 400 |
| **Task 3 — Open-ended generation** | `qa` | `CohereLabs/aya_dataset` | 4741 |
| | `qa` | `wmt25-mist-oeg-gpt-4.1` | 460 |

Totals: `qa` = 14,893 rows, `sum` = 9,026 rows.

**Why each:**

- **`facebook/belebele`** — multilingual reading-comprehension QA → Sub-task 1.
- **`copenlu/answerable_tydiqa`** — TyDi QA, context-based QA across typologically diverse languages → Sub-task 1.
- **`csebuetnlp/CrossSum`** — cross-lingual summarization (BBC articles, any-to-any) → Sub-task 2.
- **`esdurmus/wiki_lingua`** — cross-lingual summarization from WikiHow → Sub-task 2.
- **`CohereLabs/aya_dataset`** — multilingual open-ended instruction/response → Task 3 (tagged `qa`).
- **`wmt25-mist-oeg-gpt-4.1`** — last year's MIST open-ended generation set → Task 3 (tagged `qa`).

**`FBK-MT/MCIF` spans both `qa` and `sum`** (880 QA + 400 summarization rows). It is built
on scientific-talk transcripts and is **cross-lingual**: the source document is in English
while the prompt and answer are in another language (German in the sample). Its QA rows use
prompts of the form *"Beantworte die folgende Frage … basierend auf dem englischen Inhalt"*,
and include **unanswerable** questions whose gold answer is `"Nicht zu beantworten."`
(cf. `answerable_tydiqa`). Use `task` (not `source`) to split MCIF into its QA vs. summarization halves.

## Setup

```bash
pip install datasets
```

```python
from datasets import load_dataset

ds = load_dataset("pinzhenchen/wmt26-mist-sample")
print(ds)               # splits and columns
print(ds["train"][0])   # first example
```
