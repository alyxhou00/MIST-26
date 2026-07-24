# WMT26 MIST — Implementation Notes

> [!NOTE]
> The design decisions, post-mortems, and infrastructure record behind the whole **qa**
> experiment program (context QA + OEG; the `sum` sub-task is a teammate's). Kept for the
> paper write-up. Results themselves live in the experiment logs
> ([EXPERIMENTS_NEW.md](EXPERIMENTS_NEW.md), current; [EXPERIMENTS.md](EXPERIMENTS.md),
> closed); this file is the *why* and *how*, not the score tables. Everything below is
> backed by a SLURM job ID in those logs and a committed log in `logs/`.
> Co-authored by Claude.

## 1. Architecture & routing

One **Qwen3.5-9B** base, adapted at inference time with a LoRA adapter picked per task
("routing"). The official test set gives `task` ∈ {qa-context, qa-oeg, sum-sum} on every
row, so routing on it is explicitly legal.

**Final serving config** (the primary submission):

| Test task | Serving config | Why |
|---|---|---|
| `qa-context` | base + **C+D-small adapter (job 3880753)**, 0-shot, no lang-hint | Best-or-tied on every qa-context proxy on the clean v2 dev, and it carries the bho fix (§7.3). qa-context is 79% of the qa test set. |
| `qa-oeg` | base + **same adapter** | One model for everything. C-only edges it on the 90-row OEG column alone, so C-only is at most a qa-oeg-only variant, not the primary (§7.2). |
| `sum-sum` | teammate's system (same 9B base if we submit jointly) | not ours |

Both qa routes are the **same** adapter — one adapter, 0-shot, no lang-hint, so its dev
numbers are directly comparable to what `run_test.py` feeds.

The routing *principle* — a fine-tuned adapter beats prompting — was established on the v1
dev (gold-LoRA swept MCIF and OEG) and re-verified on the clean v2 split (the plain-v2
adapter beat both base 0-shot and 3-shot on every column). Two things settled along the way:

- **Never stack the adapter with few-shot demos.** They fight — the adapter was fine-tuned
  0-shot, so demos are out-of-distribution for it (§4).
- **Distillation was tried and lost** to plain gold SFT on both routing-relevant columns (§5.4).

**Adapter inference must always carry runaway protection** (§6.1). Planned but not built: a
GlotLID language gate that resamples wrong-language output (roadmap F).

## 2. The 10B parameter budget

The organizers cap the **total parameters of all deployed components** at 10B (MoE counts
total, not active):

| Component | Params | Note |
|---|---|---|
| Qwen3.5-9B base | 9,438,911,728 | shared by all routes; one copy |
| LoRA adapter, r=16 (each) | 29,097,984 | measured; ~0.31% of base |
| language-ID gate (fastText / GlotLID) | <1M | compressed model <1MB on disk |
| **Total, 1 adapter** | **≈9.47B** | comfortable headroom |

- Extra adapters are nearly free (0.03B each), so per-task LoRAs on one shared base fit easily.
- The teachers (35B, 122B) do **not** count — they are never deployed, only their outputs are,
  as training data.
- The one hard constraint: a **joint submission with the sum teammate must share the same 9B
  base**. Two ~9B bases would be ~19B and blow the cap.

## 3. Prompting: few-shot implementation

### Why prompting was needed (the zero-shot failure modes)

Error analysis on the zero-shot baseline (`error_analysis.py`) found four recurring problems,
none of which an instruction alone reliably fixes:

- **Script mismatch** — gold in a non-Latin script (Hindi, Thai, …) but the prediction comes
  back mostly Latin. `--lang-hint` asks for the target language, but zero-shot never shows what
  a concise answer in that script looks like.
- **Length mismatch** — predictions run 3×+ longer than gold. The model is a chat assistant
  trained to explain; the golds are terse extractive answers. This is style, not something an
  instruction teaches.
- **Task conventions are invisible** — e.g. MCIF's "unable to answer" convention, or belebele's
  "match the option text exactly". The model only picks these up from seeing examples.
- **Language confusion** — on cross-lingual rows (English question, non-English answer) the
  model sometimes answers in English. In-context examples are more robust than a single hint.

The through-line: few-shot demonstrations teach **answer format by imitation**, which is
exactly what these failures need. What they *don't* teach is content knowledge — so open-ended
generation (aya) stays roughly flat while format-driven sources (belebele, tydiqa) jump.

### How shot selection works

- **Three-tier fallback:** same `(source, lang_code)` → same `source` → whole train pool. The
  first tier demonstrates both the task format and the target language; the fallbacks cover thin
  strata.
- **Deterministic per example:** shots are seeded from a hash of the input text
  (`crc32("{seed}:{input}")`), not a global RNG. So `--limit`/`--source`/`--lang`/row order
  never change which shots an example gets, and A/B runs stay comparable.
- **Leakage guard:** byte-identical train/dev duplicates are dropped at each tier before
  sampling (53 of 2,978 v1 dev rows had one — sampling a verbatim duplicate would hand the model
  its own answer).
- **Real chat turns:** shots go in as completed user/assistant turns in the messages list, not
  as concatenated text — chat models learn from turn structure.
- **No thinking:** `enable_thinking=False` (Qwen's non-thinking recommendation), and shots carry
  no `<think>` blocks, so the model imitates a direct answer.

Pool is the train 80% only, so a dev example can never demonstrate itself.

## 4. LoRA fine-tuning

- **Where LoRA attaches:** the decoder's attention (`q/k/v/o_proj`) and every layer's MLP
  (`gate/up/down_proj`). Qwen3.5 is a hybrid stack (some layers full self-attention, some gated
  linear attention), but every layer has an MLP, so that half of the adaptation reaches the
  whole network.
- **Trained 0-shot, same prompt format as `benchmark.py --shots 0`.** This keeps "did SFT help"
  independent of "did few-shot help", and — more importantly — matches training to how the
  adapter is served.
- **Format has to match at train and inference time.** Because the adapter is trained 0-shot,
  feeding it few-shot demos is OOD: the v1 adapter+3-shot run scored *worse than either
  component alone* on every source but OEG. Don't stack them.
- **Test-format training.** The v2 adapters are trained `--no-lang-hint`, which is what
  `run_test.py` actually feeds. Dropping the hint costs the adapter essentially nothing (v1 job
  3866054 moved ≤0.6 COMBINED vs its hinted self on every routing column), so hinted and no-hint
  numbers are comparable — the whole routing table transfers to no-hint inference.
- **Gold SFT is complementary to prompting.** The gold adapter wins belebele, MCIF and OEG but
  looked like it "collapsed" on tydiqa (chrF 38.94→19.53). That collapse was later shown to be a
  runaway-generation artifact, not a capability loss (§6.1).

## 5. Distillation pipeline

Sequence-level knowledge distillation: a teacher generates on the **train** split (never dev),
its answers are quality-filtered against the golds, and a fresh student LoRA is trained on the
filtered mix. It was fully run and **lost to plain gold SFT** (§5.4) — the sections below are
the reasoning of record.

### 5.1 Teacher selection

Three teachers on the same 15 deterministic train rows (the seed-42 split makes smokes
row-by-row comparable):

| Teacher | Infra | Result |
|---|---|---|
| Qwen3.5-35B-A3B bf16 | 1× a100_80, transformers | fluent, but hallucinated a Japanese quiz answer, an NHL draft year, and some geography |
| Qwen3.5-27B bf16 | 1× a100_80, transformers | slightly better (fixed the Japanese answer, better MC compliance), own hallucinations |
| Qwen3.5-122B-A10B GPTQ-Int4 | 2× a100_80, **vLLM** | the only teacher to get **both** knowledge probes right |

- **Decision:** 122B for aya+oeg (knowledge-grounded, open-ended — where a better teacher raises
  the filter pass rate); 35B for the whole corpus (belebele/tydiqa/MCIF, where teacher choice was
  assumed to matter less — an assumption §5.5 later dents).
- **Infra note:** the 122B GPTQ checkpoint is unrunnable through transformers+gptqmodel (Marlin
  kernel rejects an `out_features=1` layer; the torch fallback hits CUDA illegal-memory errors).
  vLLM runs it natively, and its batched decode measured **~250× faster per row** than the serial
  transformers loop.

### 5.2 The gold filter is load-bearing

`filter_teacher.py` keeps a teacher row only if it scores well against the gold: per-row
sentence **chrF ≥ C OR BERTScore ≥ B**, thresholds calibrated per source.

- **Why OR:** chrF alone kills verbose-but-correct answers (a full answer scored against a short
  gold); BERTScore alone is too lenient on fluent, on-topic hallucinations. Two calibrated
  thresholds OR'd beat either alone.
- **The "belebele always fails, so its gold is safe" claim was wrong.** Measured: belebele passes
  **33.3%**, and every pass replaces a `2: <option>` gold with prose — wrecking the exact format
  gold SFT learned best. chrF really is ~0 for prose-vs-option, but BERTScore happily scores
  on-topic prose against a short option at ≥70. **Any "the filter will catch it" claim has to be
  checked against *both* halves of the OR** — the same slip was made twice.
- Empty teacher answers auto-fail. Mix policies: `replace` (default — teacher where it passed,
  else gold, keeping the row set identical to the gold-SFT run so the comparison stays
  one-variable), `both`, `teacher`.

### 5.3 The metric-vs-human tradeoff

The teacher's style is verbose markdown (headers, bold, emoji); automatic metrics score against
golds, but **human eval** — which decides the primary submission — plausibly prefers the
teacher's fuller answers. So the threshold is a metric-vs-human tradeoff, and we considered a
deliberately looser filter for OEG.

**Caveat:** "golds are short and dry" is true of aya and tydiqa, **not** of OEG. The OEG golds
are GPT-4.1 outputs (median 175 words, already with markdown). So on the one source that reaches
the test set as `qa-oeg`, the gold is already a strong model's verbose answer, and the human-eval
argument for preferring teacher style over gold style mostly does not apply there.

### 5.4 Why distillation didn't pay off (the value cross)

Distillation's premise is "the teacher's answer is a better training target than the gold". Line
that up against where each source lands at test time and it crosses badly:

| source | what the gold is | teacher upside | reaches the test set? |
|---|---|---|---|
| aya | human, 24 words | **high** — teacher writes fuller answers | **no** — wrong length regime |
| OEG (`qa-oeg`) | **GPT-4.1, 175 words, markdown** | **unclear** — gold is already a strong model; the filter passes ~94%, so it barely filters | **yes** |
| tydiqa (`qa-context`) | 2-word extraction | **questionable** — swaps extraction targets for prose | yes |
| belebele | `2: <option>` | **negative** — swaps the MC gold for prose | **no** — no MC at test |

**Distillation has the most headroom exactly where it doesn't transfer (aya), and the least
where it matters (OEG).** And the gold adapter already banks the OEG gain — it was trained on the
same GPT-4.1 golds.

**Measured verdict (jobs 3865036 + 3866054): it lost.** Against plain gold SFT, matched
`--no-lang-hint` on both sides: `qa-context` (MCIF) **−11.92** COMBINED, `qa-oeg` long-form (OEG)
**−12.37**, `qa-oeg` short-answer (aya) **+1.28**. The one column it wins (aya) doesn't reach the
test set; it is *last of all systems* on OEG, which does. The confound — distillation also dropped
the lang-hint — was closed by 3866054, so the ~12-point loss is the **teacher data**, not the
hint. **No routing change.** Human eval is the one thread the automatic metrics can't close, but
it isn't enough to deploy an adapter that loses ~12 COMBINED on the faithful proxy.

### 5.5 Per-source thresholds are required, not optional

A single global threshold is indefensible — the measured pass rates make it do something
different, and wrong, on each source:

- **belebele: always keep gold** (`--gold-only belebele`). Every passing row corrupts an MC target
  the test set never asks for. It also confounds the headline experiment — a distilled adapter
  trained on corrupted belebele targets craters on belebele in dev and looks like "distillation
  failed" when the only failure is a source we already treat as noise.
- **OEG: 94% pass = the filter is a near no-op.** Whatever threshold we pick here barely matters;
  the real question is §5.4's.
- **tydiqa: 31.5% pass, and it reaches the test set.** Swapping a 2-word extraction gold for prose
  on a third of rows is a real intervention on a source that counts.

**Corollary:** the ~4,577 belebele teacher rows are compute spent on answers we should never use.
Not worth killing a running generation job over, but exclude belebele at *filter* time (free)
rather than regenerating.

## 6. Output quality

### 6.1 Runaway generation (the stop fix)

After a correct short answer, the fine-tuned adapters sometimes kept going, inventing a fake chat
exchange (`\nuser\n…\nassistant\n<think>`) that got scored as part of the answer and dragged the
numbers down. Base models were never affected (0% incidence). What we established:

- **Label masking is not the bug.** Reproduced locally with the Qwen3.5-9B tokenizer: the label
  span is `answer + <|im_end|> + \n`, so EOS *is* trained. The adapter simply under-samples it
  after *short* golds at T=0.7/top-p 0.8/top-k 20; long and templated golds don't trigger it.
- **This retro-explains the v1 "tydiqa collapse"** (38.94→19.53): 78% of that adapter's tydiqa
  predictions ran on — it was truncation, not lost ability.
- **Measure refusals only on truncated predictions.** The correct refusal string is usually there
  with junk after it, so raw-CSV refusal metrics undercount by ~25× (3.9% vs the true 92.7%).
- **Fix:** `prompt_template.RUNAWAY_STOP_STRINGS` (halts at generate time, saves budget) plus
  `truncate_runaway()` on every decode. Smoke 3869113 (the slice that ran away 76% of the time):
  **0/60 contaminated**. Every adapter from job 3869088 on is clean. Any custom inference path
  must attach the stop strings too.

### 6.2 `<br>` markup and degenerate repetition

Two flaws present in *all* adapters (so neither is caused by C or D):

- **Literal `<br>` markup** in ~27–31% of qa-oeg predictions, from the web-scraped substrate
  (aya/oeg). One in three predictions carries HTML. **The cheapest unclaimed point on the whole
  roadmap** — a one-line strip in the submission script.
- **Degenerate repetition** (a sentence repeated ≥4×) in ~2% of rows — general OOD-language
  degeneration, not runaway (no fake turns, so the stop fix misses it). Would need
  `repetition_penalty` or n-gram blocking; not worth the decoding-param risk at 2% unless done
  alongside the F language gate.

### 6.3 The aya "style shift" is mostly a chrF artifact

The adapter answers aya tersely (11 words, matching the gold's register) while the base pads with
markdown (105 words, 89% markdown-formatted). That padding is what inflates the base's chrF
(character recall), while BERTScore and ROUGE-L correctly prefer the adapter. No fix needed.

## 7. Evaluation caveats

Scoring rule for all of these: **COMBINED = mean(chrF, BERTScore, ROUGE-L)** per task×source
column, never pooled across tasks; qa-oeg aggregate = 0.87·OEG + 0.13·aya (test shape). The full
proxy argument lives in the experiment logs — these are the traps specific to reading the numbers.

### 7.1 The per-language dev breakdown

`evaluate.py` prints a per-language chrF block, but it is keyed on the **question** language
(`benchmark.py` renames `question_lang` → `lang_code`), not the context language — on
cross-lingual sources those differ, so say so. Three caveats travel with the table:

- **The language inventory changed with the v2 split** (24 languages, was 27; `tha`/`swh`/`tel`
  gone, several `n` moved a lot). That's a rebuild artifact — old-vs-new is not like-for-like.
- **dev↔train leakage made the v1 dev optimistic** (DATA_AUDIT §2). Read per-language numbers as
  a **relative** ranking, never an absolute capability estimate.
- **dev is structurally blind to our two contributions** — `dev_v2` has 0 bho rows and 0 budget
  rows (both enumerated). C and D can only be measured on test (§7.3).

One trap when copying numbers: `evaluate.py` labels the single `overall` line "LEGACY — 71%
noise, do not compare systems on this", because dev's source weighting is inverted against the
test mix. Never rank systems on it.

### 7.2 Paired bootstrap — the edges are marginal

The dev qa-oeg aggregate is 87% a 90-row column (OEG), and the moving part is OEG's ROUGE-L.
Across 10,000 paired resamples (`bootstrap_compare.py`, reference = C+D-small), on the faithful
proxy for each sub-task:

- **The only gaps that clear noise are C-only vs C+D-small, and they point opposite ways:** C-only
  wins qa-oeg long-form (+1.79, p=0.041), C+D-small wins qa-context (−2.44, p=0.048). Both CIs
  graze zero.
- Every plain/C+D comparison is noise — earlier "+0.88 over C+D" / "+1.01 over plain" rankings did
  not survive.

This is why one model (C+D-small) serves everything and C-only is only a possible qa-oeg-only
variant.

### 7.3 D on qa-context (and why aggregates hide it)

The 360 bho qa-context rows are too short for `bho_lid` (median 3 words, one-sentence answers), so
script/sentence/refusal checks call all adapters identical. What actually separates them is
**contrastive function words** (which of `आ`/`खातिर` vs `और`/`के लिए` appears — one word is enough):

| | C+D | **C+D-small** | plain |
|---|---|---|---|
| bho-leaning / hin-leaning | **163 / 1** | 147 / 16 | 21 / 93 |
| % bho of the decidable | **99%** | **90%** | 18% |

This is the one check where C+D-small reads weaker than C+D (90% vs 99%) — the only place more bho
data bought more bho output. Caveats: only 39% of rows are long enough to judge, and this shows
output that *looks* bho, not that it's *correct* (the test set has no gold).

## 8. Infrastructure & reproducibility

- **atuin ($WORK) group file quota exceeded** (500K soft / 600K hard, shared across group
  `b279bb`; grace expired). All atuin *writes* fail. Working layout is a **hybrid**: active clone
  in `$HOME`, reads (venv, hf_cache, adapters) still from atuin, prepared-datasets cache copied to
  `$HOME`. Every sbatch script probes atuin writability at runtime and falls back — two jobs
  (3857583, 3859591) died in <40s from an unwritable lock path before this was baked in.
- **`$HOME` and `$HPCVAULT` double-count usage** against quota (every file's blocks are ~2× its
  apparent size; mirrored storage). Budget at 2× nominal — the distillation teachers sit on
  `$HPCVAULT` (1TB), never `$HOME`.
- **The login node is for submitting, not computing.** A BERTScore pass run via `nohup` on the
  login node pegged ~67 cores and was killed by the admin. Everything — including "quick" scoring
  — goes through sbatch.
- **Three venvs:** `mist-venv` (main transformers stack), `mist-venv2` (+gptqmodel, only for the
  transformers-GPTQ dead end), `vllm-venv` (vLLM, pins its own torch).
- **Adapter-path sharp edge (fixed).** A v2 adapter once scored on its own training data for 5h23
  because `--data` defaulted to the leaky split. Now `benchmark.py` requires `--data`, resolves
  `--lora` to an absolute path, rejects a dir with no `adapter_config.json` before loading
  anything, and logs `model: … adapter: <abs>`.
- **`run_test.py` throughput is strongly task-dependent** (one a40, 9B + LoRA, max-new-tokens 512):
  - `qa-oeg` **≈20.5 s/row** — long prompts, answers run to the full budget.
  - `qa-context` **≈5.9 s/row** — the prompt asks for one sentence, so generation stops early.
  - The full qa set is **~27.5h** (8,640×5.9 + 2,359×20.5), *not* the ~67h an earlier note got by
    extrapolating the slow task onto all rows. Still over the 24h wall, so shard — but **n=2
    (13.8h/shard) is enough**, where n=4–6 was planned. (qa-context rate measured on bho only;
    re-check one more language before the final schedule.)
- Measured runtimes that set the sbatch budgets: 9B 0-shot ~6h, 9B 3-shot ~3.7h, 9B LoRA SFT
  ~6.5h, 9B LoRA eval ~5–7h, scoring 2,978 rows ~1m, 35B teacher ~14 s/row, 122B-vLLM 4,126 rows
  in 17m.
