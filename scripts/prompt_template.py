"""Shared prompt template for the WMT26 MIST qa task.

Kept in one place so that *training* (SFT) and *inference* (benchmark.py) build the model
input identically -- a mismatch between the two is the usual reason a fine-tuned model stops
following an instruction at test time.

The only task-specific signal we inject is the **target output language**, as a natural-language
system instruction derived from `lang_code`. For the qa split, `lang_code` is the language the
answer is expected in (verified: e.g. aya rows have an English question but a `hin_Deva` code and
a Hindi gold). We deliberately:
  - use the language *name* ("Hindi"), not the raw code ("hin_Deva"), so the template stays
    compatible with a test format that may only phrase the requirement in natural language;
  - keep the instruction light ("Respond in X.") rather than "ONLY in X / never other scripts",
    because many correct answers legitimately contain Latin proper nouns or numbers
    (e.g. '演讲者是 James Finch。', 'Infinity Ward', '15,000km²').
"""

import re
import sys

# --- Runaway-generation guard (2026-07-18, EXPERIMENTS_NEW.md) ---------------------------
# LoRA adapters fine-tuned on short gold answers under-sample <|im_end|> at T=0.7/top-p 0.8
# and continue past the answer into hallucinated chat turns, rendered as plain text once
# special tokens are stripped ("answer\nuser\n<other question>\nassistant\n<think>...").
# 66% of the v2 gold adapter's dev predictions were contaminated (eval 3867140); the OLD
# adapter's famous "tydiqa collapse" was the same artifact (78% of 3857589's tydiqa rows).
# Two-layer fix, shared by benchmark.py and run_test.py:
#   1. pass RUNAWAY_STOP_STRINGS to model.generate(stop_strings=..., tokenizer=tok) so
#      generation halts at the first fake turn instead of burning the token budget;
#   2. ALWAYS pass the decoded prediction through truncate_runaway() -- catches whatever
#      slips through and cleans the stop-string remnant itself. Base models are unaffected
#      (0% incidence on jobs 3867141/3867142).
RUNAWAY_STOP_STRINGS = ["<|im_start|>", "\nuser\n", "\nassistant\n", "<think>"]
_RUNAWAY = re.compile(r"\nuser\n|\nassistant\n|<think>")


def truncate_runaway(pred: str) -> str:
    """Cut a decoded prediction at the first hallucinated-turn marker (identity for clean
    text). Verified to reproduce the 3869088 re-score cleanup exactly."""
    m = _RUNAWAY.search(pred)
    return pred[:m.start()].rstrip() if m else pred


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

# The official test set (pinzhenchen/wmt26-mist-test) identifies languages by bare
# 3-letter `question_lang` codes ("zho", "ita", ...) -- no script suffix -- and adds one
# surprise language with zero training rows: Bhojpuri ("bho", Devanagari, close to Hindi).
# Derived from LANG_NAMES so the two can never disagree on a name.
TEST_LANG_NAMES = {code.split("_")[0]: name for code, name in LANG_NAMES.items()}
TEST_LANG_NAMES["bho"] = "Bhojpuri"

# Which sample-data `source` values stand in for which test `task` (TEST_SET_ANALYSIS 5b).
# A test row's only task signal is `task`, so this is what few-shot selection at test time
# (run_test.py --shots) matches on. `facebook/belebele` is deliberately absent from both
# lists: it is multiple-choice, the test set contains none (TEST_SET_ANALYSIS 5), and its
# "answer with a letter" format is exactly the wrong thing to demonstrate.
# NOTE: the same partition is spelled out in augment_constraints.py (OPEN_ENDED_SOURCES /
# CONTEXT_SOURCES) and evaluate.py (TASK_PROXY, split into two qa-oeg columns for reporting).
# Those two predate this one; if the mapping ever changes, all three must change together.
TEST_TASK_SOURCES = {
    "qa-context": ["copenlu/answerable_tydiqa", "FBK-MT/MCIF"],
    "qa-oeg": ["wmt25-mist-oeg-gpt-4.1", "CohereLabs/aya_dataset"],
}


def system_turn(lang_name: str) -> dict:
    """The target-language system turn, shared verbatim by training (SFT), dev benchmarking
    and test inference -- the exact phrasing is part of what a fine-tuned model learns, so
    there must be only one copy of it."""
    return {"role": "system",
            "content": f"You are a helpful assistant. Respond in {lang_name}."}


_warned_codes: set[str] = set()


def _warn_unknown(lang_code: str) -> None:
    """Warn once per unknown lang_code (avoids per-row spam over a whole dev/test set)."""
    if lang_code not in _warned_codes:
        _warned_codes.add(lang_code)
        print(f"WARNING: lang_code {lang_code!r} not in LANG_NAMES; using the raw code as the "
              f"language name in the prompt. Add it to LANG_NAMES for a proper instruction.",
              file=sys.stderr, flush=True)


def build_messages(input_text: str, lang_code: str, lang_hint: bool = True,
                   examples: list[tuple[str, str]] | None = None) -> list[dict]:
    """Return chat `messages` for one qa example.

    If `lang_hint`, prepend a system turn naming the target language. An unknown lang_code
    (the two July-1 surprise languages will be unknown until we add them to LANG_NAMES) does
    NOT crash -- it falls back to the raw code so a run still completes -- but it emits a
    one-time warning, because "Respond in abc_Xyzz." is a much weaker instruction than a real
    language name. If you see that warning at test time, add the code to LANG_NAMES.

    `examples` are few-shot demonstrations as (input, gold output) pairs. They are inserted
    as completed user/assistant chat turns before the real question -- the model sees prior
    exchanges it can imitate -- rather than pasted into one big prompt, because chat models
    are trained on the turn structure and imitate it more reliably. Pass None (or []) for
    zero-shot.
    """
    messages = []
    if lang_hint:
        # Accept both the sample data's script-suffixed codes ("hin_Deva") and the test
        # set / v2 dataset's bare codes ("hin") -- data/{train,dev}_v2.jsonl use the latter.
        if lang_code in LANG_NAMES:
            lang = LANG_NAMES[lang_code]
        elif lang_code in TEST_LANG_NAMES:
            lang = TEST_LANG_NAMES[lang_code]
        else:
            lang = lang_code
            _warn_unknown(lang_code)
        messages.append(system_turn(lang))
    for ex_input, ex_output in examples or ():
        messages.append({"role": "user", "content": ex_input})
        messages.append({"role": "assistant", "content": ex_output})
    messages.append({"role": "user", "content": input_text})
    return messages
