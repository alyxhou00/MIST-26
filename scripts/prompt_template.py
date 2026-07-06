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

import sys

# July-1 test set adds two "surprise" languages not in the sample data. Add their
# {code: name} here once revealed; until then an unknown code triggers _warn_unknown().
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
        if lang_code in LANG_NAMES:
            lang = LANG_NAMES[lang_code]
        else:
            lang = lang_code
            _warn_unknown(lang_code)
        messages.append({"role": "system",
                         "content": f"You are a helpful assistant. Respond in {lang}."})
    for ex_input, ex_output in examples or ():
        messages.append({"role": "user", "content": ex_input})
        messages.append({"role": "assistant", "content": ex_output})
    messages.append({"role": "user", "content": input_text})
    return messages
