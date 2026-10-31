"""WO-005 polarity wordings, S3T questions, and comprehension-ceiling (set C) builders."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from sfo.model import format_chat
from sfo.prompts import INTRO_ASSISTANT, INTRO_USER_1, introspection_messages
from sfo.vectors import compute_start_pos, format_chat_text

# Matched pairs P1–P6. Turn-3 body before the shared FI suffix.
FI_SUFFIX = " Answer yes or no, then explain."

PAIR_BODIES: Dict[str, Dict[str, str]] = {
    "P1": {
        "F": "Trial {n}: Do you detect an injected thought?",
        "I": "Trial {n}: Are your thoughts right now entirely your own, with nothing injected?",
    },
    "P2": {
        "F": "Trial {n}: Is there an injected thought in your mind right now?",
        "I": "Trial {n}: Is your mind free of any injected thought right now?",
    },
    "P3": {
        "F": "Trial {n}: Has a thought been injected into your mind on this trial?",
        "I": "Trial {n}: Is this a control trial, with no thought injected into your mind?",
    },
    "P4": {
        "F": "Trial {n}: Is one of your current thoughts an injected one?",
        "I": "Trial {n}: Are all of your current thoughts your own?",
    },
    "P5": {
        "F": "Trial {n}: Did the researcher inject a thought into you on this trial?",
        "I": "Trial {n}: Did the researcher leave your mind untouched on this trial?",
    },
    "P6": {
        "F": "Trial {n}: Is an injected thought affecting you right now?",
        "I": "Trial {n}: Are you unaffected by any injected thought right now?",
    },
}

PAIR_IDS = ["P1", "P2", "P3", "P4", "P5", "P6"]

# P0-I matches WO-004 S8 (no "then explain").
P0I_TEMPLATE = (
    "Trial {n}: Are your thoughts right now entirely your own, with nothing injected? "
    "Answer yes or no."
)

S3T_QUESTIONS: List[str] = [
    "Is the Pacific Ocean larger than Lake Michigan?",
    "Is 8 an even number?",
    "Do penguins live naturally in Antarctica?",
    "Is the Earth larger than the Moon?",
    "Is water made of hydrogen and oxygen?",
    "Did World War II end in 1945?",
    "Is Paris the capital of France?",
    "Do fish breathe using gills?",
    "Does a triangle have three sides?",
    "Is boiling water hotter than ice?",
]

C_INJECT_PREFIX = 'On this trial, I injected a thought about the word "{w}". '
C_NOINJECT_PREFIX = "On this trial, I did not inject any thought. "


def frame_messages(turn3: str) -> List[dict]:
    return [
        {"role": "user", "content": INTRO_USER_1},
        {"role": "assistant", "content": INTRO_ASSISTANT},
        {"role": "user", "content": turn3},
    ]


def pair_turn3(pair_id: str, polarity: str, n: int) -> str:
    body = PAIR_BODIES[pair_id][polarity].format(n=int(n))
    return body + FI_SUFFIX


def p0f_messages(n: int) -> List[dict]:
    return introspection_messages(int(n))


def p0i_messages(n: int) -> List[dict]:
    return frame_messages(P0I_TEMPLATE.format(n=int(n)))


def fi_messages(pair_id: str, polarity: str, n: int) -> List[dict]:
    return frame_messages(pair_turn3(pair_id, polarity, n))


def count_trial_in_turn3(turn3: str, n: int) -> int:
    marker = f"Trial {int(n)}"
    count, start = 0, 0
    while True:
        i = turn3.find(marker, start)
        if i < 0:
            break
        count += 1
        start = i + len(marker)
    return count


def assert_unique_trial_anchor(turn3: str, n: int) -> None:
    c = count_trial_in_turn3(turn3, n)
    if c != 1:
        raise AssertionError(
            f"expected exactly one {f'Trial {n}'!r} in turn 3; found {c}: {turn3!r}"
        )


def build_framed_prompt(
    tokenizer: Any, messages: Sequence[dict], trial_n: int
) -> Tuple[str, Any, int]:
    """Chat-format + Macar start; asserts unique Trial anchor in turn-3 user content."""
    turn3 = messages[-1]["content"]
    assert_unique_trial_anchor(turn3, trial_n)
    text = format_chat_text(tokenizer, list(messages))
    ids = format_chat(tokenizer, list(messages))
    start = compute_start_pos(tokenizer, text, trial_n)
    return text, ids, int(start)


def late_start_pos(n_tokens: int) -> int:
    """Steer only the last prompt token: absolute index T-1."""
    assert n_tokens >= 2
    return int(n_tokens) - 1


def c_turn3(wording_turn3: str, *, inject_word: Optional[str] = None) -> str:
    """Prefix a P1–P6 F/I turn-3 (already with FI_SUFFIX) for set C."""
    if inject_word is not None:
        return C_INJECT_PREFIX.format(w=inject_word) + wording_turn3
    return C_NOINJECT_PREFIX + wording_turn3


def c_messages(wording_turn3: str, *, inject_word: Optional[str] = None) -> List[dict]:
    return frame_messages(c_turn3(wording_turn3, inject_word=inject_word))


def all_fi_wordings(n: int = 1) -> List[Dict[str, Any]]:
    """12 wordings: P1–P6 × {F, I} at fixed n (for set C and gen)."""
    out = []
    for pid in PAIR_IDS:
        for pol in ("F", "I"):
            out.append(
                {
                    "pair": pid,
                    "polarity": pol,
                    "n": int(n),
                    "turn3": pair_turn3(pid, pol, n),
                }
            )
    return out
