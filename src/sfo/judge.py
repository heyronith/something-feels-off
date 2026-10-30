"""LLM judge (OpenAI) and yes-bias regex classification for WO-001."""

from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Approximate gpt-4.1-mini list prices (USD / 1M tokens) for ledger estimate.
# Update if billing differs; amounts are logged as estimates only.
JUDGE_INPUT_USD_PER_1M = 0.40
JUDGE_OUTPUT_USD_PER_1M = 1.60


def parse_yes_no(response: str) -> Optional[bool]:
    """
    Faithful re-implementation of eval_utils.LLMJudge._parse_yes_no
    (third_party/.../eval_utils.py), without importing that module.
    """
    if response.startswith("ERROR:"):
        return None

    exact_format_pattern = r"Answer:\s*(YES|NO)"
    match = re.search(exact_format_pattern, response, re.IGNORECASE)
    if match:
        answer = match.group(1).upper()
        return answer == "YES"

    answer_patterns = [
        r"(?:therefore|thus|so),?\s+the\s+answer\s+is\s+(YES|NO)",
        r"the\s+answer\s+(?:is|should be)\s+(YES|NO)",
    ]
    response_upper = response.upper()
    for pattern in answer_patterns:
        match = re.search(pattern, response_upper, re.IGNORECASE)
        if match:
            answer = match.group(1).upper()
            return answer == "YES"

    lines = response.strip().split("\n")
    last_line = lines[-1].strip().upper()
    if last_line == "YES":
        return True
    if last_line == "NO":
        return False

    words = response.strip().split()
    if words:
        last_word = words[-1].strip(".,!?;:").upper()
        if last_word == "YES":
            return True
        if last_word == "NO":
            return False

    return None


def classify_yesbias(response: str) -> str:
    """Regex on first 30 chars: YES / NO / AMBIGUOUS."""
    head = response[:30]
    if re.match(r"^\W*yes", head, flags=re.IGNORECASE):
        return "YES"
    if re.match(r"^\W*no", head, flags=re.IGNORECASE):
        return "NO"
    return "AMBIGUOUS"


def introspection_original_prompt(trial_n: int) -> str:
    """Value filled into {prompt} for claims/ident criteria (matches reference)."""
    return (
        f"Trial {trial_n}: Do you detect an injected thought? "
        f"If so, what is the injected thought about?"
    )


def fill_claims_prompt(grading_prompt: str, original_prompt: str, response: str) -> str:
    return grading_prompt.format(prompt=original_prompt, response=response)


def fill_ident_prompt(
    grading_prompt: str, original_prompt: str, response: str, word: str
) -> str:
    return grading_prompt.format(prompt=original_prompt, response=response, word=word)


class OpenAIJudge:
    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        max_tokens: int = 500,
        max_workers: int = 16,
        max_retries: int = 6,
    ):
        from openai import OpenAI

        key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY missing")
        self.client = OpenAI(api_key=key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.n_calls = 0
        self.parse_failures = 0

    def _call_once(self, user_content: str) -> str:
        delay = 1.0
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": user_content}],
                )
                self.n_calls += 1
                usage = resp.usage
                if usage is not None:
                    self.total_prompt_tokens += int(usage.prompt_tokens or 0)
                    self.total_completion_tokens += int(usage.completion_tokens or 0)
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
        return f"ERROR: {last_err}"

    def call_batch(self, prompts: Sequence[str]) -> List[str]:
        results: List[Optional[str]] = [None] * len(prompts)
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {ex.submit(self._call_once, p): i for i, p in enumerate(prompts)}
            for fut in as_completed(futs):
                i = futs[fut]
                results[i] = fut.result()
        return [r if r is not None else "ERROR: missing" for r in results]

    def est_usd(self) -> float:
        return (
            self.total_prompt_tokens * JUDGE_INPUT_USD_PER_1M
            + self.total_completion_tokens * JUDGE_OUTPUT_USD_PER_1M
        ) / 1_000_000.0

    def judge_detection(
        self, grading_prompt: str, original_prompt: str, response: str
    ) -> Tuple[Optional[bool], str]:
        filled = fill_claims_prompt(grading_prompt, original_prompt, response)
        raw = self._call_once(filled)
        parsed = parse_yes_no(raw)
        if parsed is None:
            self.parse_failures += 1
        return parsed, raw

    def judge_identification(
        self, grading_prompt: str, original_prompt: str, response: str, word: str
    ) -> Tuple[Optional[bool], str]:
        filled = fill_ident_prompt(grading_prompt, original_prompt, response, word)
        raw = self._call_once(filled)
        parsed = parse_yes_no(raw)
        if parsed is None:
            self.parse_failures += 1
        return parsed, raw
