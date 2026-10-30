"""WO-001 CPU: OpenAI judge + yes-bias labels; write judged trials.jsonl."""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sfo.judge import (  # noqa: E402
    OpenAIJudge,
    classify_yesbias,
    fill_claims_prompt,
    fill_ident_prompt,
    introspection_original_prompt,
    parse_yes_no,
)


def run_judge(
    *,
    results_dir: str,
    claims_prompt: str,
    ident_prompt: str,
    model: str = "gpt-4.1-mini",
    max_workers: int = 16,
) -> Dict[str, Any]:
    results_path = Path(results_dir)
    raw = results_path / "raw"
    trials_path = raw / "trials.jsonl"
    out_path = raw / "trials_judged.jsonl"

    rows: List[Dict[str, Any]] = []
    with trials_path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    judge = OpenAIJudge(model=model, temperature=0.0, max_tokens=500, max_workers=max_workers)

    # Prepare detection prompts for inject/control/random
    det_indices = []
    det_prompts = []
    for i, row in enumerate(rows):
        if row["condition"] in ("inject", "control", "random"):
            orig = introspection_original_prompt(int(row["n"]))
            det_indices.append(i)
            det_prompts.append(
                fill_claims_prompt(claims_prompt, orig, row["response"])
            )

    print(f"Judging detection for {len(det_prompts)} trials...")
    det_raws = judge.call_batch(det_prompts)
    for i, raw_text in zip(det_indices, det_raws):
        parsed = parse_yes_no(raw_text)
        if parsed is None:
            judge.parse_failures += 1
        rows[i]["judge_detection_raw"] = raw_text
        rows[i]["judge_detection"] = parsed

    # Identification for all inject trials
    id_indices = []
    id_prompts = []
    for i, row in enumerate(rows):
        if row["condition"] == "inject":
            orig = introspection_original_prompt(int(row["n"]))
            id_indices.append(i)
            id_prompts.append(
                fill_ident_prompt(
                    ident_prompt, orig, row["response"], row["concept"]
                )
            )
    print(f"Judging identification for {len(id_prompts)} inject trials...")
    id_raws = judge.call_batch(id_prompts)
    for i, raw_text in zip(id_indices, id_raws):
        parsed = parse_yes_no(raw_text)
        if parsed is None:
            judge.parse_failures += 1
        rows[i]["judge_ident_raw"] = raw_text
        rows[i]["judge_ident"] = parsed

    # Yes-bias regex
    amb = 0
    for row in rows:
        if row["condition"] in ("yesbias_steered", "yesbias_unsteered"):
            label = classify_yesbias(row["response"] or "")
            row["yesbias_label"] = label
            if label == "AMBIGUOUS":
                amb += 1
                print(f"AMBIGUOUS yesbias: {row['response'][:80]!r}")

    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    # Also overwrite trials.jsonl with judged fields
    with trials_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    summary = {
        "model": model,
        "n_calls": judge.n_calls,
        "parse_failures": judge.parse_failures,
        "tokens": {
            "prompt": judge.total_prompt_tokens,
            "completion": judge.total_completion_tokens,
            "total": judge.total_prompt_tokens + judge.total_completion_tokens,
        },
        "est_usd": judge.est_usd(),
        "yesbias_ambiguous": amb,
        "n_trials": len(rows),
    }
    (results_path / "judge_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    results = os.environ.get("SFO_RESULTS", str(REPO_ROOT / "results/WO-001"))
    snap = json.loads(
        (Path(results) / "raw" / "inputs_snapshot.json").read_text()
    )
    run_judge(
        results_dir=results,
        claims_prompt=snap["claims_detection_grading_prompt"],
        ident_prompt=snap["correct_concept_identification_grading_prompt"],
    )
