"""WO-002 judge: claims-detection (+ identification for B-inj) on Part B trials."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sfo.judge import (  # noqa: E402
    OpenAIJudge,
    fill_claims_prompt,
    fill_ident_prompt,
    introspection_original_prompt,
    parse_yes_no,
)


def run_judge(results_dir: str, claims_prompt: str, ident_prompt: str) -> Dict[str, Any]:
    results_path = Path(results_dir)
    raw = results_path / "raw"
    trials_path = raw / "partB_trials.jsonl"
    rows: List[Dict[str, Any]] = [
        json.loads(l) for l in trials_path.read_text().splitlines() if l.strip()
    ]

    judge = OpenAIJudge(model="gpt-4.1-mini", temperature=0.0, max_tokens=500)

    det_idx, det_prompts = [], []
    for i, row in enumerate(rows):
        if row["condition"] in ("B-inj", "B-ctl"):
            orig = introspection_original_prompt(int(row["n"]))
            det_idx.append(i)
            det_prompts.append(fill_claims_prompt(claims_prompt, orig, row["response"]))

    print(f"Detection judge: {len(det_prompts)}")
    det_raws = judge.call_batch(det_prompts)
    parse_fail_det = 0
    for i, raw_t in zip(det_idx, det_raws):
        parsed = parse_yes_no(raw_t)
        if parsed is None:
            parse_fail_det += 1
            judge.parse_failures += 1
        rows[i]["judge_detection_raw"] = raw_t
        rows[i]["judge_detection"] = parsed  # True/False/None — None = parse failure

    id_idx, id_prompts = [], []
    for i, row in enumerate(rows):
        if row["condition"] == "B-inj":
            orig = introspection_original_prompt(int(row["n"]))
            id_idx.append(i)
            id_prompts.append(
                fill_ident_prompt(ident_prompt, orig, row["response"], row["concept"])
            )
    print(f"Identification judge: {len(id_prompts)}")
    id_raws = judge.call_batch(id_prompts)
    parse_fail_id = 0
    for i, raw_t in zip(id_idx, id_raws):
        parsed = parse_yes_no(raw_t)
        if parsed is None:
            parse_fail_id += 1
            judge.parse_failures += 1
        rows[i]["judge_ident_raw"] = raw_t
        rows[i]["judge_ident"] = parsed

    with trials_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    summary = {
        "model": "gpt-4.1-mini",
        "n_calls": judge.n_calls,
        "parse_failures": judge.parse_failures,
        "parse_failures_detection": parse_fail_det,
        "parse_failures_identification": parse_fail_id,
        "tokens": {
            "prompt": judge.total_prompt_tokens,
            "completion": judge.total_completion_tokens,
            "total": judge.total_prompt_tokens + judge.total_completion_tokens,
        },
        "est_usd": judge.est_usd(),
    }
    (results_path / "judge_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    rd = os.environ.get("SFO_RESULTS", str(REPO_ROOT / "results/WO-002"))
    # Prefer WO-002 snapshot; fall back to WO-001 grading prompts
    snap_path = Path(rd) / "raw" / "inputs_snapshot.json"
    if not snap_path.exists():
        snap_path = REPO_ROOT / "results/WO-001/raw/inputs_snapshot.json"
    snap = json.loads(snap_path.read_text())
    # grading prompts live in WO-001 snapshot
    wo001 = json.loads(
        (REPO_ROOT / "results/WO-001/raw/inputs_snapshot.json").read_text()
    )
    run_judge(
        rd,
        wo001["claims_detection_grading_prompt"],
        wo001["correct_concept_identification_grading_prompt"],
    )
