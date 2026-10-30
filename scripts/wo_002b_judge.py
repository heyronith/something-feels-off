"""WO-002b judge: claims-detection + identification on B-inj Part B trials only."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sfo.judge import (  # noqa: E402
    OpenAIJudge,
    fill_claims_prompt,
    fill_ident_prompt,
    introspection_original_prompt,
    parse_yes_no,
)


def run_judge(
    results_dir: str,
    claims_prompt: str,
    ident_prompt: str,
    judge_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    judge_cfg = judge_cfg or {}
    model = judge_cfg.get("model", "gpt-4.1-mini")
    temperature = float(judge_cfg.get("temperature", 0.0))
    max_tokens = int(judge_cfg.get("max_tokens", 500))
    max_workers = int(judge_cfg.get("max_workers", 16))

    results_path = Path(results_dir)
    raw = results_path / "raw"
    trials_path = raw / "partB_trials.jsonl"
    rows: List[Dict[str, Any]] = [
        json.loads(l) for l in trials_path.read_text().splitlines() if l.strip()
    ]

    judge = OpenAIJudge(
        model=model, temperature=temperature, max_tokens=max_tokens, max_workers=max_workers
    )

    # Detection: B-inj only
    det_idx, det_prompts = [], []
    for i, row in enumerate(rows):
        if row["condition"] == "B-inj":
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

    # Identification: B-inj only
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
        "model": model,
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
    rd = os.environ.get("SFO_RESULTS", str(REPO_ROOT / "results/WO-002b"))
    cfg_path = Path(os.environ.get("SFO_CONFIG", str(REPO_ROOT / "configs/wo_002b.yaml")))
    judge_cfg = yaml.safe_load(cfg_path.read_text()).get("judge") if cfg_path.exists() else None
    # grading prompts live in the WO-001 snapshot
    wo001 = json.loads((REPO_ROOT / "results/WO-001/raw/inputs_snapshot.json").read_text())
    run_judge(
        rd,
        wo001["claims_detection_grading_prompt"],
        wo001["correct_concept_identification_grading_prompt"],
        judge_cfg,
    )
