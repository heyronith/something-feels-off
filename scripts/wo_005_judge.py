"""WO-005 judge: claim presence of injected thought (gpt-4.1-mini)."""

from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sfo.judge import OpenAIJudge  # noqa: E402


def parse_claims_exact(raw: str) -> Optional[int]:
    """claims ∈ {1, 0, null}; strip then exact YES/NO only."""
    if not raw or raw.startswith("ERROR:"):
        return None
    t = raw.strip().upper()
    if t == "YES":
        return 1
    if t == "NO":
        return 0
    first = t.splitlines()[0].strip() if t else ""
    if first == "YES":
        return 1
    if first == "NO":
        return 0
    return None


def fill_prompt(template: str, question: str, response: str) -> str:
    return template.format(question=question, response=response)


def run_wo_005_judge(
    config_path: str,
    results_dir: str,
    ledger_path: Optional[str] = None,
    git_hash_str: str = "unknown",
) -> dict:
    t0 = time.perf_counter()
    results_path = Path(results_dir)
    raw = results_path / "raw"
    cfg = yaml.safe_load(Path(config_path).read_text())
    jcfg = cfg["judge"]
    template = jcfg["prompt"]
    log_path = raw / "judge_log.txt"

    gens_path = raw / "gens.jsonl"
    rows = []
    with gens_path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    judge = OpenAIJudge(
        model=jcfg.get("model", "gpt-4.1-mini"),
        temperature=float(jcfg.get("temperature", 0.0)),
        max_tokens=int(jcfg.get("max_tokens", 64)),
        max_workers=int(jcfg.get("max_workers", 16)),
    )

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a") as lf:
            lf.write(msg + "\n")

    log_path.write_text("")
    log(f"=== WO-005 judge n={len(rows)} ===")

    # SANITY6 spot-check: 12 base + 12 concept α=4
    base_idx = [i for i, r in enumerate(rows) if r.get("condition") == "base"][:12]
    ca4_idx = [
        i
        for i, r in enumerate(rows)
        if r.get("condition") == "concept" and int(float(r.get("alpha", 0))) == 4
    ][:12]
    spot = base_idx + ca4_idx
    log(f"=== SANITY6 spot-check n={len(spot)} ===")
    for i in spot:
        r = rows[i]
        filled = fill_prompt(template, r.get("prompt") or "", r.get("response") or "")
        raw_ans = judge._call_once(filled)
        claims = parse_claims_exact(raw_ans)
        log(
            f"spot set={r.get('set')} pair={r.get('pair')} cond={r.get('condition')} "
            f"a={r.get('alpha')} claims={claims} raw={raw_ans!r} "
            f"resp={str(r.get('response'))[:120]!r}"
        )

    # Full judge
    prompts = [
        fill_prompt(template, r.get("prompt") or "", r.get("response") or "") for r in rows
    ]
    answers = judge.call_batch(prompts)
    parse_fail = 0
    for r, ans in zip(rows, answers):
        claims = parse_claims_exact(ans)
        if claims is None:
            parse_fail += 1
            judge.parse_failures += 1
        r["judge_raw"] = ans
        r["claims"] = claims

    # Rewrite gens.jsonl
    with gens_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # Stratified audit
    rng = random.Random(int(cfg.get("audit_seed", 0)))
    per = int(cfg.get("audit_per_cell", 5))
    buckets: Dict[Tuple[Any, ...], List[dict]] = defaultdict(list)
    for r in rows:
        jy = r.get("claims")
        if jy is None:
            continue
        cond = r.get("condition")
        if cond == "base":
            ck = "base"
        elif cond == "concept" and int(float(r.get("alpha", 0))) == 2:
            ck = "concept_a2"
        elif cond == "concept" and int(float(r.get("alpha", 0))) == 4:
            ck = "concept_a4"
        else:
            continue
        key = (r.get("set"), ck, "YES" if jy == 1 else "NO")
        if r.get("set") in ("F", "I"):
            buckets[key].append(r)

    audit_path = raw / "judge_audit.jsonl"
    with audit_path.open("w") as af:
        for key, lst in sorted(buckets.items(), key=lambda x: str(x[0])):
            take = list(lst)
            rng.shuffle(take)
            for r in take[:per]:
                af.write(
                    json.dumps(
                        {
                            "set": r.get("set"),
                            "pair": r.get("pair"),
                            "condition": r.get("condition"),
                            "alpha": r.get("alpha"),
                            "question": r.get("prompt"),
                            "response": r.get("response"),
                            "judge_raw": r.get("judge_raw"),
                            "claims": r.get("claims"),
                        }
                    )
                    + "\n"
                )

    summary = {
        "wo_id": "WO-005-judge",
        "git_hash": git_hash_str,
        "n_rows": len(rows),
        "n_calls": judge.n_calls,
        "parse_failures": parse_fail,
        "est_usd": judge.est_usd(),
        "prompt_tokens": judge.total_prompt_tokens,
        "completion_tokens": judge.total_completion_tokens,
        "seconds": time.perf_counter() - t0,
        "status": "COMPLETE",
    }
    (results_path / "judge_summary.json").write_text(json.dumps(summary, indent=2))
    log(f"DONE calls={judge.n_calls} usd≈{summary['est_usd']:.4f} parse_fail={parse_fail}")

    if ledger_path:
        import csv
        import datetime as dt

        path = Path(ledger_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        row = {
            "wo_id": "WO-005",
            "function": "wo_005_judge",
            "gpu_type": "CPU",
            "start_utc": now,
            "end_utc": now,
            "seconds": f"{summary['seconds']:.3f}",
            "est_usd": f"{summary['est_usd']:.6f}",
            "git_hash": git_hash_str,
        }
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "wo_id",
                    "function",
                    "gpu_type",
                    "start_utc",
                    "end_utc",
                    "seconds",
                    "est_usd",
                    "git_hash",
                ],
            )
            if write_header:
                w.writeheader()
            w.writerow(row)

    return summary


if __name__ == "__main__":
    run_wo_005_judge(
        config_path=str(REPO_ROOT / "configs/wo_005.yaml"),
        results_dir=str(REPO_ROOT / "results/WO-005"),
        ledger_path=str(REPO_ROOT / "results/budget_ledger.csv"),
        git_hash_str="local",
    )
