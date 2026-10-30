"""Cost ledger for Modal GPU runs."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

# Modal A100-80GB GPU $/s + non-preemptible CPU/memory rates (per WO-000).
GPU_RATE_PER_SEC = 0.000694
CPU_RATE_PER_CORE_SEC = 0.00003942
MEM_RATE_PER_GIB_SEC = 0.00000672

LEDGER_COLUMNS = [
    "wo_id",
    "function",
    "gpu_type",
    "start_utc",
    "end_utc",
    "seconds",
    "est_usd",
    "git_hash",
]


def estimate_usd(seconds: float, n_cores: float, mem_gib: float) -> float:
    """seconds × (0.000694 + 0.00003942 × n_cores + 0.00000672 × mem_gib)."""
    return seconds * (
        GPU_RATE_PER_SEC
        + CPU_RATE_PER_CORE_SEC * n_cores
        + MEM_RATE_PER_GIB_SEC * mem_gib
    )


def append_ledger_row(
    ledger_path: Union[str, Path],
    *,
    wo_id: str,
    function: str,
    gpu_type: str,
    start_utc: datetime,
    end_utc: datetime,
    n_cores: float,
    mem_gib: float,
    git_hash: str,
) -> dict:
    """Append one GPU-run row to results/budget_ledger.csv."""
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if start_utc.tzinfo is None:
        start_utc = start_utc.replace(tzinfo=timezone.utc)
    if end_utc.tzinfo is None:
        end_utc = end_utc.replace(tzinfo=timezone.utc)

    seconds = (end_utc - start_utc).total_seconds()
    est = estimate_usd(seconds, n_cores=n_cores, mem_gib=mem_gib)
    row = {
        "wo_id": wo_id,
        "function": function,
        "gpu_type": gpu_type,
        "start_utc": start_utc.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_utc": end_utc.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seconds": f"{seconds:.3f}",
        "est_usd": f"{est:.6f}",
        "git_hash": git_hash,
    }

    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return row
