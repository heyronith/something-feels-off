"""GPU run ledger context: always append a row in finally (incl. failures)."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from sfo.budget import append_ledger_row


@contextmanager
def gpu_ledger(
    ledger_path: str | Path,
    *,
    wo_id: str,
    function: str,
    gpu_type: str,
    n_cores: float,
    mem_gib: float,
    git_hash: str,
    volume_commit=None,
) -> Iterator[dict]:
    """
    Yields a mutable info dict. Always writes a ledger row on exit.
    If volume_commit is provided (Modal Volume.commit), call it after write.
    """
    start = datetime.now(timezone.utc)
    info: dict = {"error": None, "start_utc": start}
    try:
        yield info
    except Exception as e:  # noqa: BLE001
        info["error"] = str(e)
        raise
    finally:
        end = datetime.now(timezone.utc)
        append_ledger_row(
            ledger_path,
            wo_id=wo_id,
            function=function,
            gpu_type=gpu_type,
            start_utc=start,
            end_utc=end,
            n_cores=n_cores,
            mem_gib=mem_gib,
            git_hash=git_hash,
        )
        if volume_commit is not None:
            try:
                volume_commit()
            except Exception as ve:  # noqa: BLE001
                print(f"[ledger] volume.commit failed: {ve}")
