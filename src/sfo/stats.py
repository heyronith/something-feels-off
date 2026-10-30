"""Wilson score intervals and rate helpers."""

from __future__ import annotations

import math
from typing import Dict


def wilson_ci(k: int, n: int, z: float = 1.96) -> Dict[str, float]:
    """Wilson 95% CI for binomial proportion k/n."""
    if n <= 0:
        return {"p": 0.0, "lo": 0.0, "hi": 0.0, "k": 0, "n": 0}
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p + z2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    lo = max(0.0, (center - margin) / denom)
    hi = min(1.0, (center + margin) / denom)
    return {"p": float(p), "lo": float(lo), "hi": float(hi), "k": int(k), "n": int(n)}
