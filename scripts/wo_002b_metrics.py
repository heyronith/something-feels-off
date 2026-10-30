"""WO-002b metrics.json + figures (fig_sweep, fig_geometry, fig_validation)."""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sfo.stats import wilson_ci  # noqa: E402

ARMS = ["none", "B1p", "B1p-null"]
BLOCKED_ARMS = ["B1p", "B1p-null"]
COND_KEYS = ["B-inj", "B-fid", "B-exp"]


def rate_bool(flags: List[bool]) -> Dict[str, float]:
    return wilson_ci(sum(1 for x in flags if x), len(flags))


def clean(x: Any) -> Any:
    """Recursively replace NaN/inf floats with None for strict JSON."""
    if isinstance(x, float):
        return None if (math.isnan(x) or math.isinf(x)) else x
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


def bootstrap_rel(
    per_concept_none: Dict[str, float],
    per_concept_arm: Dict[str, float],
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    """Relative change (arm-none)/none per concept, mean + bootstrap CI over concepts."""
    keys = sorted(set(per_concept_none) & set(per_concept_arm))
    if not keys:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    none = np.array([per_concept_none[k] for k in keys])
    arm = np.array([per_concept_arm[k] for k in keys])
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(np.abs(none) > 1e-9, (arm - none) / none, np.nan)
    rel = rel[np.isfinite(rel)]
    if len(rel) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    boots = np.array(
        [float(np.mean(rel[rng.integers(0, len(rel), len(rel))])) for _ in range(n_boot)]
    )
    return {
        "mean": float(np.mean(rel)),
        "lo": float(np.quantile(boots, 0.025)),
        "hi": float(np.quantile(boots, 0.975)),
        "n": int(len(rel)),
    }


def main(results_dir: Path, config_path: Optional[Path] = None) -> None:
    raw = results_dir / "raw"
    config_path = config_path or (REPO_ROOT / "configs" / "wo_002b.yaml")
    cfg = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    n_boot = int(cfg.get("bootstrap_resamples", 2000))
    boot_seed = int(cfg.get("bootstrap_seed", 0))
    ret_thr = float(cfg.get("retention_threshold", 0.80))

    partA = json.loads((results_dir / "partA_summary.json").read_text())["partA"]
    partB_sum = json.loads((results_dir / "partB_summary.json").read_text())
    judge = json.loads((results_dir / "judge_summary.json").read_text())
    rows = [
        json.loads(l) for l in (raw / "partB_trials.jsonl").read_text().splitlines() if l.strip()
    ]

    # steered-damage KL per arm (from kl_raw.jsonl; fall back to partB_summary)
    kl_by_arm: Dict[str, List[float]] = defaultdict(list)
    kl_path = raw / "kl_raw.jsonl"
    if kl_path.exists():
        for l in kl_path.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                kl_by_arm[r["arm"]].append(float(r["kl"]))
    kl_mean: Dict[str, Optional[float]] = {}
    for arm in ARMS:
        if kl_by_arm.get(arm):
            kl_mean[arm] = float(np.mean(kl_by_arm[arm]))
        else:
            kl_mean[arm] = (partB_sum.get("kl_means") or {}).get(arm)

    def sel(cond: str, arm: str) -> List[dict]:
        return [r for r in rows if r["condition"] == cond and r["arm"] == arm]

    def arm_metrics(arm: str) -> Dict[str, Any]:
        inj, fid, exp = sel("B-inj", arm), sel("B-fid", arm), sel("B-exp", arm)
        # Rates use parsed judge outputs only; parse failures reported separately.
        inj_parsed = [r for r in inj if r["judge_detection"] is not None]
        det = [bool(r["judge_detection"]) for r in inj_parsed]
        id_given = [
            bool(r["judge_ident"])
            for r in inj_parsed
            if r["judge_detection"] is True and r["judge_ident"] is not None
        ]
        degen = {
            c: rate_bool([bool(r["degenerate"]) for r in sel(c, arm)]) for c in COND_KEYS
        }
        return {
            "tpr": rate_bool(det),
            "id_given_detect": wilson_ci(sum(id_given), len(id_given)),
            "fid": rate_bool([r["match"] is not None for r in fid]),
            "exp": rate_bool([r["match"] is not None for r in exp]),
            "degenerate": degen,
            "kl_steered": kl_mean.get(arm),
        }

    partB: Dict[str, Any] = {arm: arm_metrics(arm) for arm in ARMS}

    kn, kb, kz = kl_mean.get("none"), kl_mean.get("B1p"), kl_mean.get("B1p-null")
    partB["kl_ratio"] = (kb / kn) if (kn and kb is not None) else None  # KL_B1p / KL_none
    partB["kl_ratio_null"] = (kz / kn) if (kn and kz is not None) else None

    def per_concept_rate(condition: str, arm: str, field: str) -> Dict[str, float]:
        d = defaultdict(list)
        for r in rows:
            if r["condition"] == condition and r["arm"] == arm and r.get("concept"):
                if field == "det":
                    if r["judge_detection"] is None:
                        continue
                    d[r["concept"]].append(1.0 if r["judge_detection"] else 0.0)
                else:  # fid / exp: concept-match
                    d[r["concept"]].append(1.0 if r["match"] is not None else 0.0)
        return {k: float(np.mean(v)) for k, v in d.items()}

    rel_change: Dict[str, Any] = {}
    for arm in BLOCKED_ARMS:
        rel_change[arm] = {
            "tpr": bootstrap_rel(
                per_concept_rate("B-inj", "none", "det"),
                per_concept_rate("B-inj", arm, "det"),
                n_boot,
                boot_seed,
            ),
            "fid": bootstrap_rel(
                per_concept_rate("B-fid", "none", "fid"),
                per_concept_rate("B-fid", arm, "fid"),
                n_boot,
                boot_seed,
            ),
            "exp": bootstrap_rel(
                per_concept_rate("B-exp", "none", "exp"),
                per_concept_rate("B-exp", arm, "exp"),
                n_boot,
                boot_seed,
            ),
        }
    partB["rel_change"] = rel_change
    partB["parse_failures"] = judge.get("parse_failures", 0)

    metrics = {
        "partA": partA,
        "partB": partB,
        "judge": {
            "n_calls": judge["n_calls"],
            "tokens": judge["tokens"],
            "est_usd": judge["est_usd"],
        },
        "git_hash": partB_sum.get("git_hash"),
        "model_revision": partB_sum.get("model_revision"),
    }
    (results_dir / "metrics.json").write_text(json.dumps(clean(metrics), indent=2))

    # ------------------------------------------------------------ fig_sweep
    sweep = partA["sweep"]
    l0s = sorted(int(k) for k in sweep.keys())
    l0_star = partA.get("l0_star")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, ylabel in (
        (axes[0], "rec", "recovery"),
        (axes[1], "ret", "retention"),
    ):
        ax.plot(
            l0s,
            [sweep[str(l)]["B1p"][key] for l in l0s],
            "o-",
            label="B1p",
        )
        ax.plot(
            l0s,
            [sweep[str(l)]["B1p_null"][key] for l in l0s],
            "s--",
            label="B1p-null",
        )
        if l0_star is not None:
            ax.axvline(l0_star, color="k", ls="--", lw=1, label=f"l0*={l0_star}")
        ax.set_xticks(l0s)
        ax.set_xlabel("l0")
        ax.set_ylabel(ylabel)
    axes[1].axhline(ret_thr, color="gray", ls=":", lw=1, label=f"threshold {ret_thr:g}")
    axes[0].set_title("Recovery vs l0 (dir-dev subset)")
    axes[1].set_title("Retention vs l0 (dir-dev subset)")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(results_dir / "fig_sweep.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------ fig_geometry
    per = partA["per_layer"]
    ls = sorted(int(k) for k in per.keys())

    def col(key: str) -> List[float]:
        out = []
        for l in ls:
            v = per[str(l)].get(key)
            out.append(float("nan") if v is None else float(v))
        return out

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(ls, col("d_norm_ratio"), "-o", ms=3)
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("‖d_l‖ / median twin residual norm")
    axes[0].set_title("Direction norm ratio")
    for key, label in (
        ("cos_rand", "cos(d, d_rand)"),
        ("cos_mean_v", "cos(d, mean v̂)"),
        ("max_abs_cos_val50", "max |cos| val50"),
        ("p95_abs_cos_val50", "p95 |cos| val50"),
        ("cos_wo002", "cos(d, WO-002 d)"),
        ("cos_next", "cos(d_l, d_{l+1})"),
    ):
        axes[1].plot(ls, col(key), label=label)
    axes[1].legend(fontsize=7)
    axes[1].set_xlabel("layer")
    axes[1].set_title("Direction cosines")
    fig.tight_layout()
    fig.savefig(results_dir / "fig_geometry.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------ fig_validation
    labels = ["TPR", "ID|detect", "FID", "EXP"]
    keys = ["tpr", "id_given_detect", "fid", "exp"]
    x = np.arange(len(labels))
    width = 0.25
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), gridspec_kw={"width_ratios": [3, 2.4, 1.4]})

    ax = axes[0]
    for i, arm in enumerate(ARMS):
        vals = [partB[arm][k]["p"] for k in keys]
        lo = [partB[arm][k]["p"] - partB[arm][k]["lo"] for k in keys]
        hi = [partB[arm][k]["hi"] - partB[arm][k]["p"] for k in keys]
        ax.bar(x + (i - 1) * width, vals, width, yerr=[lo, hi], label=arm, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Part B rates (50 WO-001 concepts)")

    ax = axes[1]
    xd = np.arange(len(COND_KEYS))
    for i, arm in enumerate(ARMS):
        d = partB[arm]["degenerate"]
        vals = [d[c]["p"] for c in COND_KEYS]
        lo = [d[c]["p"] - d[c]["lo"] for c in COND_KEYS]
        hi = [d[c]["hi"] - d[c]["p"] for c in COND_KEYS]
        ax.bar(xd + (i - 1) * width, vals, width, yerr=[lo, hi], label=arm, capsize=2)
    ax.set_xticks(xd)
    ax.set_xticklabels(COND_KEYS)
    ax.set_ylim(0, 1.05)
    ax.set_title("Degenerate-output rate")

    ax = axes[2]
    kvals = [kl_mean.get(a) if kl_mean.get(a) is not None else 0.0 for a in ARMS]
    ax.bar(np.arange(len(ARMS)), kvals, color=["C0", "C1", "C2"])
    ax.set_xticks(np.arange(len(ARMS)))
    ax.set_xticklabels(ARMS, fontsize=8)
    ax.set_ylabel("mean KL(p_twin ‖ p_steered)")
    ax.set_title("Steered-damage KL")

    fig.tight_layout()
    fig.savefig(results_dir / "fig_validation.png", dpi=150)
    plt.close(fig)
    print("Wrote metrics.json and figures")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "results" / "WO-002b")
