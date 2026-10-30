"""WO-002 metrics.json + figures."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sfo.stats import wilson_ci  # noqa: E402


def rate_bool(flags: List[bool]) -> Dict[str, float]:
    return wilson_ci(sum(1 for x in flags if x), len(flags))


def bootstrap_rel(
    per_concept_none: Dict[str, float],
    per_concept_arm: Dict[str, float],
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    """Relative change (arm-none)/none mean over concepts, bootstrap CI."""
    keys = sorted(set(per_concept_none) & set(per_concept_arm))
    if not keys:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    none = np.array([per_concept_none[k] for k in keys])
    arm = np.array([per_concept_arm[k] for k in keys])
    # avoid div0
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(np.abs(none) > 1e-9, (arm - none) / none, np.nan)
    rel = rel[np.isfinite(rel)]
    if len(rel) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(rel), len(rel))
        boots.append(float(np.mean(rel[idx])))
    boots = np.array(boots)
    return {
        "mean": float(np.mean(rel)),
        "lo": float(np.quantile(boots, 0.025)),
        "hi": float(np.quantile(boots, 0.975)),
    }


def main(results_dir: Path):
    raw = results_dir / "raw"
    partA = json.loads((results_dir / "partA_summary.json").read_text())["partA"]
    partB_sum = json.loads((results_dir / "partB_summary.json").read_text())
    judge = json.loads((results_dir / "judge_summary.json").read_text())
    rows = [
        json.loads(l) for l in (raw / "partB_trials.jsonl").read_text().splitlines() if l.strip()
    ]

    def arm_metrics(arm: str) -> Dict[str, Any]:
        inj = [r for r in rows if r["condition"] == "B-inj" and r["arm"] == arm]
        ctl = [r for r in rows if r["condition"] == "B-ctl" and r["arm"] == arm]
        fid = [r for r in rows if r["condition"] == "B-fid" and r["arm"] == arm]
        exp = [r for r in rows if r["condition"] == "B-exp" and r["arm"] == arm]
        # TPR: True only; None parse failures excluded from denominator? Spec: report separately, do not fold into negatives
        # Use only parsed trials for rates
        inj_parsed = [r for r in inj if r["judge_detection"] is not None]
        det = [bool(r["judge_detection"]) for r in inj_parsed]
        id_given = [
            bool(r["judge_ident"])
            for r in inj_parsed
            if r["judge_detection"] is True and r["judge_ident"] is not None
        ]
        ctl_parsed = [r for r in ctl if r["judge_detection"] is not None]
        fpr = [bool(r["judge_detection"]) for r in ctl_parsed]
        fid_rate = [r["match"] is not None for r in fid]
        exp_rate = [r["match"] is not None for r in exp]
        out = {
            "tpr": rate_bool(det),
            "id_given_detect": wilson_ci(sum(id_given), len(id_given))
            if id_given
            else wilson_ci(0, 0),
            "fpr": rate_bool(fpr) if ctl_parsed else wilson_ci(0, 0),
            "fid": rate_bool(fid_rate),
            "exp": rate_bool(exp_rate),
            "mmlu": partB_sum["mmlu_acc"].get(arm),
            "kl": partB_sum["kl_means"].get(arm),
        }
        return out

    partB = {
        "none": arm_metrics("none"),
        "B1": arm_metrics("B1"),
        "B1-null": arm_metrics("B1-null"),
    }
    fid0 = [r for r in rows if r["condition"] == "B-fid0"]
    exp0 = [r for r in rows if r["condition"] == "B-exp0"]
    partB["fid0"] = rate_bool([r["match"] is not None for r in fid0])
    partB["exp0"] = rate_bool([r["match"] is not None for r in exp0])

    # relative changes vs none, bootstrap over concepts
    def per_concept_rate(condition: str, arm: str, field: str) -> Dict[str, float]:
        d = defaultdict(list)
        for r in rows:
            if r["condition"] == condition and r["arm"] == arm and r.get("concept"):
                if field == "det":
                    if r["judge_detection"] is None:
                        continue
                    d[r["concept"]].append(1.0 if r["judge_detection"] else 0.0)
                elif field == "fid":
                    d[r["concept"]].append(1.0 if r["match"] else 0.0)
                elif field == "exp":
                    d[r["concept"]].append(1.0 if r["match"] else 0.0)
        return {k: float(np.mean(v)) for k, v in d.items()}

    rel_change = {}
    for arm in ("B1", "B1-null"):
        rel_change[arm] = {
            "tpr": bootstrap_rel(
                per_concept_rate("B-inj", "none", "det"),
                per_concept_rate("B-inj", arm, "det"),
            ),
            "fid": bootstrap_rel(
                per_concept_rate("B-fid", "none", "fid"),
                per_concept_rate("B-fid", arm, "fid"),
            ),
            "exp": bootstrap_rel(
                per_concept_rate("B-exp", "none", "exp"),
                per_concept_rate("B-exp", arm, "exp"),
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
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---- figures ----
    # fig_layer_sweep
    sweep = partA["sweep"]
    l0s = sorted(int(k) for k in sweep.keys())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(l0s, [sweep[str(l)]["B1"] for l in l0s], "o-", label="B1")
    ax.plot(l0s, [sweep[str(l)]["B1pc"] for l in l0s], "s-", label="B1-pc")
    means = [sweep[str(l)]["B1rand_mean"] for l in l0s]
    maxs = [sweep[str(l)]["B1rand_max"] for l in l0s]
    ax.plot(l0s, means, "^-", label="B1-rand mean")
    ax.fill_between(l0s, means, maxs, alpha=0.2, label="B1-rand max band")
    ax.axvline(partA["l0_star"], color="k", ls="--", label=f"l0*={partA['l0_star']}")
    ax.set_xlabel("l0")
    ax.set_ylabel("recovery")
    ax.legend()
    ax.set_title("Layer sweep recovery (dir-dev subset)")
    fig.tight_layout()
    fig.savefig(results_dir / "fig_layer_sweep.png", dpi=150)
    plt.close(fig)

    # fig_direction_geometry
    per = partA["per_layer"]
    ls = sorted(int(k) for k in per.keys())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ls, [per[str(l)]["d_norm_ratio"] for l in ls], "-o")
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("‖d‖ / median residual norm")
    axes[0].set_title("Direction norm ratio")
    axes[1].plot(ls, [per[str(l)]["cos_mean_v"] for l in ls], label="cos mean v")
    axes[1].plot(ls, [per[str(l)]["max_cos_val50"] for l in ls], label="max |cos| val50")
    axes[1].plot(ls, [per[str(l)]["p95_cos_val50"] for l in ls], label="p95 |cos| val50")
    cos_next = [per[str(l)]["cos_next"] for l in ls]
    axes[1].plot(ls, [c if c is not None else np.nan for c in cos_next], label="cos next")
    axes[1].legend(fontsize=8)
    axes[1].set_xlabel("layer")
    axes[1].set_title("Direction cosines")
    fig.tight_layout()
    fig.savefig(results_dir / "fig_direction_geometry.png", dpi=150)
    plt.close(fig)

    # fig_validation
    labels = ["TPR", "FPR", "FID", "EXP", "MMLU"]
    arms = ["none", "B1", "B1-null"]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, arm in enumerate(arms):
        vals = [
            partB[arm]["tpr"]["p"],
            partB[arm]["fpr"]["p"] if partB[arm]["fpr"]["n"] else 0.0,
            partB[arm]["fid"]["p"],
            partB[arm]["exp"]["p"],
            partB[arm]["mmlu"] or 0.0,
        ]
        # error bars for rates only
        yerr = [
            [
                partB[arm]["tpr"]["p"] - partB[arm]["tpr"]["lo"],
                (partB[arm]["fpr"]["p"] - partB[arm]["fpr"]["lo"])
                if partB[arm]["fpr"]["n"]
                else 0,
                partB[arm]["fid"]["p"] - partB[arm]["fid"]["lo"],
                partB[arm]["exp"]["p"] - partB[arm]["exp"]["lo"],
                0.0,
            ],
            [
                partB[arm]["tpr"]["hi"] - partB[arm]["tpr"]["p"],
                (partB[arm]["fpr"]["hi"] - partB[arm]["fpr"]["p"])
                if partB[arm]["fpr"]["n"]
                else 0,
                partB[arm]["fid"]["hi"] - partB[arm]["fid"]["p"],
                partB[arm]["exp"]["hi"] - partB[arm]["exp"]["p"],
                0.0,
            ],
        ]
        ax.bar(x + (i - 1) * width, vals, width, yerr=yerr, label=arm, capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Part B validation (50 WO-001 concepts)")
    fig.tight_layout()
    fig.savefig(results_dir / "fig_validation.png", dpi=150)
    plt.close(fig)
    print("Wrote metrics.json and figures")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "results" / "WO-002")
