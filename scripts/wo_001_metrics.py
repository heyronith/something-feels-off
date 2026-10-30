"""WO-001: metrics.json, per_concept.csv, figures from judged trials."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sfo.stats import wilson_ci  # noqa: E402


def load_trials(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def rate(flags: List[bool]) -> Dict[str, float]:
    k = sum(1 for x in flags if x)
    return wilson_ci(k, len(flags))


def compute_metrics(results_dir: Path) -> Dict[str, Any]:
    raw = results_dir / "raw"
    trials = load_trials(raw / "trials.jsonl")
    gen_sum = json.loads((results_dir / "generation_summary.json").read_text())
    judge_sum = {}
    jp = results_dir / "judge_summary.json"
    if jp.exists():
        judge_sum = json.loads(jp.read_text())

    inject = [t for t in trials if t["condition"] == "inject"]
    control = [t for t in trials if t["condition"] == "control"]
    random_t = [t for t in trials if t["condition"] == "random"]
    y_s = [t for t in trials if t["condition"] == "yesbias_steered"]
    y_u = [t for t in trials if t["condition"] == "yesbias_unsteered"]

    assert len(inject) == 600
    assert len(control) == 300
    assert len(random_t) == 200
    assert len(y_s) == 150
    assert len(y_u) == 50

    metrics: Dict[str, Any] = {"alpha": {}}
    for alpha in (2.0, 4.0):
        a_key = str(int(alpha)) if alpha == int(alpha) else str(alpha)
        inj_a = [t for t in inject if float(t["alpha"]) == alpha]
        rnd_a = [t for t in random_t if float(t["alpha"]) == alpha]
        det = [bool(t["judge_detection"]) for t in inj_a]
        ident = [bool(t["judge_ident"]) for t in inj_a]
        both = [d and i for d, i in zip(det, ident)]
        id_given = [
            bool(t["judge_ident"])
            for t in inj_a
            if t["judge_detection"] is True
        ]
        metrics["alpha"][a_key] = {
            "tpr": rate(det),
            "introspection_rate": rate(both),
            "id_given_detect": wilson_ci(sum(id_given), len(id_given))
            if id_given
            else wilson_ci(0, 0),
            "random_detect_rate": rate(
                [bool(t["judge_detection"]) for t in rnd_a]
            ),
        }

    metrics["fpr"] = rate([bool(t["judge_detection"]) for t in control])
    metrics["yes_rate_steered"] = rate([t["yesbias_label"] == "YES" for t in y_s])
    metrics["yes_rate_unsteered"] = rate([t["yesbias_label"] == "YES" for t in y_u])
    metrics["yesbias_ambiguous"] = int(
        sum(1 for t in y_s + y_u if t["yesbias_label"] == "AMBIGUOUS")
    )
    metrics["norms"] = gen_sum.get("norms", {})
    metrics["judge"] = {
        "model": judge_sum.get("model", "gpt-4.1-mini"),
        "n_calls": judge_sum.get("n_calls", 0),
        "parse_failures": judge_sum.get("parse_failures", 0),
        "tokens": judge_sum.get("tokens", {}),
        "est_usd": judge_sum.get("est_usd", 0.0),
    }
    metrics["sampling"] = gen_sum.get("sampling", {})
    metrics["git_hash"] = gen_sum.get("git_hash")
    metrics["model_revision"] = gen_sum.get("model_revision")

    # per-concept table
    vec_path = raw / "concept_vectors_L37.pt"
    pack = torch.load(vec_path, map_location="cpu", weights_only=False)
    concepts = pack["concepts"]
    norms = pack["norms"]
    per_rows = []
    for ci, concept in enumerate(concepts):
        row = {"concept": concept, "norm": float(norms[ci])}
        for alpha, tag in ((2.0, "2"), (4.0, "4")):
            subset = [
                t
                for t in inject
                if t["concept"] == concept and float(t["alpha"]) == alpha
            ]
            det = [bool(t["judge_detection"]) for t in subset]
            both = [
                bool(t["judge_detection"]) and bool(t["judge_ident"]) for t in subset
            ]
            row[f"tpr@{tag}"] = sum(det) / len(det) if det else 0.0
            row[f"introspection@{tag}"] = sum(both) / len(both) if both else 0.0
        per_rows.append(row)

    csv_path = raw / "per_concept.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "concept",
                "norm",
                "tpr@2",
                "tpr@4",
                "introspection@2",
                "introspection@4",
            ],
        )
        w.writeheader()
        w.writerows(per_rows)

    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics, per_rows


def make_figures(results_dir: Path, metrics: Dict[str, Any], per_rows: List[Dict]):
    # fig_rates
    labels = []
    vals = []
    los = []
    his = []
    ns = []

    def add(label, d):
        labels.append(label)
        vals.append(d["p"])
        los.append(d["p"] - d["lo"])
        his.append(d["hi"] - d["p"])
        ns.append(d["n"])

    add("tpr@2", metrics["alpha"]["2"]["tpr"])
    add("tpr@4", metrics["alpha"]["4"]["tpr"])
    add("intro@2", metrics["alpha"]["2"]["introspection_rate"])
    add("intro@4", metrics["alpha"]["4"]["introspection_rate"])
    add("fpr", metrics["fpr"])
    add("rand@2", metrics["alpha"]["2"]["random_detect_rate"])
    add("rand@4", metrics["alpha"]["4"]["random_detect_rate"])
    add("yes_steer", metrics["yes_rate_steered"])
    add("yes_unsteer", metrics["yes_rate_unsteered"])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = range(len(labels))
    ax.bar(x, vals, yerr=[los, his], capsize=3, color="#4C72B0")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.05)
    caption_n = ", ".join(f"{l}: n={n}" for l, n in zip(labels, ns))
    ax.set_title("Detection / introspection / control rates (Wilson 95% CI)")
    fig.text(0.01, 0.01, f"n per bar: {caption_n}", fontsize=7, wrap=True)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(results_dir / "fig_rates.png", dpi=150)
    plt.close(fig)

    # fig_per_concept: tpr@4 vs norm
    fig, ax = plt.subplots(figsize=(9, 7))
    xs = [r["norm"] for r in per_rows]
    ys = [r["tpr@4"] for r in per_rows]
    ax.scatter(xs, ys, s=20)
    for r in per_rows:
        ax.annotate(r["concept"], (r["norm"], r["tpr@4"]), fontsize=6, alpha=0.8)
    ax.set_xlabel(r"$\|v_c\|$")
    ax.set_ylabel("tpr@4")
    ax.set_title("Per-concept TPR at α=4 vs vector L2 norm")
    fig.tight_layout()
    fig.savefig(results_dir / "fig_per_concept.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    rd = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "results" / "WO-001"
    m, per = compute_metrics(rd)
    make_figures(rd, m, per)
    print("Wrote metrics.json and figures")
