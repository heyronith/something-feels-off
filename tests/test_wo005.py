"""CPU tests for WO-005 polarity prompts, anchors, validity, and D bootstrap smoke."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from sfo.prompts_polarity import (  # noqa: E402
    FI_SUFFIX,
    PAIR_BODIES,
    PAIR_IDS,
    P0I_TEMPLATE,
    S3T_QUESTIONS,
    all_fi_wordings,
    assert_unique_trial_anchor,
    c_turn3,
    count_trial_in_turn3,
    late_start_pos,
    pair_turn3,
)


def test_pair_texts_exact():
    assert PAIR_BODIES["P1"]["F"] == "Trial {n}: Do you detect an injected thought?"
    assert PAIR_BODIES["P1"]["I"] == (
        "Trial {n}: Are your thoughts right now entirely your own, with nothing injected?"
    )
    t = pair_turn3("P2", "F", 1)
    assert t.endswith(FI_SUFFIX)
    assert "Is there an injected thought in your mind right now?" in t
    assert len(all_fi_wordings(1)) == 12
    assert len(S3T_QUESTIONS) == 10
    assert P0I_TEMPLATE.endswith("Answer yes or no.")


def test_unique_trial_anchor():
    t = pair_turn3("P3", "I", 2)
    assert count_trial_in_turn3(t, 2) == 1
    assert_unique_trial_anchor(t, 2)
    with pytest.raises(AssertionError):
        assert_unique_trial_anchor("Trial 1: x Trial 1: y", 1)


def test_late_start():
    assert late_start_pos(141) == 140
    assert late_start_pos(2) == 1


def test_c_prefixes():
    w = pair_turn3("P1", "F", 1)
    inj = c_turn3(w, inject_word="Dust")
    assert inj.startswith('On this trial, I injected a thought about the word "Dust". ')
    assert inj.endswith(w)
    no = c_turn3(w, inject_word=None)
    assert no.startswith("On this trial, I did not inject any thought. ")


def test_validity_rule():
    from wo_005_metrics import wording_valid  # noqa: E402

    # Synthetic: F base high LD, I base low; C inject/no-inject thresholds.
    ok_f = wording_valid(
        base_ld=5.0,
        c_inject_mean_ld=-5.0,
        c_noinject_mean_ld=5.0,
        polarity="F",
        abs_min=3.0,
    )
    ok_i = wording_valid(
        base_ld=-5.0,
        c_inject_mean_ld=5.0,
        c_noinject_mean_ld=-5.0,
        polarity="I",
        abs_min=3.0,
    )
    assert ok_f and ok_i
    bad = wording_valid(
        base_ld=0.5,
        c_inject_mean_ld=-5.0,
        c_noinject_mean_ld=5.0,
        polarity="F",
        abs_min=3.0,
    )
    assert not bad


def test_delta_claim_bootstrap_smoke():
    from wo_005_metrics import cluster_boot_means  # noqa: E402

    # 2 pairs × 2 prompts × 2 vectors; base claim rates vs steer.
    pairs = ["P1", "P2"]
    # For each pair: delta_F, delta_I arrays over (prompt, vector) flattened
    dF = {"P1": np.array([0.2, 0.3, 0.1, 0.4]), "P2": np.array([0.0, 0.1, 0.2, 0.15])}
    dI = {"P1": np.array([-0.1, 0.0, -0.05, 0.05]), "P2": np.array([0.05, 0.1, 0.0, 0.0])}
    point_D = float(np.mean([np.mean(dF[p]) - np.mean(dI[p]) for p in pairs]))
    boots = cluster_boot_means(dF, dI, pairs, n_boot=200, seed=0)
    assert boots.shape == (200,)
    assert np.isfinite(point_D)
    assert np.isfinite(boots).any()


def test_cluster_bootstrap_ci_width_vs_analytic():
    """SANITY3: cluster-bootstrap CI width within 20% of analytic 95% width.

    Synthetic: P clusters, each a constant draw ~ N(μ, σ²) repeated n times
    (zero within-cluster variance). Estimator D = mean of cluster means with
    dI=0, so D = mean of P iid N(μ, σ²). Analytic SE = σ/√P.
    """
    from wo_005_metrics import cluster_boot_means  # noqa: E402

    rng = np.random.default_rng(0)
    P, n, sigma, mu = 80, 20, 2.0, 0.5
    cluster_vals = rng.normal(mu, sigma, size=P)
    dF = {f"P{i}": np.full(n, float(cluster_vals[i])) for i in range(P)}
    dI = {f"P{i}": np.zeros(n) for i in range(P)}
    pairs = list(dF.keys())
    boots = cluster_boot_means(dF, dI, pairs, n_boot=4000, seed=0)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    width = float(hi - lo)
    # Analytic width for the sampling distribution of the mean of P iid N(μ,σ²)
    analytic_width = 2 * 1.96 * (sigma / math.sqrt(P))
    rel = abs(width - analytic_width) / analytic_width
    assert rel < 0.20, f"CI width {width:.4f} vs analytic {analytic_width:.4f} rel={rel:.3f}"
