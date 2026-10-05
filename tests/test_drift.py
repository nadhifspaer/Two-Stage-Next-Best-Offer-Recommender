# Drift check: threshold logic and Evidently PSI on generated frames
import numpy as np
import pandas as pd
import pytest

import monitoring.drift as drift

ROWS = 60_000


def _frame(seed: int, shift_columns: set[str] = frozenset(), probability_shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {}
    for c in drift.FEATURE_COLUMNS:
        if c in drift.CATEGORICAL_FEATURES:
            probs = [0.7, 0.3] if c not in shift_columns else [0.2, 0.8]
            data[c] = rng.choice(["a", "b"], size=ROWS, p=probs)
        else:
            data[c] = rng.normal(100.0 if c in shift_columns else 0.0, 1.0, ROWS)
    data[drift.PROBABILITY_COLUMN] = rng.beta(1, 50 + probability_shift, ROWS)
    return pd.DataFrame(data)


def test_same_distribution_passes():
    result = drift.evaluate_drift(_frame(1), _frame(2))
    assert result["passed"] and result["drifted_feature_count"] == 0 and result["probability_psi"] < drift.PSI_PROBABILITY_THRESHOLD


def test_few_drifted_features_stay_under_the_share_rule():
    shifted = set(drift.FEATURE_COLUMNS[:5])
    result = drift.evaluate_drift(_frame(1), _frame(2, shift_columns=shifted))
    assert result["passed"] and result["drifted_features"] == sorted(shifted)


def test_drifted_share_over_a_quarter_fails():
    shifted = set(drift.FEATURE_COLUMNS[:12])
    result = drift.evaluate_drift(_frame(1), _frame(2, shift_columns=shifted))
    assert not result["passed"] and result["share_rule_failed"] and not result["probability_rule_failed"]


def test_just_under_a_quarter_passes():
    shifted = set(drift.FEATURE_COLUMNS[:11])
    result = drift.evaluate_drift(_frame(1), _frame(2, shift_columns=shifted))
    assert result["drifted_feature_count"] == 11 and result["drifted_feature_share"] < drift.MAX_DRIFTED_FEATURE_SHARE and result["passed"]


def test_probability_shift_alone_fails():
    result = drift.evaluate_drift(_frame(1), _frame(2, probability_shift=400.0))
    assert not result["passed"] and result["probability_rule_failed"] and result["drifted_feature_count"] == 0


def test_too_few_rows_raises():
    psi = {c: 0.0 for c in drift.FEATURE_COLUMNS} | {drift.PROBABILITY_COLUMN: 0.0}
    with pytest.raises(ValueError, match="at least"):
        drift.decide(psi, drift.MIN_SAMPLE_ROWS - 1, drift.MIN_SAMPLE_ROWS)


def test_uncomputable_psi_counts_as_drifted():
    psi = {c: 0.0 for c in drift.FEATURE_COLUMNS} | {drift.PROBABILITY_COLUMN: 0.0}
    psi[drift.FEATURE_COLUMNS[0]] = float("nan")
    result = drift.decide(psi, ROWS, ROWS)
    assert result["drifted_features"] == [drift.FEATURE_COLUMNS[0]]
    psi[drift.PROBABILITY_COLUMN] = float("nan")
    assert not drift.decide(psi, ROWS, ROWS)["passed"]


def test_gate_follows_the_matched_comparison_not_the_unmatched_one():
    result = {"gated_matched": {"passed": True}, "diagnostic_unmatched": {"passed": False}}
    assert drift.gate_passed(result)
    result = {"gated_matched": {"passed": False}, "diagnostic_unmatched": {"passed": True}}
    assert not drift.gate_passed(result)
