# isotonic calibration tests for src/calibrate.py
import numpy as np
import polars as pl

import src.calibrate as calibrate


def test_calibrated_output_is_monotonic_in_raw_output():
    rng = np.random.default_rng(0)
    raw = np.sort(rng.uniform(0, 1, 2000))
    y = (rng.random(2000) < raw).astype(int)

    calibrator = calibrate.fit_calibrator(y, raw)
    calibrated = calibrate.apply_calibration(calibrator, raw)

    assert np.all(np.diff(calibrated) >= -1e-12)


def test_calibrated_output_is_bounded_in_unit_interval():
    rng = np.random.default_rng(1)
    raw = rng.uniform(0, 1, 500)
    y = (rng.random(500) < 0.05).astype(int)

    calibrator = calibrate.fit_calibrator(y, raw)
    # evaluate outside the fit range too, to exercise the out_of_bounds clip
    calibrated = calibrate.apply_calibration(calibrator, np.array([-1.0, 0.0, 0.5, 1.0, 2.0]))

    assert calibrated.min() >= 0.0
    assert calibrated.max() <= 1.0


def test_fit_and_eval_customer_buckets_share_no_customer():
    customer_ids = pl.Series([f"c{i}" for i in range(5000)])
    bucket = calibrate.customer_bucket(customer_ids)

    fit_ids = set(customer_ids.filter(calibrate.fit_bucket_mask(bucket)).to_list())
    eval_ids = set(customer_ids.filter(calibrate.eval_bucket_mask(bucket)).to_list())

    assert len(fit_ids) > 0
    assert len(eval_ids) > 0
    assert fit_ids.isdisjoint(eval_ids)
    assert 0.15 < len(fit_ids) / 5000 < 0.25
    assert 0.15 < len(eval_ids) / 5000 < 0.25


def test_brier_skill_score_perfect_and_constant_predictor():
    y = np.array([0, 0, 1, 1])
    base_rate = y.mean()

    perfect_brier = calibrate.brier_score(y, np.array([0.0, 0.0, 1.0, 1.0]))
    assert calibrate.brier_skill_score(perfect_brier, base_rate) == 1.0

    constant_brier = calibrate.brier_score(y, np.full_like(y, base_rate, dtype=float))
    assert abs(calibrate.brier_skill_score(constant_brier, base_rate)) < 1e-9


def test_quantile_reliability_table_bins_have_near_equal_row_counts():
    rng = np.random.default_rng(2)
    n = 1000
    y = rng.integers(0, 2, n)
    probs = rng.random(n)

    keys = pl.DataFrame({"customer_id": [f"c{i}" for i in range(n)], "article_id": ["a0"] * n})
    table = calibrate.quantile_reliability_table(y, probs, keys, n_bins=10)
    assert table.height == 10
    counts = table["n"].to_list()
    assert max(counts) - min(counts) <= 1


def _tied_eval_frame(n: int = 60_000, seed: int = 3):
    # isotonic-like output: few distinct probability values, so bin edges cut tied runs
    rng = np.random.default_rng(seed)
    probs = np.sort(rng.random(40))[rng.integers(0, 40, size=n)]
    y = (rng.random(n) < probs * 0.05).astype(np.int8)
    keys = pl.DataFrame(
        {"customer_id": [f"c{i % 9000}" for i in range(n)], "article_id": [f"a{i // 9000}" for i in range(n)]}
    )
    return y, probs, keys


def test_quantile_reliability_table_is_independent_of_row_order():
    # regression guard: rank(method="ordinal") broke ties by row position, so a different row order moved positives between bins
    y, probs, keys = _tied_eval_frame()
    reference = calibrate.quantile_reliability_table(y, probs, keys, n_bins=10)
    rng = np.random.default_rng(11)
    for _ in range(2):
        perm = rng.permutation(len(y))
        shuffled = calibrate.quantile_reliability_table(y[perm], probs[perm], keys[perm], n_bins=10)
        assert shuffled.equals(reference)


def test_quantile_bin_index_is_independent_of_row_order():
    y, probs, keys = _tied_eval_frame()
    reference = calibrate.quantile_bin_index(probs, keys, n_bins=10)
    perm = np.random.default_rng(5).permutation(len(y))
    shuffled = calibrate.quantile_bin_index(probs[perm], keys[perm], n_bins=10)
    assert np.array_equal(shuffled, reference[perm])


def test_brier_score_is_independent_of_row_order_for_float32_input():
    y, probs, _keys = _tied_eval_frame()
    probs32 = probs.astype(np.float32)
    reference = calibrate.brier_score(y, probs32)
    rng = np.random.default_rng(9)
    for _ in range(2):
        perm = rng.permutation(len(y))
        assert calibrate.brier_score(y[perm], probs32[perm]) == reference
