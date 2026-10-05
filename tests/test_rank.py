# feature allowlist, categorical mapping, and training determinism tests
# for src/rank.py
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import src.rank as rank

TRAINING_MATRIX_PATH = Path("data/processed/features/train/window-2020-08-19.parquet")


def test_feature_columns_excludes_keys_label_and_window():
    for col in rank.EXCLUDED_COLUMNS:
        assert col not in rank.FEATURE_COLUMNS


def test_feature_columns_has_no_duplicates():
    assert len(rank.FEATURE_COLUMNS) == len(set(rank.FEATURE_COLUMNS))


@pytest.mark.skipif(
    not TRAINING_MATRIX_PATH.exists(),
    reason="training matrix not built on this machine",
)
def test_feature_columns_all_exist_in_the_real_training_matrix():
    # checked against the actual pipeline artifact, not a hand-written list,
    # so a future rename in src/features.py that FEATURE_COLUMNS misses is caught
    actual_columns = set(pl.read_parquet(TRAINING_MATRIX_PATH, n_rows=0).columns)
    for col in rank.FEATURE_COLUMNS:
        assert col in actual_columns
    for col in rank.EXCLUDED_COLUMNS:
        assert col in actual_columns


def test_apply_categorical_mapping_unseen_category_becomes_unknown_code():
    train_df = pl.DataFrame({"age_bucket": ["<25", "25-34", "<25"]})
    mapping = rank.fit_categorical_mapping(train_df, columns=["age_bucket"])

    # "65+" never appeared when the mapping was fit
    inference_df = pl.DataFrame({"age_bucket": ["<25", "65+", "25-34"]})
    encoded = rank.apply_categorical_mapping(inference_df, mapping, columns=["age_bucket"])
    codes = encoded["age_bucket"].to_list()

    known_lt25 = mapping["age_bucket"]["<25"]
    known_2534 = mapping["age_bucket"]["25-34"]
    assert known_lt25 != rank.UNKNOWN_CATEGORY_CODE
    assert known_2534 != rank.UNKNOWN_CATEGORY_CODE
    # known categories keep their fitted codes, unshifted by the unseen one
    assert codes[0] == known_lt25
    assert codes[2] == known_2534
    assert codes[1] == rank.UNKNOWN_CATEGORY_CODE


def _synthetic_training_matrix(n_customers_per_window: int = 12, n_candidates_per_customer: int = 10):
    rng = np.random.default_rng(0)
    windows = rank.EARLY_STOPPING_TRAIN_WINDOWS + [rank.EARLY_STOPPING_VALIDATION_WINDOW]
    age_buckets = ["<25", "25-34", "35-44"]
    departments = ["dept1", "dept2"]

    rows = []
    for window in windows:
        for c in range(n_customers_per_window):
            customer_id = f"{window}-c{c}"
            for a in range(n_candidates_per_customer):
                rows.append(
                    {
                        "window_start": window,
                        "customer_id": customer_id,
                        "article_id": f"a{a}",
                        # first candidate per customer is the positive, rest are negatives
                        "label": 1 if a == 0 else 0,
                        "prior_purchase_count_this_article": int(rng.integers(0, 3)),
                        "days_since_last_purchase_this_article": float(rng.integers(0, 300)),
                        "prior_purchase_count_same_product_code": int(rng.integers(0, 5)),
                        "prior_purchase_count_same_colour": int(rng.integers(0, 5)),
                        "prior_purchase_count_same_department": int(rng.integers(0, 5)),
                        "price_ratio_to_customer_mean": float(rng.uniform(0.1, 3.0)),
                        "text_cosine_similarity": float(rng.uniform(-1, 1)),
                        "n_strategies": int(rng.integers(1, 4)),
                        "best_rank": int(rng.integers(1, 30)),
                        "rank_repurchase": float(rng.integers(1, 30)),
                        "rank_colour_variant": float(rng.integers(1, 30)),
                        "rank_recent_popularity": float(rng.integers(1, 20)),
                        "rank_segment_popularity": float(rng.integers(1, 20)),
                        "rank_als": float(rng.integers(1, 30)),
                        "from_repurchase": bool(rng.integers(0, 2)),
                        "from_colour_variant": bool(rng.integers(0, 2)),
                        "from_recent_popularity": bool(rng.integers(0, 2)),
                        "from_segment_popularity": bool(rng.integers(0, 2)),
                        "from_als": bool(rng.integers(0, 2)),
                        "club_member_status": "ACTIVE",
                        "fashion_news_frequency": "NONE",
                        "age_bucket": age_buckets[c % len(age_buckets)],
                        "txn_count_total": int(rng.integers(0, 50)),
                        "distinct_article_count_total": int(rng.integers(0, 50)),
                        "mean_purchase_price": float(rng.uniform(0.01, 0.1)),
                        "std_purchase_price": float(rng.uniform(0.0, 0.05)),
                        "share_sales_channel_1": float(rng.uniform(0, 1)),
                        "share_sales_channel_2": float(rng.uniform(0, 1)),
                        "txn_count_1w": int(rng.integers(0, 5)),
                        "distinct_article_count_1w": int(rng.integers(0, 5)),
                        "txn_count_4w": int(rng.integers(0, 10)),
                        "distinct_article_count_4w": int(rng.integers(0, 10)),
                        "txn_count_12w": int(rng.integers(0, 20)),
                        "distinct_article_count_12w": int(rng.integers(0, 20)),
                        "days_since_last_purchase": float(rng.integers(0, 300)),
                        "product_type_name": "t1",
                        "department_name": departments[a % len(departments)],
                        "index_group_name": "ig1",
                        "garment_group_name": "g1",
                        "article_mean_price": float(rng.uniform(0.01, 0.1)),
                        "distinct_buyers_total": int(rng.integers(0, 500)),
                        "purchase_count_1w": int(rng.integers(0, 200)),
                        "purchase_count_4w": int(rng.integers(0, 500)),
                        "purchase_count_12w": int(rng.integers(0, 1000)),
                        "purchase_count_wow_trend": int(rng.integers(-100, 100)),
                        "days_since_first_sale": float(rng.integers(0, 700)),
                        "days_since_last_sale": float(rng.integers(0, 30)),
                    }
                )
    return pl.DataFrame(rows)


def test_train_ranker_is_deterministic_across_two_runs():
    df = _synthetic_training_matrix()
    mapping = rank.fit_categorical_mapping(df)

    model1, meta1 = rank.train_ranker(df, mapping)
    model2, meta2 = rank.train_ranker(df, mapping)

    assert meta1["best_iteration"] == meta2["best_iteration"]
    X = rank.to_pandas_X(rank.prepare_matrix(df, mapping))
    preds1 = model1.predict(X)
    preds2 = model2.predict(X)
    assert np.array_equal(preds1, preds2)


def test_train_classifier_is_deterministic_across_two_runs():
    df = _synthetic_training_matrix()
    mapping = rank.fit_categorical_mapping(df)

    model1, meta1 = rank.train_classifier(df, mapping)
    model2, meta2 = rank.train_classifier(df, mapping)

    assert meta1["best_iteration"] == meta2["best_iteration"]
    X = rank.to_pandas_X(rank.prepare_matrix(df, mapping))
    preds1 = model1.predict_proba(X)
    preds2 = model2.predict_proba(X)
    assert np.array_equal(preds1, preds2)


def test_score_distributions_match_direct_prediction_and_batching_is_invisible():
    df = _synthetic_training_matrix()
    mapping = rank.fit_categorical_mapping(df)
    ranker, _ = rank.train_ranker(df, mapping)
    classifier, _ = rank.train_classifier(df, mapping)

    whole = rank.score_distributions(ranker, classifier, df, mapping, n_bins=20, batch_rows=10**9)
    batched = rank.score_distributions(ranker, classifier, df, mapping, n_bins=20, batch_rows=37)
    assert whole == batched

    scored = rank.prepare_matrix(df, mapping)
    X = rank.to_pandas_X(scored)
    labels = scored["label"].to_numpy()
    probs = classifier.predict_proba(X)[:, 1]
    edges = np.array(whole["classifier"]["bin_edges"])
    assert whole["rows"] == df.height and whole["positives"] == int(labels.sum())
    assert whole["classifier"]["negative_counts"] == np.histogram(probs[labels == 0], bins=edges)[0].tolist()
    assert whole["classifier"]["positive_counts"] == np.histogram(probs[labels == 1], bins=edges)[0].tolist()
    assert sum(whole["ranker"]["negative_counts"]) + sum(whole["ranker"]["positive_counts"]) == df.height


def test_write_score_distributions_records_pin_and_leaves_no_temp_file(tmp_path):
    import json

    import pipelines.report_score_distributions as report_score_distributions

    df = _synthetic_training_matrix()
    mapping = rank.fit_categorical_mapping(df)
    ranker, _ = rank.train_ranker(df, mapping)
    classifier, _ = rank.train_classifier(df, mapping)
    path = tmp_path / "models" / "_score_distributions.json"
    report_score_distributions.write(ranker, classifier, df, mapping, {"ranker_run_id": "r", "classifier_run_id": "c"}, path)

    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["ranker_run_id"] == "r" and stored["classifier_run_id"] == "c"
    assert stored["rows"] == df.height
    assert [p.name for p in path.parent.iterdir()] == ["_score_distributions.json"]
