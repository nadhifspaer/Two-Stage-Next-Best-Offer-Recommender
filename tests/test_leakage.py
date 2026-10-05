# temporal leakage checks for src/features.py and src/dataset.py
import datetime

import polars as pl

import src.dataset as dataset
import src.features as features


def test_features_never_use_transactions_on_or_after_window_start():
    # a purchase dated exactly on window_start is the label source for that window, never a feature source;
    # only the strictly-earlier purchase may reach a feature value
    pl.enable_string_cache()
    window_start = datetime.date(2020, 9, 9)
    window_end = datetime.date(2020, 9, 15)

    transactions = pl.DataFrame(
        {
            "t_dat": [
                datetime.date(2020, 9, 5),  # pre-window: the only transaction features may see
                datetime.date(2020, 9, 9),  # window_start itself: label source, not a feature source
            ],
            "customer_id": ["c1", "c1"],
            "article_id": ["a2", "a1"],
            "price": [1.0, 1.0],
            "sales_channel_id": [1, 1],
        }
    ).with_columns(pl.col("customer_id", "article_id").cast(pl.Categorical))

    customers = pl.DataFrame(
        {
            "customer_id": ["c1"],
            "age": [30],
            "club_member_status": ["ACTIVE"],
            "fashion_news_frequency": ["NONE"],
        }
    ).with_columns(pl.col("customer_id").cast(pl.Categorical))

    candidates = pl.DataFrame(
        {
            "customer_id": ["c1", "c1"],
            "article_id": ["a1", "a2"],
            "strategies": [["repurchase"], ["repurchase"]],
            "n_strategies": [1, 1],
            "best_rank": [1, 1],
        }
    ).with_columns(pl.col("customer_id", "article_id").cast(pl.Categorical))

    labeled = dataset.assemble_training_labels(candidates, transactions, window_start, window_end)
    a1_row = labeled.filter(pl.col("article_id") == "a1").to_dicts()[0]
    assert a1_row["label"] == 1  # purchased on window_start, correctly a positive

    customer_block = features.customer_features(transactions, customers, window_start)
    c1 = customer_block.to_dicts()[0]
    # only the 09-05 purchase counts; the 09-09 (window_start) purchase must not
    assert c1["txn_count_total"] == 1
    assert c1["days_since_last_purchase"] == 4  # 09-05 to 09-09


def test_validation_window_never_used_as_a_training_label_window():
    windows = dataset.label_windows()
    validation_window = next(w for w in windows if w["split"] == "validation")
    train_windows = [w for w in windows if w["split"] == "train"]

    assert len(train_windows) == 4
    for w in train_windows:
        assert w["window_end"] < validation_window["window_start"]


def test_validation_window_transactions_never_become_training_positives():
    # the last training window ends 2020-09-15; a purchase the day after
    # (2020-09-16, the validation window start) must not count as a positive
    pl.enable_string_cache()
    window_start = datetime.date(2020, 9, 9)
    window_end = datetime.date(2020, 9, 15)

    transactions = pl.DataFrame(
        {
            "t_dat": [datetime.date(2020, 9, 16)],
            "customer_id": ["c1"],
            "article_id": ["a1"],
            "price": [1.0],
            "sales_channel_id": [1],
        }
    ).with_columns(pl.col("customer_id", "article_id").cast(pl.Categorical))

    candidates = pl.DataFrame(
        {
            "customer_id": ["c1"],
            "article_id": ["a1"],
            "strategies": [["repurchase"]],
            "n_strategies": [1],
            "best_rank": [1],
        }
    ).with_columns(pl.col("customer_id", "article_id").cast(pl.Categorical))

    labeled = dataset.assemble_training_labels(candidates, transactions, window_start, window_end)
    # c1's only purchase falls outside [window_start, window_end], so no
    # positive exists and the customer contributes zero rows
    assert labeled.height == 0


def test_no_feature_is_derived_from_label():
    # candidate_provenance_features must produce the same output regardless
    # of the label column's values, proving no feature reads it
    pl.enable_string_cache()
    candidates_a = pl.DataFrame(
        {
            "customer_id": ["c1", "c2"],
            "article_id": ["a1", "a2"],
            "strategies": [["repurchase"], ["als"]],
            "n_strategies": [1, 1],
            "best_rank": [1, 1],
            "rank_repurchase": [1, None],
            "rank_colour_variant": [None, None],
            "rank_recent_popularity": [None, None],
            "rank_segment_popularity": [None, None],
            "rank_als": [None, 1],
            "label": [1, 0],
        }
    ).with_columns(pl.col("customer_id", "article_id").cast(pl.Categorical))
    candidates_b = candidates_a.with_columns(pl.col("label").reverse())

    prov_a = features.candidate_provenance_features(candidates_a)
    prov_b = features.candidate_provenance_features(candidates_b)

    assert "label" not in prov_a.columns
    assert prov_a.equals(prov_b)
