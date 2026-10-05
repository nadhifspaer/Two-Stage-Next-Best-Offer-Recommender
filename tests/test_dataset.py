# label window and training-label assembly tests for src/dataset.py
import datetime

import polars as pl

import src.dataset as dataset

WINDOW_START = datetime.date(2020, 9, 9)
WINDOW_END = datetime.date(2020, 9, 15)


def _synthetic_candidates_and_transactions():
    # c1: bought a1 in-window (positive), a2/a3/a4/a5 are non-positive candidates
    # c2: bought nothing in-window, so contributes zero rows
    candidates = pl.DataFrame(
        {
            "customer_id": ["c1"] * 5 + ["c2"] * 3,
            "article_id": ["a1", "a2", "a3", "a4", "a5", "a1", "a2", "a3"],
            "strategies": [["repurchase"]] * 5 + [["als"]] * 3,
            "n_strategies": [1] * 8,
            "best_rank": [1] * 8,
        }
    ).with_columns(pl.col("customer_id", "article_id").cast(pl.Categorical))

    transactions = pl.DataFrame(
        {
            "t_dat": [
                datetime.date(2020, 9, 10),  # c1 buys a1, inside the window -> positive
                datetime.date(2020, 9, 5),  # c1's pre-window history, not a label source
            ],
            "customer_id": ["c1", "c1"],
            "article_id": ["a1", "a2"],
            "price": [1.0, 1.0],
            "sales_channel_id": [1, 1],
        }
    ).with_columns(pl.col("customer_id", "article_id").cast(pl.Categorical))

    return candidates, transactions


def test_assemble_training_labels_negative_sampling_is_deterministic():
    pl.enable_string_cache()
    candidates, transactions = _synthetic_candidates_and_transactions()

    run1 = dataset.assemble_training_labels(candidates, transactions, WINDOW_START, WINDOW_END)
    run2 = dataset.assemble_training_labels(candidates, transactions, WINDOW_START, WINDOW_END)

    sort_cols = ["customer_id", "article_id"]
    run1 = run1.sort(sort_cols).with_columns(pl.col("strategies").list.sort())
    run2 = run2.sort(sort_cols).with_columns(pl.col("strategies").list.sort())
    assert run1.equals(run2)


def test_assemble_training_labels_customer_with_no_positive_contributes_no_rows():
    pl.enable_string_cache()
    candidates, transactions = _synthetic_candidates_and_transactions()

    labeled = dataset.assemble_training_labels(candidates, transactions, WINDOW_START, WINDOW_END)

    assert labeled.filter(pl.col("customer_id") == "c1").height > 0
    assert labeled.filter(pl.col("customer_id") == "c2").height == 0


def test_assemble_training_labels_negative_ratio_and_positive_flag():
    pl.enable_string_cache()
    candidates, transactions = _synthetic_candidates_and_transactions()

    labeled = dataset.assemble_training_labels(
        candidates, transactions, WINDOW_START, WINDOW_END, negatives_per_positive=2
    )
    c1_rows = labeled.filter(pl.col("customer_id") == "c1")
    positives = c1_rows.filter(pl.col("label") == 1)
    negatives = c1_rows.filter(pl.col("label") == 0)

    assert positives.height == 1
    assert positives["article_id"].to_list() == ["a1"]
    # 1 positive * negatives_per_positive=2, capped by the 4 available non-positive candidates
    assert negatives.height == 2
