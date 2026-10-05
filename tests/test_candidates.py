# candidate strategy and coverage tests for src/candidates.py
import datetime

import polars as pl

import src.candidates as candidates

WINDOW_START = datetime.date(2020, 9, 16)


def _synthetic_data():
    transactions = pl.DataFrame(
        {
            "t_dat": [
                datetime.date(2020, 9, 10),
                datetime.date(2020, 9, 12),
                datetime.date(2020, 9, 14),
                datetime.date(2020, 9, 15),
                datetime.date(2020, 9, 13),
                datetime.date(2020, 9, 11),
                datetime.date(2020, 9, 14),
                datetime.date(2020, 9, 12),
                datetime.date(2020, 9, 9),
                datetime.date(2020, 9, 10),
            ],
            "customer_id": ["c1", "c1", "c2", "c2", "c3", "c3", "c4", "c4", "c5", "c6"],
            "article_id": ["a1", "a2", "a1", "a3", "a1", "a2", "a3", "a4", "a2", "a1"],
            "price": [0.1] * 10,
            "sales_channel_id": [1] * 10,
        }
    ).with_columns(pl.col("customer_id", "article_id").cast(pl.Categorical))

    articles = pl.DataFrame(
        {
            "article_id": ["a1", "a2", "a3", "a4", "a5"],
            "product_code": ["p1", "p1", "p2", "p2", "p3"],
        }
    ).with_columns(pl.col("article_id", "product_code").cast(pl.Categorical))

    # c7 has zero transactions before the window, so it exercises the
    # no-history fallback path in the same run as customers with history
    customers = pl.DataFrame(
        {
            "customer_id": ["c1", "c2", "c3", "c4", "c5", "c6", "c7"],
            "age": [22, 40, None, 55, 30, 65, 19],
        }
    ).with_columns(pl.col("customer_id").cast(pl.Categorical))

    return transactions, articles, customers


def test_generate_candidates_chunked_is_deterministic_with_fixed_seed(tmp_path):
    pl.enable_string_cache()
    transactions, articles, customers = _synthetic_data()

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    # chunk_size smaller than the customer base so the test also exercises
    # chunk-boundary handling, not just a single pass
    candidates.generate_candidates_chunked(
        transactions, articles, customers, WINDOW_START, out1, chunk_size=2
    )
    candidates.generate_candidates_chunked(
        transactions, articles, customers, WINDOW_START, out2, chunk_size=2
    )

    df1 = pl.concat([pl.read_parquet(p) for p in sorted(out1.glob("part-*.parquet"))])
    df2 = pl.concat([pl.read_parquet(p) for p in sorted(out2.glob("part-*.parquet"))])

    # per-strategy rank columns are included: df.equals compares every column, so a narrowed comparison would be caught here
    for col in [f"rank_{s}" for s in candidates.STRATEGY_NAMES]:
        assert col in df1.columns

    # row order across concat and list-column element order are not guaranteed
    # stable by group_by/unique, independent of the pipeline's own determinism
    sort_cols = ["customer_id", "article_id"]
    df1 = df1.sort(sort_cols).with_columns(pl.col("strategies").list.sort())
    df2 = df2.sort(sort_cols).with_columns(pl.col("strategies").list.sort())

    assert df1.equals(df2)


def test_pool_candidates_rank_columns_null_where_strategy_absent(tmp_path):
    # rank_<strategy> is non-null exactly for the producing strategies; their count and minimum match n_strategies and best_rank
    pl.enable_string_cache()
    transactions, articles, customers = _synthetic_data()

    out = tmp_path / "run"
    candidates.generate_candidates_chunked(transactions, articles, customers, WINDOW_START, out, chunk_size=2)
    df = pl.concat([pl.read_parquet(p) for p in sorted(out.glob("part-*.parquet"))])

    rank_cols = [f"rank_{s}" for s in candidates.STRATEGY_NAMES]
    assert df.height > 0
    for row in df.to_dicts():
        non_null_ranks = [row[c] for c in rank_cols if row[c] is not None]
        assert len(non_null_ranks) == row["n_strategies"]
        assert min(non_null_ranks) == row["best_rank"]
        for strategy, col in zip(candidates.STRATEGY_NAMES, rank_cols):
            assert (row[col] is not None) == (strategy in row["strategies"])
