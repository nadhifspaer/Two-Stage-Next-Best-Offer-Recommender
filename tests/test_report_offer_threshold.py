# trailing price mean determinism for pipelines/report_offer_threshold_analysis.py
from datetime import date

import numpy as np
import polars as pl

import pipelines.report_offer_threshold_analysis as report


def _content_checksum(df: pl.DataFrame) -> int:
    df = df.with_columns(pl.col("article_id").cast(pl.Utf8)).sort("article_id")
    return int(df.select(pl.struct(pl.all()).hash().sum()).item())


def test_trailing_article_price_is_deterministic_across_two_runs():
    # regression guard: a float mean under engine="streaming" combines partial sums in a varying order;
    # generated data sized well above the 50-row floor that reproduced it
    rng = np.random.default_rng(7)
    n_rows = 400_000
    start = date(2020, 8, 19)
    transactions = pl.DataFrame(
        {
            "article_id": [f"a{i}" for i in rng.integers(0, 4_000, size=n_rows)],
            "price": rng.random(n_rows) * 0.1,
            "t_dat": [start] * n_rows,
        }
    ).with_columns(pl.col("article_id").cast(pl.Categorical))
    window_start = date(2020, 9, 16)
    checksums = {
        _content_checksum(report.trailing_article_price(transactions.lazy(), start, window_start)) for _ in range(2)
    }
    assert len(checksums) == 1
