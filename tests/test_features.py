# customer, article, pair, and provenance feature block tests for src/features.py
import datetime

import numpy as np
import polars as pl

import src.features as features

WINDOW_START = datetime.date(2020, 9, 16)


def _synthetic_data():
    # c1 has a pre-window purchase of a1 (2020-09-10) and an on-window purchase of a3 (2020-09-16, the window start);
    # the on-window row must never influence a feature
    transactions = pl.DataFrame(
        {
            "t_dat": [
                datetime.date(2020, 9, 10),
                datetime.date(2020, 9, 12),
                datetime.date(2020, 9, 16),
                datetime.date(2020, 9, 11),
                datetime.date(2020, 9, 9),
            ],
            "customer_id": ["c1", "c1", "c1", "c2", "c2"],
            "article_id": ["a1", "a2", "a3", "a1", "a1"],
            "price": [10.0, 20.0, 999.0, 5.0, 5.0],
            "sales_channel_id": [1, 2, 1, 1, 1],
        }
    ).with_columns(pl.col("customer_id", "article_id").cast(pl.Categorical))

    articles = pl.DataFrame(
        {
            "article_id": ["a1", "a2", "a3"],
            "product_code": ["p1", "p1", "p2"],
            "colour_group_name": ["red", "red", "blue"],
            "department_name": ["dept1", "dept1", "dept2"],
            "product_type_name": ["t1", "t1", "t2"],
            "index_group_name": ["ig1", "ig1", "ig2"],
            "garment_group_name": ["g1", "g1", "g2"],
        }
    ).with_columns(pl.col("article_id", "product_code").cast(pl.Categorical))

    customers = pl.DataFrame(
        {
            "customer_id": ["c1", "c2", "c3"],
            "age": [30, None, 40],
            "club_member_status": ["ACTIVE", "ACTIVE", None],
            "fashion_news_frequency": ["Regularly", "NONE", "NONE"],
        }
    ).with_columns(pl.col("customer_id").cast(pl.Categorical))

    return transactions, articles, customers


def test_customer_features_excludes_on_window_transaction():
    pl.enable_string_cache()
    transactions, _, customers = _synthetic_data()
    result = features.customer_features(transactions, customers, WINDOW_START)

    c1 = result.filter(pl.col("customer_id") == "c1").to_dicts()[0]
    # only a1 (09-10) and a2 (09-12) count; a3 on 09-16 is the label window start, excluded
    assert c1["txn_count_total"] == 2
    assert c1["distinct_article_count_total"] == 2
    assert c1["mean_purchase_price"] == 15.0
    assert c1["days_since_last_purchase"] == 4  # 09-12 to 09-16

    c3 = result.filter(pl.col("customer_id") == "c3").to_dicts()[0]
    assert c3["txn_count_total"] == 0
    assert c3["days_since_last_purchase"] is None
    assert c3["age_bucket"] == "35-44"


def test_article_features_excludes_on_window_transaction():
    pl.enable_string_cache()
    transactions, articles, _ = _synthetic_data()
    result = features.article_features(transactions, articles, WINDOW_START)

    a1 = result.filter(pl.col("article_id") == "a1").to_dicts()[0]
    # a1 sold to c1 (09-10) and c2 (09-11, 09-09); 3 purchases, 2 distinct buyers
    assert a1["distinct_buyers_total"] == 2
    assert a1["article_mean_price"] == (10.0 + 5.0 + 5.0) / 3

    a3 = result.filter(pl.col("article_id") == "a3").to_dicts()[0]
    # a3's only transaction is on the window start itself, so it has no pre-window history
    assert a3["distinct_buyers_total"] == 0
    assert a3["article_mean_price"] is None


def test_article_features_wow_trend_is_negative_without_underflow():
    # purchase_count_1w and purchase_count_prior_1w are unsigned: a declining article's difference must stay a small negative, not wrap
    pl.enable_string_cache()
    transactions = pl.DataFrame(
        {
            "t_dat": [
                datetime.date(2020, 9, 5),
                datetime.date(2020, 9, 6),
                datetime.date(2020, 9, 7),
                datetime.date(2020, 9, 13),
            ],
            "customer_id": ["c1", "c2", "c3", "c1"],
            "article_id": ["a1", "a1", "a1", "a1"],
            "price": [1.0] * 4,
            "sales_channel_id": [1] * 4,
        }
    ).with_columns(pl.col("customer_id", "article_id").cast(pl.Categorical))
    articles = pl.DataFrame(
        {
            "article_id": ["a1"],
            "product_code": ["p1"],
            "colour_group_name": ["red"],
            "department_name": ["dept1"],
            "product_type_name": ["t1"],
            "index_group_name": ["ig1"],
            "garment_group_name": ["g1"],
        }
    ).with_columns(pl.col("article_id", "product_code").cast(pl.Categorical))

    result = features.article_features(transactions, articles, WINDOW_START)
    a1 = result.filter(pl.col("article_id") == "a1").to_dicts()[0]
    # 3 purchases in the prior week (09-05 to 09-07), 1 in the last week (09-13)
    assert a1["purchase_count_wow_trend"] == -2


def test_customer_article_pair_features_same_product_code_and_price_ratio():
    pl.enable_string_cache()
    transactions, articles, customers = _synthetic_data()
    customer_block = features.customer_features(transactions, customers, WINDOW_START)
    article_block = features.article_features(transactions, articles, WINDOW_START)

    text_vectors = pl.DataFrame(
        {"article_id": ["a1", "a2", "a3"], **{f"text_vec_{i}": [0.0] * 3 for i in range(32)}}
    ).with_columns(pl.col("article_id").cast(pl.Categorical))
    recent_text_block = features.customer_recent_text_mean(transactions, text_vectors, WINDOW_START)

    candidates = pl.DataFrame(
        {
            "customer_id": ["c1"],
            "article_id": ["a2"],
            "strategies": [["repurchase"]],
            "n_strategies": [1],
            "best_rank": [1],
        }
    ).with_columns(pl.col("customer_id", "article_id").cast(pl.Categorical))

    history = transactions.filter(pl.col("t_dat") < WINDOW_START).select(
        "customer_id", "article_id", "t_dat"
    )

    pair = features.customer_article_pair_features(
        history,
        articles,
        candidates,
        customer_block,
        article_block,
        text_vectors,
        recent_text_block,
        WINDOW_START,
    )
    row = pair.to_dicts()[0]
    # c1 never bought a2's exact id before the window in this candidate row's
    # own history slice besides the one purchase on 09-12
    assert row["prior_purchase_count_this_article"] == 1
    assert row["days_since_last_purchase_this_article"] == 4
    # a1 and a2 share product_code p1, both purchased by c1 pre-window
    assert row["prior_purchase_count_same_product_code"] == 2
    assert row["prior_purchase_count_same_colour"] == 2
    assert row["prior_purchase_count_same_department"] == 2
    # article_mean_price(a2) / customer_mean_purchase_price(c1) = 20 / 15
    assert abs(row["price_ratio_to_customer_mean"] - (20.0 / 15.0)) < 1e-9


def _content_checksum(df: pl.DataFrame) -> int:
    # order-independent content hash: struct(all columns).hash().sum()
    return int(df.select(pl.struct(pl.all()).hash().sum()).item())


def _large_synthetic_transactions(n_customers: int = 2_000, rows_per_customer: int = 4) -> pl.DataFrame:
    # generated data, sized well above the roughly 50-row floor that reproduced the original defect
    rng = np.random.default_rng(42)
    n = n_customers * rows_per_customer
    customer_idx = np.repeat(np.arange(n_customers), rows_per_customer)
    article_idx = rng.integers(0, 500, size=n)
    days_ago = rng.integers(1, 365, size=n)
    base = datetime.date(2020, 9, 16)
    return pl.DataFrame(
        {
            "t_dat": [base - datetime.timedelta(days=int(d)) for d in days_ago],
            "customer_id": [f"c{i}" for i in customer_idx],
            "article_id": [f"a{i}" for i in article_idx],
            "price": rng.uniform(1.0, 500.0, size=n),
            "sales_channel_id": rng.integers(1, 3, size=n),
        }
    ).with_columns(pl.col("customer_id", "article_id").cast(pl.Categorical))


def test_customer_features_price_aggregation_is_deterministic_across_two_runs():
    # regression guard: mean/std and share_sales_channel_* are float reductions that varied run to run under engine="streaming";
    # fixed by dropping it
    pl.enable_string_cache()
    transactions = _large_synthetic_transactions()
    customers = transactions.select("customer_id").unique().with_columns(
        age=pl.lit(30), club_member_status=pl.lit("ACTIVE"), fashion_news_frequency=pl.lit("NONE")
    )
    r1 = features.customer_features(transactions, customers, WINDOW_START)
    r2 = features.customer_features(transactions, customers, WINDOW_START)
    assert _content_checksum(r1.sort("customer_id")) == _content_checksum(r2.sort("customer_id"))


def test_article_features_price_aggregation_is_deterministic_across_two_runs():
    # regression guard, same defect class as customer_features, in article_mean_price
    pl.enable_string_cache()
    transactions = _large_synthetic_transactions()
    articles = transactions.select("article_id").unique().with_columns(
        product_code=pl.col("article_id"),
        colour_group_name=pl.lit("red"),
        department_name=pl.lit("dept1"),
        product_type_name=pl.lit("t1"),
        index_group_name=pl.lit("ig1"),
        garment_group_name=pl.lit("g1"),
    ).with_columns(pl.col("product_code").cast(pl.Categorical))
    r1 = features.article_features(transactions, articles, WINDOW_START)
    r2 = features.article_features(transactions, articles, WINDOW_START)
    assert _content_checksum(r1.sort("article_id")) == _content_checksum(r2.sort("article_id"))


def test_customer_recent_text_mean_is_deterministic_across_two_runs():
    # regression guard, same defect class: the 32 recent-text-vector columns are float means feeding text_cosine_similarity
    pl.enable_string_cache()
    transactions = _large_synthetic_transactions()
    article_ids = transactions.select("article_id").unique()["article_id"].to_list()
    rng = np.random.default_rng(43)
    text_vectors = pl.DataFrame(
        {
            "article_id": article_ids,
            **{f"text_vec_{i}": rng.uniform(-1.0, 1.0, size=len(article_ids)) for i in range(32)},
        }
    ).with_columns(pl.col("article_id").cast(pl.Categorical))
    r1 = features.customer_recent_text_mean(transactions, text_vectors, WINDOW_START)
    r2 = features.customer_recent_text_mean(transactions, text_vectors, WINDOW_START)
    assert _content_checksum(r1.sort("customer_id")) == _content_checksum(r2.sort("customer_id"))


def test_customer_article_pair_features_content_checksum_invariant_to_batch_size():
    # batch_size smaller than the candidate rows forces multiple batches in _pair_features_batched;
    # content must not depend on where the batch boundaries fall
    pl.enable_string_cache()
    transactions, articles, customers = _synthetic_data()
    customer_block = features.customer_features(transactions, customers, WINDOW_START)
    article_block = features.article_features(transactions, articles, WINDOW_START)

    text_vectors = pl.DataFrame(
        {"article_id": ["a1", "a2", "a3"], **{f"text_vec_{i}": [0.0] * 3 for i in range(32)}}
    ).with_columns(pl.col("article_id").cast(pl.Categorical))
    recent_text_block = features.customer_recent_text_mean(transactions, text_vectors, WINDOW_START)

    candidates = pl.DataFrame(
        {
            "customer_id": ["c1", "c1", "c2", "c2", "c1"],
            "article_id": ["a1", "a2", "a1", "a2", "a3"],
            "strategies": [["repurchase"]] * 5,
            "n_strategies": [1] * 5,
            "best_rank": [1] * 5,
        }
    ).with_columns(pl.col("customer_id", "article_id").cast(pl.Categorical))

    history = transactions.filter(pl.col("t_dat") < WINDOW_START).select(
        "customer_id", "article_id", "t_dat"
    )

    result_small_batch = features.customer_article_pair_features(
        history, articles, candidates, customer_block, article_block,
        text_vectors, recent_text_block, WINDOW_START, batch_size=1,
    )
    result_large_batch = features.customer_article_pair_features(
        history, articles, candidates, customer_block, article_block,
        text_vectors, recent_text_block, WINDOW_START, batch_size=1000,
    )

    assert result_small_batch.height == result_large_batch.height == 5
    sort_cols = ["customer_id", "article_id"]
    assert _content_checksum(result_small_batch.sort(sort_cols)) == _content_checksum(
        result_large_batch.sort(sort_cols)
    )


def test_assemble_partition_features_content_checksum_invariant_to_batch_size():
    pl.enable_string_cache()
    pair, provenance, customer_block, article_block, n = _synthetic_assembly_data()

    result_small_batch = features.assemble_partition_features(
        pair, provenance, customer_block, article_block, batch_size=1
    )
    result_large_batch = features.assemble_partition_features(
        pair, provenance, customer_block, article_block, batch_size=1000
    )

    assert result_small_batch.height == result_large_batch.height == n
    assert _content_checksum(result_small_batch.sort("customer_id")) == _content_checksum(
        result_large_batch.sort("customer_id")
    )


def test_candidate_provenance_features_flags():
    pl.enable_string_cache()
    candidates = pl.DataFrame(
        {
            "customer_id": ["c1", "c2"],
            "article_id": ["a1", "a2"],
            "strategies": [["repurchase", "als"], ["recent_popularity"]],
            "n_strategies": [2, 1],
            "best_rank": [1, 3],
            "rank_repurchase": [1, None],
            "rank_colour_variant": [None, None],
            "rank_recent_popularity": [None, 3],
            "rank_segment_popularity": [None, None],
            "rank_als": [4, None],
        }
    ).with_columns(pl.col("customer_id", "article_id").cast(pl.Categorical))

    result = features.candidate_provenance_features(candidates)
    row1 = result.filter(pl.col("customer_id") == "c1").to_dicts()[0]
    assert row1["from_repurchase"] is True
    assert row1["from_als"] is True
    assert row1["from_colour_variant"] is False
    assert row1["n_strategies"] == 2
    assert row1["best_rank"] == 1
    assert row1["rank_repurchase"] == 1
    assert row1["rank_als"] == 4
    assert row1["rank_colour_variant"] is None

    row2 = result.filter(pl.col("customer_id") == "c2").to_dicts()[0]
    assert row2["rank_recent_popularity"] == 3
    assert row2["rank_repurchase"] is None


def _synthetic_assembly_data():
    n = 7
    pair = pl.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(n)],
            "article_id": ["a1"] * n,
            "text_cosine_similarity": [0.1 * i for i in range(n)],
        }
    ).with_columns(pl.col("customer_id").cast(pl.Categorical), pl.col("article_id").cast(pl.Categorical))
    provenance = pl.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(n)],
            "article_id": ["a1"] * n,
            "n_strategies": list(range(n)),
        }
    ).with_columns(pl.col("customer_id").cast(pl.Categorical), pl.col("article_id").cast(pl.Categorical))
    customer_block = pl.DataFrame(
        {"customer_id": [f"c{i}" for i in range(n)], "age_bucket": ["unknown"] * n}
    ).with_columns(pl.col("customer_id").cast(pl.Categorical))
    article_block = pl.DataFrame({"article_id": ["a1"], "department_name": ["dept1"]}).with_columns(
        pl.col("article_id").cast(pl.Categorical)
    )
    return pair, provenance, customer_block, article_block, n


def test_assemble_partition_features_batching_preserves_all_rows():
    # batch_size smaller than the data forces multiple batches; every (customer_id, article_id) pair must still appear
    # exactly once, matched to its own provenance and customer/article block row
    pl.enable_string_cache()
    pair, provenance, customer_block, article_block, n = _synthetic_assembly_data()

    result = features.assemble_partition_features(
        pair, provenance, customer_block, article_block, batch_size=3
    )
    result = result.sort("customer_id")
    assert result.height == n
    for i, row in enumerate(result.to_dicts()):
        assert row["customer_id"] == f"c{i}"
        assert row["n_strategies"] == i
        assert abs(row["text_cosine_similarity"] - 0.1 * i) < 1e-9


def test_assemble_partition_features_write_dir_produces_readable_files(tmp_path):
    pl.enable_string_cache()
    pair, provenance, customer_block, article_block, n = _synthetic_assembly_data()

    rows_written = features.assemble_partition_features(
        pair,
        provenance,
        customer_block,
        article_block,
        batch_size=3,
        write_dir=tmp_path,
        file_stem="part-0000",
    )
    assert rows_written == n

    written_files = sorted(tmp_path.glob("part-0000-batch-*.parquet"))
    assert len(written_files) == 3  # 7 rows at batch_size=3 -> 3, 3, 1

    result = pl.concat([pl.read_parquet(p) for p in written_files]).sort("customer_id")
    assert result.height == n
    for i, row in enumerate(result.to_dicts()):
        assert row["customer_id"] == f"c{i}"
        assert row["n_strategies"] == i


def _synthetic_text_vectors(transactions: pl.DataFrame) -> pl.DataFrame:
    article_ids = transactions.select("article_id").unique()["article_id"].to_list()
    rng = np.random.default_rng(43)
    return pl.DataFrame(
        {
            "article_id": article_ids,
            **{f"text_vec_{i}": rng.uniform(-1.0, 1.0, size=len(article_ids)) for i in range(32)},
        }
    ).with_columns(pl.col("article_id").cast(pl.Categorical))


def _synthetic_customers(transactions: pl.DataFrame) -> pl.DataFrame:
    return transactions.select("customer_id").unique().with_columns(
        age=pl.lit(30), club_member_status=pl.lit("ACTIVE"), fashion_news_frequency=pl.lit("NONE")
    )


def test_customer_features_chunked_is_deterministic_across_two_runs():
    # chunk_size 300 over 2,000 customers forces 7 passes
    pl.enable_string_cache()
    transactions = _large_synthetic_transactions()
    customers = _synthetic_customers(transactions)
    r1 = features.customer_features(transactions, customers, WINDOW_START, chunk_size=300)
    r2 = features.customer_features(transactions, customers, WINDOW_START, chunk_size=300)
    assert _content_checksum(r1.sort("customer_id")) == _content_checksum(r2.sort("customer_id"))


def test_customer_features_chunked_equals_unchunked():
    pl.enable_string_cache()
    transactions = _large_synthetic_transactions()
    customers = _synthetic_customers(transactions)
    chunked = features.customer_features(transactions, customers, WINDOW_START, chunk_size=300)
    whole = features.customer_features(transactions, customers, WINDOW_START, chunk_size=10_000_000)
    assert chunked.height == whole.height == customers.height
    assert chunked.columns == whole.columns
    assert chunked.sort("customer_id").equals(whole.sort("customer_id"))


def test_customer_recent_text_mean_chunked_is_deterministic_across_two_runs():
    pl.enable_string_cache()
    transactions = _large_synthetic_transactions()
    text_vectors = _synthetic_text_vectors(transactions)
    r1 = features.customer_recent_text_mean(transactions, text_vectors, WINDOW_START, chunk_size=300)
    r2 = features.customer_recent_text_mean(transactions, text_vectors, WINDOW_START, chunk_size=300)
    assert _content_checksum(r1) == _content_checksum(r2)


def test_customer_recent_text_mean_chunked_equals_unchunked():
    pl.enable_string_cache()
    transactions = _large_synthetic_transactions()
    text_vectors = _synthetic_text_vectors(transactions)
    chunked = features.customer_recent_text_mean(transactions, text_vectors, WINDOW_START, chunk_size=300)
    whole = features.customer_recent_text_mean(
        transactions, text_vectors, WINDOW_START, chunk_size=10_000_000
    )
    assert chunked.height == whole.height
    assert chunked.equals(whole)
