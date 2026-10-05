# Pandera contract tests for the four raw tables
import datetime

import pandera.errors
import polars as pl
import polars.exceptions
import pytest

import src.dataset as dataset
from src.schema import (
    ArticlesSchema,
    CustomersSchema,
    SampleSubmissionSchema,
    TransactionsSchema,
)


def valid_transactions_df():
    return pl.DataFrame(
        {
            "t_dat": [datetime.date(2020, 9, 16)],
            "customer_id": ["a" * 64],
            "article_id": ["0108775015"],
            "price": [0.05],
            "sales_channel_id": [1],
        }
    )


def valid_articles_df():
    return pl.DataFrame(
        {
            "article_id": ["0108775015"],
            "product_code": ["0108775"],
            "prod_name": ["Strap top"],
            "product_type_no": [253],
            "product_type_name": ["Vest top"],
            "product_group_name": ["Garment Upper body"],
            "graphical_appearance_no": [1010016],
            "graphical_appearance_name": ["Solid"],
            "colour_group_code": ["09"],
            "colour_group_name": ["Black"],
            "perceived_colour_value_id": [4],
            "perceived_colour_value_name": ["Dark"],
            "perceived_colour_master_id": [5],
            "perceived_colour_master_name": ["Black"],
            "department_no": [1676],
            "department_name": ["Jersey Basic"],
            "index_code": ["A"],
            "index_name": ["Ladieswear"],
            "index_group_no": [1],
            "index_group_name": ["Ladieswear"],
            "section_no": [16],
            "section_name": ["Womens Everyday Basics"],
            "garment_group_no": [1002],
            "garment_group_name": ["Jersey Basic"],
            "detail_desc": ["Jersey top with narrow shoulder straps."],
        }
    )


def valid_customers_df():
    return pl.DataFrame(
        {
            "customer_id": ["a" * 64],
            "FN": [1.0],
            "Active": [1.0],
            "club_member_status": ["ACTIVE"],
            "fashion_news_frequency": ["NONE"],
            "age": [49],
            "postal_code": ["b" * 64],
        }
    )


def valid_sample_submission_df():
    return pl.DataFrame(
        {
            "customer_id": ["a" * 64],
            "prediction": ["0706016001 0706016002"],
        }
    )


def test_valid_transactions_frame_passes():
    TransactionsSchema.validate(valid_transactions_df())


def test_valid_articles_frame_passes():
    ArticlesSchema.validate(valid_articles_df())


def test_valid_customers_frame_passes():
    CustomersSchema.validate(valid_customers_df())


def test_valid_sample_submission_frame_passes():
    SampleSubmissionSchema.validate(valid_sample_submission_df())


def test_article_id_bare_int_fails_length_check():
    # simulates schema_overrides being bypassed: article_id arrives as an int and coerce=True casts it to a string
    # one character short once the leading zero is gone; the length check is what catches it
    df = valid_transactions_df().with_columns(pl.lit(108775015).alias("article_id"))
    with pytest.raises(pandera.errors.SchemaError):
        TransactionsSchema.validate(df)


def test_product_code_bare_int_fails_length_check():
    # simulates schema_overrides being bypassed: product_code arrives as an int and coerce=True casts it to a string
    # one character short once the leading zero is gone; the length check is what catches it
    df = valid_articles_df().with_columns(pl.lit(108775).alias("product_code"))
    with pytest.raises(pandera.errors.SchemaError):
        ArticlesSchema.validate(df)


def test_colour_group_code_bare_int_fails_length_check():
    # simulates schema_overrides being bypassed: colour_group_code arrives as an int and coerce=True casts it to a string
    # one character short once the leading zero is gone; the length check is what catches it
    df = valid_articles_df().with_columns(pl.lit(9).alias("colour_group_code"))
    with pytest.raises(pandera.errors.SchemaError):
        ArticlesSchema.validate(df)


def test_extra_column_rejected_under_strict():
    df = valid_transactions_df().with_columns(pl.lit("x").alias("extra_col"))
    with pytest.raises(pandera.errors.SchemaError):
        TransactionsSchema.validate(df)


def test_missing_column_raises_raw_polars_error_not_schema_error():
    # pandera-polars 0.33.1 behaviour, not intended design: a missing declared column raises a raw polars ColumnNotFoundError
    # from the coercion step, not a SchemaError, so `except SchemaError` alone does not cover it
    df = valid_transactions_df().drop("sales_channel_id")
    with pytest.raises(polars.exceptions.ColumnNotFoundError):
        TransactionsSchema.validate(df)


def test_convert_raw_to_parquet_cleans_up_on_partial_failure(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()

    valid_transactions_df().write_csv(raw_dir / "transactions_train.csv")
    valid_articles_df().write_csv(raw_dir / "articles.csv")
    # customer_id is 9 characters, not 64: customers.csv is deliberately invalid
    # so the third conversion in CONVERSIONS order fails
    pl.DataFrame(
        {
            "customer_id": ["too_short"],
            "FN": [1.0],
            "Active": [1.0],
            "club_member_status": ["ACTIVE"],
            "fashion_news_frequency": ["NONE"],
            "age": [49],
            "postal_code": ["b" * 64],
        }
    ).write_csv(raw_dir / "customers.csv")
    valid_sample_submission_df().write_csv(raw_dir / "sample_submission.csv")

    monkeypatch.setattr(dataset, "RAW_DIR", raw_dir)
    monkeypatch.setattr(dataset, "PROCESSED_DIR", processed_dir)

    with pytest.raises(pandera.errors.SchemaError):
        dataset.convert_raw_to_parquet()

    assert list(processed_dir.glob("*.parquet")) == []
