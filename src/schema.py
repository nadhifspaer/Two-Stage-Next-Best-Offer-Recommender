# Pandera contracts for the four raw tables
import pandera.polars as pa
import polars as pl

TransactionsSchema = pa.DataFrameSchema(
    {
        "t_dat": pa.Column(pl.Date),
        "customer_id": pa.Column(pl.Utf8, pa.Check.str_length(64, 64)),
        "article_id": pa.Column(pl.Utf8, pa.Check.str_length(10, 10)),
        "price": pa.Column(pl.Float64, pa.Check.ge(0)),
        "sales_channel_id": pa.Column(pl.Int64, pa.Check.isin([1, 2])),
    },
    strict=True,
    coerce=True,
)

ArticlesSchema = pa.DataFrameSchema(
    {
        "article_id": pa.Column(pl.Utf8, pa.Check.str_length(10, 10), unique=True),
        "product_code": pa.Column(pl.Utf8, pa.Check.str_length(7, 7)),
        "prod_name": pa.Column(pl.Utf8),
        "product_type_no": pa.Column(pl.Int64),
        "product_type_name": pa.Column(pl.Utf8),
        "product_group_name": pa.Column(pl.Utf8),
        "graphical_appearance_no": pa.Column(pl.Int64),
        "graphical_appearance_name": pa.Column(pl.Utf8),
        "colour_group_code": pa.Column(pl.Utf8, pa.Check.str_length(2, 2)),
        "colour_group_name": pa.Column(pl.Utf8),
        "perceived_colour_value_id": pa.Column(pl.Int64),
        "perceived_colour_value_name": pa.Column(pl.Utf8),
        "perceived_colour_master_id": pa.Column(pl.Int64),
        "perceived_colour_master_name": pa.Column(pl.Utf8),
        "department_no": pa.Column(pl.Int64),
        "department_name": pa.Column(pl.Utf8),
        "index_code": pa.Column(pl.Utf8),
        "index_name": pa.Column(pl.Utf8),
        "index_group_no": pa.Column(pl.Int64),
        "index_group_name": pa.Column(pl.Utf8),
        "section_no": pa.Column(pl.Int64),
        "section_name": pa.Column(pl.Utf8),
        "garment_group_no": pa.Column(pl.Int64),
        "garment_group_name": pa.Column(pl.Utf8),
        "detail_desc": pa.Column(pl.Utf8, nullable=True),
    },
    strict=True,
    coerce=True,
)

CustomersSchema = pa.DataFrameSchema(
    {
        "customer_id": pa.Column(pl.Utf8, pa.Check.str_length(64, 64), unique=True),
        "FN": pa.Column(pl.Float64, nullable=True),
        "Active": pa.Column(pl.Float64, nullable=True),
        "club_member_status": pa.Column(pl.Utf8, nullable=True),
        "fashion_news_frequency": pa.Column(pl.Utf8, nullable=True),
        "age": pa.Column(pl.Int64, nullable=True),
        "postal_code": pa.Column(pl.Utf8, pa.Check.str_length(64, 64)),
    },
    strict=True,
    coerce=True,
)

SampleSubmissionSchema = pa.DataFrameSchema(
    {
        "customer_id": pa.Column(pl.Utf8, pa.Check.str_length(64, 64), unique=True),
        "prediction": pa.Column(pl.Utf8),
    },
    strict=True,
    coerce=True,
)
