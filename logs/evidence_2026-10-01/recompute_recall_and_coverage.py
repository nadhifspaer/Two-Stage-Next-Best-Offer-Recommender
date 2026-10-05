# recomputes Recall@100 (overall, per strategy, sole-contribution marginal), candidates per customer, and the popularity-only count
# from data/processed/candidates/validation and transactions_train.parquet
import duckdb
c = duckdb.connect(); D = "data/processed/"
c.execute(f"create view tx as select * from read_parquet('{D}transactions_train.parquet')")
c.execute(f"create view cand as select customer_id, article_id, strategies from read_parquet('{D}candidates/validation/part-*.parquet')")
c.execute("create table truth as select distinct customer_id, article_id from tx where t_dat >= DATE '2020-09-16' and t_dat <= DATE '2020-09-22'")
tp = c.execute("select count(*) from truth").fetchone()[0]
c.execute("create table hit as select t.customer_id, t.article_id, cd.strategies from truth t join cand cd using (customer_id, article_id)")
h = c.execute("select count(*) from hit").fetchone()[0]
print("true pairs", tp, "hits", h, "pooled recall", round(h / tp, 4))
for s in ["repurchase", "colour_variant", "recent_popularity", "segment_popularity", "als"]:
    a = c.execute(f"select count(*) from hit where list_contains(strategies,'{s}')").fetchone()[0]
    b = c.execute(f"select count(*) from hit where len(list_distinct(strategies))=1 and list_contains(strategies,'{s}')").fetchone()[0]
    print(s, "recall", round(a / tp, 4), "marginal(sole)", round(b / tp, 4))
print("candidates per customer min/mean/max, customers:", c.execute(f"select min(n), avg(n), max(n), count(*) from (select customer_id, count(*) n from read_parquet('{D}candidates/validation/part-*.parquet') group by 1)").fetchone())
print("popularity-only customers:", c.execute(f"select count(*) from (select customer_id, bool_or(list_has_any(strategies, ['repurchase','colour_variant','als'])) p from read_parquet('{D}candidates/validation/part-*.parquet') group by 1) where not p").fetchone()[0])
print("partition files:", len(__import__('glob').glob(D + "candidates/validation/part-*.parquet")))
