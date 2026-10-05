import json, glob, os, sqlite3
import duckdb
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
c = duckdb.connect()
q = lambda s: c.execute(s).fetchall()
D = "data/processed/"
print("transactions rows, min, max dates:", q(f"select count(*), min(t_dat), max(t_dat) from read_parquet('{D}transactions_train.parquet')"))
print("customers, articles, sample_sub:", q(f"select count(*) from read_parquet('{D}customers.parquet')"), q(f"select count(*) from read_parquet('{D}articles.parquet')"), q(f"select count(*) from read_parquet('{D}sample_submission.parquet')"))
print("pre-window rows:", q(f"select count(*) from read_parquet('{D}transactions_train.parquet') where t_dat < DATE '2020-09-16'"))
print("validation purchasers:", q(f"select count(distinct customer_id) from read_parquet('{D}transactions_train.parquet') where t_dat >= DATE '2020-09-16'"))
print("customers last purchased > 52w before 2020-09-22:", q(f"""select count(*) from (select customer_id, max(t_dat) m from read_parquet('{D}transactions_train.parquet') group by 1) where m < DATE '2020-09-22' - INTERVAL 364 DAY"""))
cp = D + "candidates/validation/part-*.parquet"
print("cand per customer min/mean/max, customers:", q(f"select min(n), avg(n), max(n), count(*) from (select customer_id, count(*) n from read_parquet('{cp}') group by 1)"))
print("popularity-only:", q(f"""select count(*) from (select customer_id, bool_or(list_has_any(strategies, ['repurchase','colour_variant','als'])) p from read_parquet('{cp}') group by 1) where not p"""))
off = q(f"select count(*) from read_parquet('{D}nbo/offers_2020-09-16.parquet')")
print("offers rows:", off)
con = sqlite3.connect("file:" + D + "nbo/offer_store.db?mode=ro", uri=True)
print("offer_store tables:", con.execute("select name from sqlite_master where type='table'").fetchall())
print("offer_store served_top12 rows, customers:", con.execute("select count(*), count(distinct customer_id) from served_top12").fetchone(), os.path.getsize(D + "nbo/offer_store.db"))
print("feature_columns.json:", len(json.load(open(D + "models/feature_columns.json"))))
m = json.load(open(D + "evaluation/_backtest_map12_manifest.json"))
print({k: v for k, v in m.items() if k in ("orderings", "ranker_ratio_vs", "wall_time_s", "peak_rss_gb")})
for f in ["evaluation/_backtest_train_manifest.json", "features/backtest_2020-09-09/_manifest.json", "evaluation/_backtest_ranker_score_manifest.json"]:
    d = json.load(open(D + f)); print(f, {k: v for k, v in d.items() if not isinstance(v, (list, dict))})
cs = json.load(open(D + "calibration/_calibration_manifest.json")); print({k: v for k, v in cs.items() if k != "reliability_table"})
nb = json.load(open(D + "nbo/_offer_store_manifest.json")); print(nb)
