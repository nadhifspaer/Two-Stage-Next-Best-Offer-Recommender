import src.env_config  # noqa: F401
import json, sys
from pathlib import Path
import polars as pl

d = Path(sys.argv[1]); out = Path(sys.argv[2])
res = {}
for f in sorted(d.glob("*.parquet")):
    df = pl.read_parquet(f)
    cats = [c for c, t in df.schema.items() if t in (pl.Categorical, pl.Enum)]
    if cats:
        df = df.with_columns([pl.col(c).cast(pl.String) for c in cats])
    h = df.select(pl.struct(pl.all()).hash().alias("h"))["h"]
    res[f.name] = {
        "rows": df.height,
        "cols": df.columns,
        "set_hash": int(h.sum()),
        "ordered_hash": int(h.cum_sum().tail(1).item()) if False else int((h * (pl.Series(range(1, df.height + 1)).cast(pl.UInt64))).sum()) if df.height else 0,
    }
    del df, h
out.write_text(json.dumps(res))
print(len(res), "files", sum(v["rows"] for v in res.values()), "rows")
