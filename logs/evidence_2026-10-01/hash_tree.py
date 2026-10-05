# content-hash every file under a directory tree; parquet by content (order-independent and order-sensitive),
# everything else by sha256, small JSON stored inline. usage: python hash_tree.py <root> <out.json>
import src.env_config  # noqa: F401
import hashlib
import json
import sys
import time
from pathlib import Path

import polars as pl

root = Path(sys.argv[1])
out = Path(sys.argv[2])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parquet_hash(path: Path) -> dict:
    df = pl.read_parquet(path)
    casts = []
    for name, dtype in df.schema.items():
        if dtype in (pl.Categorical, pl.Enum):
            casts.append(pl.col(name).cast(pl.String))
        elif isinstance(dtype, pl.List):
            # list element order normalized, as in the candidate content checksums
            casts.append(pl.col(name).list.sort())
    if casts:
        df = df.with_columns(casts)
    h = df.select(pl.struct(pl.all()).hash().alias("h"))["h"]
    n = df.height
    ordered = int((h * pl.Series(range(1, n + 1)).cast(pl.UInt64)).sum()) if n else 0
    return {"rows": n, "cols": df.columns, "set_hash": int(h.sum()), "ordered_hash": ordered}


result = {}
t0 = time.time()
for path in sorted(p for p in root.rglob("*") if p.is_file()):
    rel = path.relative_to(root).as_posix()
    entry = {"bytes": path.stat().st_size}
    try:
        if path.suffix == ".parquet":
            entry.update(parquet_hash(path))
        else:
            entry["sha256"] = sha256(path)
            if path.suffix == ".json" and entry["bytes"] < 2_000_000:
                entry["json"] = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # recorded, not fatal
        entry["error"] = repr(exc)
    result[rel] = entry
out.write_text(json.dumps(result), encoding="utf-8")
errors = [k for k, v in result.items() if "error" in v]
print(f"{len(result)} files, {sum(v['bytes'] for v in result.values()) / 1e9:.2f} GB, {len(errors)} errors, {time.time() - t0:.0f}s")
for k in errors[:10]:
    print("ERROR", k, result[k]["error"])
