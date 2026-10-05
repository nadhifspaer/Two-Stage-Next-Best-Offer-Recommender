# compare two hash_tree.py outputs. usage: python compare_hash_trees.py <before.json> <after.json>
# parquet: order-independent content hash is the verdict; ordered-hash differences are listed separately as informational.
# JSON: structural diff ignoring timing, memory and MLflow run-id fields. other files: sha256.
# JSON floats: relative tolerance DEFAULT_REL_TOL; per-field overrides in FIELD_REL_TOL; integers, strings, hashes and row counts exact.
import json
import sys
from pathlib import Path

VOLATILE = {"wall_time_s", "peak_rss_gb", "peak_uss_gb", "parent_peak_rss_gb", "partition_peak_rss_gb",
            "mlflow_run_id", "ranker_run_id", "classifier_run_id", "run_id", "start_rss_gb", "end_rss_gb", "parent_phases"}

DEFAULT_REL_TOL = 1e-8
# float32-accumulated diagnostics, observed drift 7e-8 to 9e-8 between identical-content runs
FIELD_REL_TOL = {
    "raw_brier": 1e-6,
    "raw_brier_skill_score": 1e-6,
    "sum_prob": 1e-6,
    "db_file_size_bytes": 1e-3,  # SQLite page layout and freelist ordering
}
# reliability_table.observed_rate: one positive moving between adjacent bins (ordinal rank ties) changes a bin by 1/n
RELIABILITY_POSITIVE_SLACK = 1.01


def _close(key, x, y, n=None):
    if key == "observed_rate" and n:
        return abs(x - y) <= RELIABILITY_POSITIVE_SLACK / n
    tol = FIELD_REL_TOL.get(key, DEFAULT_REL_TOL)
    return abs(x - y) <= tol * max(abs(x), abs(y))


before = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
after = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))


def json_diff(a, b, path="", n=None):
    diffs = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in VOLATILE:
                continue
            if k not in a or k not in b:
                diffs.append(f"{path}/{k}: only in {'after' if k in b else 'before'}")
            else:
                diffs += json_diff(a[k], b[k], f"{path}/{k}", n=a.get("n") if "n" in a else n)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} -> {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diffs += json_diff(x, y, f"{path}[{i}]")
    elif (isinstance(a, float) and isinstance(b, float)) or (
        path.rsplit("/", 1)[-1] in FIELD_REL_TOL and isinstance(a, (int, float)) and isinstance(b, (int, float))
    ):
        if a != b and not _close(path.rsplit("/", 1)[-1], a, b, n):
            diffs.append(f"{path}: {a!r} -> {b!r}")
    elif a != b:
        diffs.append(f"{path}: {a!r} -> {b!r}")
    return diffs


only_before = sorted(set(before) - set(after))
only_after = sorted(set(after) - set(before))
content_changed, order_only, json_changed, bytes_changed, same = [], [], [], [], 0
for k in sorted(set(before) & set(after)):
    a, b = before[k], after[k]
    if "set_hash" in a:
        if a.get("rows") != b.get("rows") or a.get("cols") != b.get("cols") or a["set_hash"] != b.get("set_hash"):
            content_changed.append(k)
        elif a["ordered_hash"] != b["ordered_hash"]:
            order_only.append(k)
        else:
            same += 1
    elif "json" in a and "json" in b:
        d = json_diff(a["json"], b["json"])
        if d:
            json_changed.append((k, d))
        else:
            same += 1
    elif a.get("sha256") != b.get("sha256"):
        bytes_changed.append(k)
    else:
        same += 1

print(f"files before {len(before)}, after {len(after)}, identical {same}")
print(f"only in before: {len(only_before)}", only_before[:10])
print(f"only in after:  {len(only_after)}", only_after[:10])
print(f"parquet CONTENT changed: {len(content_changed)}")
for k in content_changed[:40]:
    print("  ", k, before[k].get("rows"), "->", after[k].get("rows"))
print(f"parquet row ORDER changed only (informational): {len(order_only)}")
print(f"JSON changed outside volatile fields: {len(json_changed)}")
for k, d in json_changed[:30]:
    print("  ", k)
    for line in d[:6]:
        print("      ", line)
print(f"non-parquet non-JSON bytes changed: {len(bytes_changed)}", bytes_changed[:20])
