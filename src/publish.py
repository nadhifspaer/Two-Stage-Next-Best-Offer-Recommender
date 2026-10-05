# clear a globbed output pattern before a stage republishes it
from pathlib import Path


def clear_globbed(directory: Path, pattern: str) -> int:
    """Deletes files matching pattern in directory (non-recursive); returns the count."""
    directory = Path(directory)
    if not directory.exists():
        return 0
    stale = list(directory.glob(pattern))
    for path in stale:
        path.unlink()
    return len(stale)


def assert_rows_match_manifest(
    directory: Path, pattern: str, manifest_path: Path, key: str = "rows_written"
) -> int:
    """Raises when the rows in the files matching pattern differ from the producing
    stage's manifest count; returns the row count. Reads parquet metadata only."""
    import json

    import polars as pl

    expected = json.loads(Path(manifest_path).read_text(encoding="utf-8"))[key]
    paths = sorted(Path(directory).glob(pattern))
    actual = sum(pl.scan_parquet(p).select(pl.len()).collect().item() for p in paths)
    if actual != expected:
        raise RuntimeError(
            f"{directory}/{pattern}: {len(paths)} files hold {actual} rows, "
            f"manifest {key}={expected}; stale or missing files"
        )
    return actual


def assert_top12_exact(directory: Path, n_customers: int, k: int = 12, pattern: str = "part-*.parquet") -> int:
    """Raises unless the files hold exactly k rows for each of n_customers customers (per-file counts == k, total == n_customers * k).
    Returns the row count."""
    import polars as pl

    paths = sorted(Path(directory).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no {pattern} files under {directory}")
    rows = 0
    customers = 0
    for path in paths:
        per_customer = pl.scan_parquet(path).select("customer_id").group_by("customer_id").len().collect()
        rows += int(per_customer["len"].sum())
        customers += per_customer.height
        if per_customer.height and (per_customer["len"].min() != k or per_customer["len"].max() != k):
            raise RuntimeError(f"{path}: a customer has a row count other than {k}")
    if rows != n_customers * k or customers != n_customers:
        raise RuntimeError(
            f"{directory}: {rows} rows for {customers} customers, expected {n_customers * k} "
            f"rows for {n_customers} customers"
        )
    return rows


def customer_count(customers_path: Path) -> int:
    import polars as pl

    return int(pl.scan_parquet(customers_path).select(pl.len()).collect().item())
