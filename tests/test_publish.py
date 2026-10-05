import src.publish as publish


def test_clear_globbed_removes_only_matching_files(tmp_path):
    (tmp_path / "part-0000-batch-0000.parquet").write_bytes(b"x")
    (tmp_path / "part-0000-batch-0007.parquet").write_bytes(b"x")
    (tmp_path / "_manifest.json").write_text("{}")
    removed = publish.clear_globbed(tmp_path, "part-*-batch-*.parquet")
    assert removed == 2
    assert [p.name for p in tmp_path.iterdir()] == ["_manifest.json"]


def test_clear_globbed_missing_directory_is_a_noop(tmp_path):
    assert publish.clear_globbed(tmp_path / "absent", "part-*.parquet") == 0


def test_assert_rows_match_manifest_flags_stale_file(tmp_path):
    import json

    import polars as pl
    import pytest

    pl.DataFrame({"a": [1, 2, 3]}).write_parquet(tmp_path / "part-0000-batch-0000.parquet")
    (tmp_path / "_manifest.json").write_text(json.dumps({"rows_written": 3}))
    assert (
        publish.assert_rows_match_manifest(tmp_path, "part-*-batch-*.parquet", tmp_path / "_manifest.json") == 3
    )
    pl.DataFrame({"a": [4, 5]}).write_parquet(tmp_path / "part-0000-batch-0007.parquet")
    with pytest.raises(RuntimeError, match="stale or missing"):
        publish.assert_rows_match_manifest(tmp_path, "part-*-batch-*.parquet", tmp_path / "_manifest.json")
