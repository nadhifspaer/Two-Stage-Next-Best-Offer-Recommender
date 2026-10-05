# Committed notebooks come from one clean run: no duplicate images, no stream output,
# no absolute paths or usernames, sequential execution counts, valid nbformat.
import getpass
import re
from pathlib import Path

import nbformat
import pytest

NOTEBOOKS = sorted(Path("notebooks").glob("*.ipynb"))

# drive-letter paths, Windows and POSIX home directories
_PATH_PATTERN = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]|[\\/]Users[\\/]|[\\/]home[\\/]|~[\\/]")


def _usernames() -> list[str]:
    names = {Path.home().name, getpass.getuser()}
    return sorted(n for n in names if len(n) >= 4)


def _text_of(output) -> list[str]:
    # text carried by an output; image payloads are excluded from the path scan
    if output.output_type == "stream":
        return [output.get("text", "")]
    if output.output_type in ("execute_result", "display_data"):
        data = output.get("data", {})
        return [v if isinstance(v, str) else "".join(v) for k, v in data.items() if not k.startswith("image/")]
    if output.output_type == "error":
        return [output.get("evalue", ""), "\n".join(output.get("traceback", []))]
    return []


def violations(nb, name: str, usernames: list[str] | None = None) -> list[str]:
    usernames = _usernames() if usernames is None else usernames
    found = []

    expected = 1
    for index, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            count = cell.get("execution_count")
            if count != expected:
                found.append(
                    f"{name} cell {index}: execution count {count!r}, expected {expected} "
                    "(counts must run 1..n with no nulls or gaps, proof of a single run)"
                )
            expected = (count if isinstance(count, int) else expected) + 1

            images = [
                (o.output_type, o["data"]["image/png"])
                for o in cell.outputs
                if o.output_type in ("execute_result", "display_data") and "image/png" in o.get("data", {})
            ]
            payloads = [p for _, p in images]
            if len(payloads) != len(set(payloads)):
                kinds = sorted({k for k, _ in images})
                found.append(f"{name} cell {index}: the same image is emitted twice ({', '.join(kinds)})")

            streams = [o.get("name") for o in cell.outputs if o.output_type == "stream"]
            if streams:
                found.append(f"{name} cell {index}: {len(streams)} stream output block(s) ({', '.join(sorted(set(streams)))})")

        texts = [cell.source] + [t for o in cell.get("outputs", []) for t in _text_of(o)]
        for text in texts:
            match = _PATH_PATTERN.search(text)
            if match:
                found.append(f"{name} cell {index}: absolute path or home directory near {text[max(0, match.start() - 10):match.end() + 25]!r}")
                break
            hit = next((u for u in usernames if re.search(re.escape(u), text, re.IGNORECASE)), None)
            if hit:
                found.append(f"{name} cell {index}: username appears in output or source")
                break
    return found


def _nb(*cells):
    nb = nbformat.v4.new_notebook()
    nb.cells = list(cells)
    return nb


def _code(source="x", count=1, outputs=()):
    cell = nbformat.v4.new_code_cell(source)
    cell.execution_count = count
    cell.outputs = list(outputs)
    return cell


def _png(kind, payload="AAAA"):
    if kind == "execute_result":
        return nbformat.v4.new_output("execute_result", data={"image/png": payload, "text/plain": "<Figure>"}, execution_count=1)
    return nbformat.v4.new_output("display_data", data={"image/png": payload, "text/plain": "<Figure>"})


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_committed_notebook_is_valid_and_clean(path):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    found = violations(nb, path.name)
    assert not found, "\n" + "\n".join(found)


def test_there_are_notebooks_to_check():
    assert len(NOTEBOOKS) >= 4


def test_rule_duplicate_image_trips():
    nb = _nb(_code(outputs=[_png("execute_result"), _png("display_data")]))
    assert any("emitted twice" in v and "cell 0" in v for v in violations(nb, "nb.ipynb", []))


def test_distinct_images_in_one_cell_pass():
    nb = _nb(_code(outputs=[_png("display_data", "AAAA"), _png("display_data", "BBBB")]))
    assert violations(nb, "nb.ipynb", []) == []


def test_rule_stream_output_trips_for_stdout_and_stderr():
    for name in ("stdout", "stderr"):
        nb = _nb(_code(outputs=[nbformat.v4.new_output("stream", name=name, text="hello")]))
        assert any("stream output" in v for v in violations(nb, "nb.ipynb", []))


def test_rule_absolute_paths_trip_in_source_and_outputs():
    for text in (r"C:\Users\someone\project", "C:/work/project", "/Users/someone/project", "/home/someone/project", "~/project"):
        assert any("absolute path" in v for v in violations(_nb(_code(source=f'open("{text}")')), "nb.ipynb", [])), text
        out = nbformat.v4.new_output("execute_result", data={"text/plain": text}, execution_count=1)
        assert any("absolute path" in v for v in violations(_nb(_code(outputs=[out])), "nb.ipynb", [])), text
    markdown = nbformat.v4.new_markdown_cell(r"see C:\Users\someone")
    assert any("cell 0" in v and "absolute path" in v for v in violations(_nb(markdown), "nb.ipynb", []))


def test_relative_paths_and_urls_pass():
    nb = _nb(_code(source='open("data/processed/x.parquet"); "https://example.com/a"'))
    assert violations(nb, "nb.ipynb", []) == []


def test_rule_username_trips():
    out = nbformat.v4.new_output("execute_result", data={"text/plain": "owner: Someone"}, execution_count=1)
    assert any("username" in v for v in violations(_nb(_code(outputs=[out])), "nb.ipynb", ["someone"]))


def test_rule_execution_counts_trip_on_null_gap_and_wrong_start():
    assert any("cell 0" in v for v in violations(_nb(_code(count=None)), "nb.ipynb", []))
    assert any("cell 1" in v and "expected 2" in v for v in violations(_nb(_code(count=1), _code(count=3)), "nb.ipynb", []))
    assert any("cell 0" in v and "expected 1" in v for v in violations(_nb(_code(count=2)), "nb.ipynb", []))
    assert violations(_nb(_code(count=1), nbformat.v4.new_markdown_cell("m"), _code(count=2)), "nb.ipynb", []) == []


def test_invalid_notebook_is_rejected_by_validation():
    nb = _nb(_code())
    nb.cells[0]["cell_type"] = "not-a-cell-type"
    with pytest.raises(nbformat.ValidationError):
        nbformat.validate(nb)
