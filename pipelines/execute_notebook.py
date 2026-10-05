# Executes a notebook through a real kernel from the repository root and writes it back in place.
# Usage: python -m pipelines.execute_notebook notebooks/02_candidates.ipynb
import src.env_config  # noqa: F401 - must stay the first import: sets OPENBLAS_NUM_THREADS before any numeric library loads

import os
import sys
import time
from pathlib import Path

# keep MLflow hints, download progress bars and the tqdm widget warning out of cell output
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
os.environ.setdefault("PYTHONWARNINGS", "ignore:IProgress not found:Warning")

import nbformat  # noqa: E402
from nbclient import NotebookClient  # noqa: E402


def run(path: Path, timeout_s: int = 1800) -> dict:
    path = Path(path)
    nb = nbformat.read(path, as_version=4)
    t0 = time.time()
    NotebookClient(nb, timeout=timeout_s, kernel_name="python3", resources={"metadata": {"path": "."}}).execute()
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
    if errors:
        raise RuntimeError(f"{path}: {len(errors)} cell(s) raised")
    nbformat.validate(nb)
    tmp = path.with_suffix(".ipynb.tmp")
    nbformat.write(nb, tmp)
    os.replace(tmp, path)
    return {"notebook": str(path), "wall_time_s": round(time.time() - t0, 1)}


if __name__ == "__main__":
    print(run(Path(sys.argv[1])))
