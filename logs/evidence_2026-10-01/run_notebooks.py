import json, sys, time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path.cwd()
nb_path = root / "notebooks" / sys.argv[1]
if len(sys.argv) > 2 and sys.argv[2] == "--patch03":
    nb = nbformat.read(nb_path, as_version=4)
    src = nb.cells[1].source
    start = src.index("ranker_run_id = None")
    end = src.index("ranker = mlflow.lightgbm.load_model")
    src = src[:start] + "production = production_models.load_production_models()\n\n" + src[end:]
    src = src.replace('f"runs:/{ranker_run_id}/ranker"', 'f"runs:/{production.ranker_run_id}/ranker"').replace('f"runs:/{classifier_run_id}/classifier"', 'f"runs:/{production.classifier_run_id}/classifier"')
    src = src.replace("import src.rank as rank\n", "import src.production_models as production_models\nimport src.rank as rank\n")
    nb.cells[1].source = src
    nbformat.write(nb, nb_path)
    print("patched", nb_path.name)
nb = nbformat.read(nb_path, as_version=4)
t0 = time.time()
NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(root)}}).execute()
nbformat.write(nb, nb_path)
print(nb_path.name, "executed in", round(time.time() - t0, 1), "s")
