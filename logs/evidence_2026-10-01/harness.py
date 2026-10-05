import time
t_proc = time.time()
import src.env_config  # noqa: F401
import json, os, sys
from pathlib import Path
import polars as pl
import pipelines.score_validation_ranker as m

t_imp = time.time() - t_proc
scratch = Path(os.environ["HARNESS_OUT"])
m.OUTPUT_DIR = scratch
res = {"openblas": os.environ.get("OPENBLAS_NUM_THREADS"), "import_s": round(t_imp, 2)}
t = time.time(); m._load_ranker_and_config(); res["model_load_s"] = round(time.time() - t, 2)
for part in sys.argv[1:]:
    t = time.time(); r = m.score_partition(part); r["total_s"] = round(time.time() - t, 2)
    res[part] = {k: r[k] for k in ("rows_scored", "wall_time_s", "total_s")}
print(json.dumps(res))
