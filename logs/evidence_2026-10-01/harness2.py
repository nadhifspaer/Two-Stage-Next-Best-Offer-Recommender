import time
t_proc = time.time()
import src.env_config  # noqa: F401
import json, os, sys
from pathlib import Path
import pipelines.score_nbo_probabilities as m
t_imp = time.time() - t_proc
m.OUTPUT_DIR = Path(os.environ["HARNESS_OUT"])
res = {"openblas": os.environ.get("OPENBLAS_NUM_THREADS"), "import_s": round(t_imp, 2)}
for part in sys.argv[1:]:
    t = time.time(); r = m.score_partition(part); res[part] = {"rows": r["rows"], "fresh": r["freshly_scored_rows"], "wall_time_s": r["wall_time_s"], "total_s": round(time.time() - t, 2)}
print(json.dumps(res))
