# Must be the first import in every entry point: OPENBLAS_NUM_THREADS is read when the library loads, not at call time.
# Unset, OpenBLAS threads contend with Polars during ALS and candidate generation slows by more than an order of magnitude.
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
