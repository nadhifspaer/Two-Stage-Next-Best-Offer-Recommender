# One-off: embed articles detail_desc, fit PCA, persist vectors and the fitted PCA under data/processed/.
import src.env_config  # noqa: F401 - must stay the first import: sets OPENBLAS_NUM_THREADS before any numeric library loads

import time
from pathlib import Path

import polars as pl

import src.text_embed as text_embed

DATA_DIR = Path("data/processed")


def run() -> dict:
    t0 = time.time()

    articles = pl.read_parquet(DATA_DIR / "articles.parquet")
    vectors, pca = text_embed.build_article_text_vectors(articles)
    text_embed.persist(vectors, pca)

    wall_time_s = time.time() - t0
    return {
        "articles": articles.height,
        "missing_detail_desc": int(vectors["detail_desc_missing"].sum()),
        "explained_variance_ratio_sum": round(float(pca.explained_variance_ratio_.sum()), 4),
        "wall_time_s": round(wall_time_s, 1),
    }


if __name__ == "__main__":
    result = run()
    print(result)
