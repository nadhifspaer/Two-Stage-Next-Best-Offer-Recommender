# detail_desc embedding and PCA reduction
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

MODEL_NAME = "all-MiniLM-L6-v2"
RAW_DIM = 384
REDUCED_DIM = 32
PCA_RANDOM_STATE = 42
ENCODE_BATCH_SIZE = 256

PROCESSED_DIR = Path("data/processed")
VECTORS_PATH = PROCESSED_DIR / "article_text_vectors.parquet"
PCA_PATH = PROCESSED_DIR / "text_pca.joblib"

VECTOR_COLUMNS = [f"text_vec_{i}" for i in range(REDUCED_DIM)]


def encode_detail_desc(
    articles: pl.DataFrame, model: SentenceTransformer, batch_size: int = ENCODE_BATCH_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    # null or empty detail_desc never reaches the model; explicit zero vector instead
    texts = articles["detail_desc"].to_list()
    is_missing = np.array([t is None or t == "" for t in texts])

    embeddings = np.zeros((len(texts), RAW_DIM), dtype=np.float32)
    present_idx = np.where(~is_missing)[0]
    if len(present_idx) > 0:
        present_texts = [texts[i] for i in present_idx]
        present_embeddings = model.encode(
            present_texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
        )
        embeddings[present_idx] = present_embeddings.astype(np.float32)

    return embeddings, is_missing


def fit_pca(
    embeddings: np.ndarray,
    is_missing: np.ndarray,
    n_components: int = REDUCED_DIM,
    random_state: int = PCA_RANDOM_STATE,
) -> PCA:
    # fit on real embeddings only; zero-vector placeholders carry no semantic signal
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(embeddings[~is_missing])
    return pca


def reduce_embeddings(embeddings: np.ndarray, is_missing: np.ndarray, pca: PCA) -> np.ndarray:
    reduced = np.zeros((embeddings.shape[0], pca.n_components_), dtype=np.float32)
    present_idx = np.where(~is_missing)[0]
    if len(present_idx) > 0:
        reduced[present_idx] = pca.transform(embeddings[present_idx]).astype(np.float32)
    return reduced


def build_article_text_vectors(
    articles: pl.DataFrame,
    model_name: str = MODEL_NAME,
    n_components: int = REDUCED_DIM,
    random_state: int = PCA_RANDOM_STATE,
) -> tuple[pl.DataFrame, PCA]:
    model = SentenceTransformer(model_name)
    embeddings, is_missing = encode_detail_desc(articles, model)
    pca = fit_pca(embeddings, is_missing, n_components=n_components, random_state=random_state)
    reduced = reduce_embeddings(embeddings, is_missing, pca)

    vectors = pl.DataFrame(
        {
            "article_id": articles["article_id"],
            "detail_desc_missing": is_missing,
            **{col: reduced[:, i] for i, col in enumerate(VECTOR_COLUMNS)},
        }
    )
    return vectors, pca


def persist(
    vectors: pl.DataFrame,
    pca: PCA,
    vectors_path: Path = VECTORS_PATH,
    pca_path: Path = PCA_PATH,
) -> None:
    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    vectors.write_parquet(vectors_path)
    joblib.dump(pca, pca_path)


def load_article_text_vectors(vectors_path: Path = VECTORS_PATH) -> pl.DataFrame:
    return pl.read_parquet(vectors_path)


def load_pca(pca_path: Path = PCA_PATH) -> PCA:
    return joblib.load(pca_path)
