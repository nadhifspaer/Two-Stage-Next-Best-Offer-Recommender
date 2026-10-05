# Offer summary for the local dashboard: coverage, strategy mix, assumptions, top selected articles.
# Reads the decision layer's manifest and offers parquet, never the SQLite offer store.
import json
from pathlib import Path

NBO_DIR = Path("data/processed/nbo")
MANIFEST_PATH = NBO_DIR / "_manifest.json"


def load_offer_summary(top_n: int = 10, nbo_dir: Path = NBO_DIR) -> dict:
    import polars as pl

    manifest = json.loads((nbo_dir / "_manifest.json").read_text(encoding="utf-8"))
    offers_path = nbo_dir / f"offers_{manifest['scoring_date']}.parquet"
    top_articles = (
        pl.scan_parquet(offers_path)
        .group_by(pl.col("article_id").cast(pl.String))
        .agg(pl.len().alias("selections"), pl.col("price").first().alias("price"))
        .sort("selections", "article_id", descending=[True, False])
        .head(top_n)
        .collect()
    )
    return {
        "scoring_date": manifest["scoring_date"],
        "assumptions": {
            "margin_rate": manifest["margin_rate"],
            "contact_cost_multiplier": manifest["contact_cost_multiplier"],
            "median_trailing_4w_price": manifest["median_trailing_4w_price"],
            "contact_cost": manifest["contact_cost"],
        },
        "coverage": manifest["coverage"],
        "top_article_concentration": manifest["top_article_concentration"],
        "strategy_mix": manifest["strategy_mix"],
        "top_selected_articles": top_articles.to_dicts(),
    }
