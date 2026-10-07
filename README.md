# H&M Next Best Offer Recommender

**Live demo: https://two-stage-next-best-offer-recommender-croyeynkit6mhwn8gp9ksn.streamlit.app/**

## Introduction & Goals

This project recommends 12 articles to each of 1,371,980 customers and then decides, separately, whether to push a single offer to any of them. It runs on the H&M Personalized Fashion Recommendations dataset: 31.8 million transactions over two years across a catalogue of 105,542 articles.

Scoring every customer against every article is roughly 145 billion pairs, far beyond what a ranking model can handle. So the work splits into three layers:

- **Candidate generation.** Narrows the catalogue to at most 100 candidates per customer using five cheap strategies. Nothing expensive runs at this scale.
- **Ranking.** Orders those candidates with a model that can afford to be expensive, because it sees only a hundred rows per customer. This retrieval-then-ranking shape sits behind search ranking and personalisation wherever either is done at scale.
- **Offer decisioning.** Ranking asks what a customer would most likely buy; pushing an offer asks whether contacting them is worth it at all, and about which article. Different question, different inputs, so it stays its own layer.


### Goals

- **Get recommendations well above what a no-model rule would give.** **How I know it worked:** MAP@12 of 0.03529 on a validation week no model was fitted on, against 0.00875 for a recent-popularity rule, 4.03x better.
- **Know what limits the system, before tuning anything.** **How I know it worked:** candidate recall was measured first and came out at 0.1011, which caps MAP@12 at 0.12031 no matter how good the ranker is. The ranker reaches 29.3% of that ceiling, so the remaining headroom is in candidate generation, not ranking.
- **Confirm the result holds on a week the model never saw.** **How I know it worked:** a separately trained ranker on an earlier week reaches 0.03400 against a 0.00660 popularity baseline, 5.15x, and lands at 0.296 of its oracle ceiling against the validation week's 0.293. Two independent weeks, the same ratios.
- **Serve it without the pipeline running behind the request.** **How I know it worked:** `/recommend` reads a precomputed SQLite store and answers in about 3 ms with the API process holding roughly 50 MB. An earlier design that loaded the store into memory measured 14 GB.
- **Make every reported number reproducible from the raw CSV files.** **How I know it worked:** one command runs the whole pipeline in 1h25m and reproduces every figure in the reports, verified by content hash across 845 files with zero content changes.

## Approach

**Candidate generation** narrows 105,542 articles to about 100 per customer using five strategies: past purchases, other colours of those, what is selling now, what is selling in their age group, and what similar customers bought. Caps apply as candidates are generated, so memory stays bounded, and the two history-based strategies look back only 52 weeks because older purchases point at articles that have stopped selling.

**LightGBM** supplies both models, for two purposes rather than as an ensemble: the ranker orders candidates, and the classifier produces a purchase probability the ranker cannot give. The classifier trains at 10 negatives per positive, so its raw output runs about 350 times too high and is calibrated on the unsampled pool.

**The offer decision layer** drops articles that should not be pushed, scores the rest as `margin_rate x price x purchase_probability - contact_cost`, and offers the best if it clears zero. It is a rule, not a model, and a simulation: the dataset has no record of offers sent, so acceptance and revenue cannot be measured.

**SQLite** holds 16.5 million finished offers keyed by customer and rank, so each customer's rows come back in one index seek. Batch scoring is the only place the pipeline runs; serving reads its output and never recomputes, which keeps training and serving from drifting apart.

**FastAPI** exposes `/recommend`, `/health` and `/metrics`, calling one scoring function that queries the store read-only. Responses state that prices are in scaled units and that the ranker score is not a probability, both tested.

**Streamlit** runs one codebase in two modes. Cloud mode reads a 3.5 MB bundled sample in process with no backend, which is what makes a public demo possible; local mode reads the full store and calls FastAPI under Docker Compose.

**Prefect** orchestrates two flows, twelve tasks from raw CSV to metrics and eight from scoring date to offer store. Each scoring task runs in its own process, because running them together left memory retained between tasks and pushed the heaviest one 1.3 GB over budget.

## Results Highlights

Full methodology and every number's derivation are in `reports/model_evaluation.md`.

- **Candidate recall is the ceiling: 0.1011.** 21,614 of 213,728 true purchase pairs make it into a candidate pool. This was measured and recorded before any ranker tuning, because tuning a ranker under an unmeasured ceiling optimises against a bottleneck nobody has quantified.
- **MAP@12 of 0.03529** on the validation week, against 0.00875 for recent popularity (4.03x) and 0.01346 for the candidate pool's own ordering (2.62x). The second comparison isolates what the ranker and its 47 features add over candidate order alone.
- **29.3% of the oracle ceiling.** The best possible ordering of these exact candidates scores 0.12031. Further gains have to come from candidate generation.
- **Out-of-time check holds.** A separately trained ranker on the 2020-09-09 week reaches 0.296 of its oracle against the validation week's 0.293, and 2.65x over candidate order against 2.62x. One leak is disclosed: its iteration count was chosen by early stopping that watched that same week.
- **Calibration works where it matters.** The aggregate Brier score barely moves at a 0.02% base rate, but the reliability table tracks across all ten bins, and the top 1% of candidates shows a 23.1x lift over the base rate.
- **MAP@12 is a ranking measure, never accuracy.** At a 0.02% positive rate a model predicting zero everywhere would score above 99.9% on accuracy, which is why it is not computed anywhere in this project.

## Business Impact

Full narrative and every assumption are in `reports/business_impact.md`.

The offer layer sends one offer to about 642,000 customers, roughly 47%, and leaves the rest alone. For about 254,000 of those, the offered article is not their top recommendation, and three quarters of those cases are the economics overriding the ranking rather than an article being off-limits. When it does override an eligible top pick, the replacement costs about 26% more.

**None of this is validated.** The dataset has no record of offers sent, so acceptance rate, incremental revenue and uplift are unmeasured and are not reported anywhere in this repository. Measuring them would need a log of offers actually sent, a group of customers deliberately left alone for comparison, and a way to estimate what a different rule would have done. Prices are in the dataset's scaled units, which carry no stated unit, so no figure here is a currency amount.

## Repository Structure

```
data/                     # Kaggle CSVs and derived Parquet (not committed)
notebooks/                # EDA, candidates, ranking, offer layer
src/
  schema.py               # Pandera contracts for the four raw tables
  dataset.py              # weekly label windows, labels, negative sampling
  candidates.py           # the five candidate strategies and pooling
  features.py             # customer, article, pair and provenance features
  text_embed.py           # detail_desc embedding, reduced to 32 dimensions
  rank.py                 # ranker and classifier training
  calibrate.py            # isotonic calibration and reliability
  nbo.py                  # eligibility and expected value
  evaluate.py             # Recall@100, MAP@12, oracle
  score_pipeline.py       # the one function serving calls
pipelines/                # one script per stage, plus the two Prefect flows
api/main.py               # FastAPI /recommend, /health, /metrics
dashboard/app.py          # APP_MODE=cloud or local
dashboard/sample/         # 3.5 MB bundled sample, the only data that ships
monitoring/drift.py       # Evidently drift check with a pass/fail gate
docker/                   # two Dockerfiles and the compose stack
reports/                  # model evaluation and business impact
```

## Setup

Place the four Kaggle CSV files in `data/raw/`. The competition's images folder is never used.

```
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0+cpu
pip install -r requirements.lock.txt
```

torch is installed first, CPU-only, from PyTorch's own index so package resolution does not cross between indexes. `requirements.txt` is the readable list of direct dependencies; `requirements.lock.txt` is what actually gets installed.

Run the pipeline:

```
python -m pipelines.train_flow
python -m pipelines.batch_score_flow --scoring-date 2020-09-16
```

Run the dashboard:

```
streamlit run dashboard/app.py                    # cloud mode, standalone
APP_MODE=local streamlit run dashboard/app.py     # needs FastAPI running
```

Run the full stack:

```
docker compose -f docker/docker-compose.yml up --build
```

Re-execute a notebook with `python -m pipelines.execute_notebook notebooks/<name>.ipynb`. It runs through a real kernel and writes back in place, and sets the environment variables that keep absolute paths and progress bars out of the committed output.

## Known Limitations

- **The offer layer is a simulation** and is not validated against outcomes. Its selections also concentrate on few articles, 41% of offers on ten of them, because no per-article capacity is modelled.
- **One scoring week.** Everything covers 2020-09-16. The drift gate needs the scoring week's labels, so it runs retrospectively, and a monitor that runs at scoring time needs a second scored week that does not exist yet.
- **Calibrated probabilities carry little spread.** They cluster near the base rate except in the top percent of candidates, so offer coverage is driven mainly by the contact-cost assumption.
- **The out-of-time backtest is not fully blind.** Its iteration count was chosen with a view of its own evaluation week, and it differs from the served model's.
- **No load test.** There are no p95 or throughput figures. The Locust file is a stub.
- **Hosted CI has not run.** Every step was executed locally against a clean copy of the repository.

## Further Reading

- `reports/model_evaluation.md`: the full evaluation, the candidate recall ceiling and the cap sweep behind it, calibration, the out-of-time backtest and its disclosed leak, and the drift check with its thresholds and reasoning.
- `reports/business_impact.md`: what the offer layer decides, the two assumptions behind it, how sensitive the results are to them, and what the figures do not support.