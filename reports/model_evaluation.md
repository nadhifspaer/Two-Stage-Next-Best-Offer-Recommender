# Model Evaluation

## Metrics and Evaluation Population

- Validation label week is 2020-09-16 to 2020-09-22. No model is fitted on it.
- 68,984 customers purchased in that week, 5.03% of the 1,371,980 customer base. Every figure below is measured against those purchasers.
- 213,728 distinct (customer, article) purchase pairs exist in the window.
- Metrics: Recall@100 at the candidate stage, MAP@12 at the ranking stage, Brier score and a reliability table for the calibrated probability.
- Plain accuracy is never computed. At a 0.02% positive rate on the full pool, predicting zero everywhere would score above 99.9%.

## Leakage Control

- Features for a label window come strictly from transactions dated before that window starts.
- Splitting is by date cutoff only. `train_test_split` with shuffling appears nowhere in the project.
- Worked example, one customer in the 2020-08-19 window: `days_since_last_purchase = 68` traces to a last pre-window transaction on 2020-06-12, and `txn_count_12w = 4` matches exactly 4 transactions in [2020-05-27, 2020-08-19). The label purchase fell on 2020-08-19 itself, untouched by any feature.
- `tests/test_leakage.py` asserts all three and fails if any feature derives from the label.

## Determinism

- Candidate generation is seeded (`ALS_RANDOM_STATE` 42) and every rank-and-cap site carries an explicit `article_id` tiebreak. Polars does not guarantee stable tie resolution; unstable ties changed the pooled candidate count by 572,366 rows between runs on identical input.
- Both models use a fixed seed, `deterministic=True` and `force_row_wise=True`. Two training runs give identical predictions.
- Float reductions never run under the streaming engine: parallel partial sums combine in a non-deterministic order and float addition is not associative. The default engine measured both faster and lighter here.
- `sum_prob` and the Brier score accumulate in float64 via `math.fsum`, and the reliability table bins from a fully ordered sort.

## Candidate Stage

Validation week, `CUSTOMER_CHUNK_SIZE` 40,000, 35 partitions, 313.2s, 4.50 GB peak RSS.

| Metric | Value |
|---|---|
| Pre-pool candidate rows | 128,080,320 |
| Pooled candidate rows | 105,365,317 |
| Customer coverage | 1,371,980 of 1,371,980 |
| Candidates per customer (min / mean / max) | 25 / 76.8 / 100 |

- Five strategies, capped before pooling: repurchase 30, colour variant 30, recent popularity 20, segment popularity 20, ALS 30. Capping at generation bounds memory, not just the final pool.
- Uncapped, colour variant alone produced about 72 candidates per active customer, making the stage materialise roughly twice what it kept.
- Repurchase and colour variant look back only 52 weeks: a stale candidate still consumes one of the 100 slots and displaces a popularity candidate with a real chance.
- ALS carries no horizon. It trains on full history and does not point at a specific past purchase, so it has no stale-article risk.

**Customer coverage by candidate source:**

| Population | Customers | Share |
|---|---|---|
| Popularity candidates only | 15,271 | 1.11% |
| At least one personalised candidate | 1,356,709 | 98.89% |

- Only 1.11% fall back entirely to popularity, far fewer than the roughly 369,000 beyond the 52-week horizon, because ALS still reaches them.

## Recall@100

**Pooled Recall@100: 0.1011.** 21,614 of the 213,728 true purchase pairs are present in the candidate pool.

- This is the ceiling on MAP@12 for everything downstream: the ranker cannot recover a purchase that was never a candidate.
- Measured and recorded before any ranker tuning, since tuning under an unmeasured ceiling optimises against an unquantified bottleneck.

| Strategy | Recall@100 | Marginal contribution |
|---|---|---|
| repurchase | 0.0330 | +0.0261 |
| colour_variant | 0.0211 | +0.0135 |
| recent_popularity | 0.0339 | +0.0075 |
| segment_popularity | 0.0359 | +0.0090 |
| als | 0.0142 | +0.0112 |

- Marginal contribution is pooled recall minus the recall of the pool with that strategy's sole contributions removed.
- Repurchase has the highest marginal contribution despite not having the highest standalone recall: it finds pairs no other strategy reaches.
- The two popularity strategies overlap heavily: high standalone recall, low marginal contribution.
- Marginal contributions sum to 0.0673 against a pooled 0.1011. The gap is pairs found by more than one strategy.

## Cap Allocation Sweep

Pool cap held at 100, everything else fixed, only the five per-strategy caps varied.

| Allocation | repurchase | colour_variant | recent_pop | segment_pop | als | Recall@100 |
|---|---|---|---|---|---|---|
| A (deployed) | 30 | 30 | 20 | 20 | 30 | **0.1011** |
| B | 40 | 30 | 10 | 10 | 40 | 0.0876 |
| C | 40 | 20 | 10 | 10 | 50 | 0.0868 |
| D | 50 | 30 | 10 | 10 | 30 | 0.0864 |
| E | 30 | 30 | 20 | 0 | 40 | 0.0942 |

- Every reallocation scored below the baseline.
- Mean pool size is 76.8 against a cap of 100, so the cap does not bind for most customers and the slots are not zero-sum.
- Dropping segment popularity (E) raised recent popularity's marginal contribution from +0.0075 to +0.0306, confirming the overlap. The redundant coverage still mattered: pooled recall fell to 0.0942.

**Whether the caps truncate anything:**

| Strategy | Cap | Customers reaching it | Share |
|---|---|---|---|
| repurchase | 30 | 108,639 | 7.9% |
| colour_variant | 30 | 578,404 | 42.2% |

- Raising the repurchase cap would add candidates for only 1 customer in 13, which is why allocation D lost.
- This measures available volume only. Whether extra candidates would be positives has not been tested.

## Training

Two models on the same 1,002,471-row matrix, from four training label windows (2020-08-19 to 2020-09-15), negatives sampled at 10 per positive.

| Metric | Value |
|---|---|
| Training rows | 1,002,471 |
| Positive rate | 9.10% |
| Wall time, both models | 68.6s |
| Peak RSS | 2.62 GB |
| Ranker best iteration | 207 |
| Classifier best iteration | 279 |

- **Two models, two purposes, not an ensemble.** `LGBMRanker` (`lambdarank`) emits a relative ordering score with no probabilistic meaning. `LGBMClassifier` emits a probability, which the offer layer needs and the ranker cannot give.
- **The classifier is unusable before calibration.** Its raw output is calibrated to the sampled 9.10% positive rate, not the real population rate.
- **Early stopping never touches the validation week.** Both fit on the first three windows, early stop on 2020-09-09 for an iteration count, then refit on all four with that count fixed.
- **Early stopping scores are not reported anywhere.** Those groups hold roughly 1 positive per 11 rows, so any MAP or NDCG there is inflated against the real pool of 76.8 candidates per customer. Only the iteration counts are kept.
- **Feature allowlist.** `FEATURE_COLUMNS` in `src/rank.py` is the single source of what reaches either model: 47 columns, excluding `customer_id`, `article_id`, the label and the window.
- **One categorical mapping**, fit once and persisted, applied identically at training and inference. An unseen category maps to a fixed unknown code rather than shifting other codes.

**Top 10 features by gain.** Both models agree closely on the order.

| Feature | Ranker share | Classifier share |
|---|---|---|
| purchase_count_1w | 25.08% | 27.27% |
| days_since_last_purchase_this_article | 17.85% | 19.69% |
| department_name | 10.66% | 9.19% |
| rank_repurchase | 4.39% | 4.53% |
| prior_purchase_count_same_product_code | 4.25% | 4.48% |
| days_since_last_purchase | 3.60% | 3.42% |
| prior_purchase_count_same_department | 3.39% | 2.47% |
| product_type_name | 3.26% | 2.38% |
| best_rank | 3.23% | 3.04% |
| days_since_last_sale | 2.95% | 4.03% |

- No single feature dominates. The top three carry about 54% of gain across three different signals: short-term article demand, repurchase recency, and category preference.
- The second feature is null for articles the customer never bought, so part of its gain encodes whether this is a repurchase at all. That lines up with repurchase having the highest marginal contribution at the candidate stage.

## Calibration

Scored over the full, unsampled validation pool, never the sampled training rows.

| Metric | Value |
|---|---|
| Candidates scored | 105,365,317 |
| Observed positives | 21,614 |
| Observed positive rate | 0.0205% |
| Mean raw predicted probability | 7.23% |

- **The raw classifier overestimates by roughly 350x**, exactly what training at 9.10% and serving at 0.02% produces.
- Customers are split by a deterministic hash: the first 20% fit the isotonic calibrator, the next 20% evaluate it, no overlap.

| Predictor | Brier score | Brier skill score |
|---|---|---|
| Constant at base rate | 0.00019815 | 0.0 (reference) |
| Raw classifier | 0.01199719 | -59.55 |
| Calibrated | 0.00019774 | 0.0020 |

- Raw Brier alone says nothing at this base rate: a constant predictor already scores near zero, and the uncorrected classifier is 60x worse than that constant.
- Calibration barely moves the aggregate Brier either, because the error is dominated by the negative majority where a near-zero probability and the base rate contribute almost the same tiny error.
- **The reliability table is where calibration shows.** Across 10 equal-count bins, predicted and observed track closely in every bin, from 0.0000015 against 0.0000009 at the bottom to 0.0011394 against 0.0010666 at the top.
- Splitting the top decile into 10 finer bins, the top 1% predicts 0.4974% against an observed 0.4579%, a **lift of 23.1x over the base rate**. Mildly optimistic at the very top, still correctly ordered.

**What this means for the offer layer:**

- The maximum calibrated probability anywhere in the evaluation subset is 2.66%; per-customer maxima have a median of 0.163%.
- Probabilities cluster near the base rate and separate meaningfully only in the top percent.
- That constrains any rule with a probability threshold: at a contact cost of 0.02 of median price, only 1.08% of customers had a candidate clearing the bar, against 50.57% at 0.001.

## Ranking Evaluation

Every validation candidate scored, top 12 per customer with an `article_id` tiebreak.

| Metric | Value |
|---|---|
| Candidates scored | 105,365,317 |
| Customers covered | 1,371,980 of 1,371,980 |
| Scoring peak RSS | 1.57 GB |

**MAP@12, four orderings.** Customers with no purchase in the window are excluded, so the average is over the same 68,984 purchasers.

| Ordering | MAP@12 | Ranker vs. this |
|---|---|---|
| Ranker top-12 | 0.03529 | 1.00 (reference) |
| Recent popularity, same for everyone | 0.00875 | 4.03x |
| Candidate pool order, no ranker | 0.01346 | 2.62x |
| Oracle, the ceiling these candidates allow | 0.12031 | 0.29x |

- Beats a no-model popularity rule by 4.03x.
- Beats the candidate pool's own ordering by 2.62x, which isolates what the ranker and its 47 features add over candidate order alone.
- Reaches 29.3% of the oracle ceiling. The remaining headroom is bounded by candidate recall, not ranking quality.
- **Oracle definition:** for a customer with `m` true purchases and `m_found` of them in the pool, the best any ordering could do is `min(m_found, 12) / min(m, 12)`.
- MAP@12 is small in absolute terms here. Published solutions land near 0.030 to 0.037 on a different week under a different setup, so those are context for the order of magnitude, not a benchmark.

**By customer recency:**

| Recency bucket | Purchasers | MAP@12 |
|---|---|---|
| last 1 week | 12,670 | 0.10035 |
| last 4 weeks | 18,425 | 0.03006 |
| last 12 weeks | 18,124 | 0.01762 |
| last 52 weeks | 12,077 | 0.01799 |
| more than 52 weeks ago | 2,116 | 0.01075 |
| never | 5,572 | 0.00896 |

- Counts sum to exactly 68,984.
- Customers active in the last week carry the score: 18% of purchasers contribute about half the total.
- Customers with no prior history score 0.00896, essentially the popularity baseline's 0.00875, which is what they receive.
- The two smallest buckets are noisier than the rest.

## Temporal Backtest

An out-of-time check in place of an external benchmark: a second ranker trains on the first three windows only and is evaluated on 2020-09-09, a week it never saw.

| Ordering | 2020-09-09 backtest | 2020-09-16 validation |
|---|---|---|
| Purchasers scored | 72,019 | 68,984 |
| Ranker top-12 | 0.03400 | 0.03529 |
| Recent popularity | 0.00660 | 0.00875 |
| Candidate pool order | 0.01283 | 0.01346 |
| Oracle | 0.11492 | 0.12031 |
| Ranker / recent popularity | 5.15x | 4.03x |
| Ranker / candidate pool order | 2.65x | 2.62x |
| Ranker / oracle | 0.296x | 0.293x |

- The two ratios that matter land within a few points across two independent weeks: 0.296 against 0.293 of the oracle, 2.65x against 2.62x over candidate order.
- Absolute MAP@12 differs (0.03400 against 0.03529), consistent with two different weeks of real behaviour rather than a model difference.
- **The disclosed leak.** The backtest ranker's iteration count was selected by early stopping that watched 2020-09-09 itself. Its training carries that one number from its own evaluation week, though none of that week's rows.
- It does not establish that the served ranker performs identically off its validation week. It shows that a differently trained ranker of the same design gives consistent ratios one week earlier.

## Drift Check

- Reference: the four training feature windows sampled to 100,000 rows. Current: the validation week.
- Both sides scored through the pinned classifier and calibrator, with the calibrated probability compared as one more column.
- Thresholds were fixed before the first run and not adjusted afterwards.

| Rule | Value | Why | When it trips |
|---|---|---|---|
| Feature drifted | PSI at or above 0.25 | The significant band; ordinary weekly variation sits below it | Check how that feature is built and its date cutoff first |
| Dataset fails | Over 25% of 47 features drifted | Many features share the same dates and move together, so a few trips are routine | Retrain on a window nearer the scoring week |
| Probability fails | PSI at or above 0.25 | It feeds the offer decision directly, so its shift moves coverage | Check calibrator reliability on the new week, refit if the base rate moved |
| Sample size | At least 50,000 rows per side | Below that PSI bins are unstable | The run raises instead of returning a verdict |

| Comparison | Gated | Features drifted | Probability PSI | Verdict |
|---|---|---|---|---|
| Sampling matched | yes | 3 of 47 (6.4%) | 0.0004 | pass |
| Unmatched | no | 13 of 47 (27.7%) | 0.0617 | fails the share rule, not gated |

- **Why the matched comparison gates.** The training reference holds only purchasers at 10 negatives per positive, so it is 9.1% positive, against 0.02% for the scoring pool. Comparing those measures sample construction, not drift. Thresholds are identical for both and neither was lowered to make anything pass.
- **Three features still drift once sampling is matched**, and they are the real week-over-week signal: `purchase_count_1w` (0.339), `purchase_count_4w` (0.265) and `purchase_count_wow_trend` (1.325). All three are article demand features, and the scoring week has a heavier tail of high-volume articles: `purchase_count_wow_trend` at p99 is 1235 against 474. The cause was not investigated.
- Ten of the 13 unmatched failures return to normal once sampling is matched. They are customer activity features.
- The calibrated probability did not drift in either comparison. Caveat: the reference probabilities are classifier-in-sample.
- **This gate is retrospective.** Applying the sampling needs the scoring week's labels, which exist only after the week ends. A monitor running at scoring time would need a previous week's unlabeled scored pool, which needs a second scoring week that does not exist yet.

## Reproducibility

- The full pipeline runs end to end from the raw CSV files in one command, 1h25m across twelve tasks, and reproduces every figure in this document.
- Batch scoring runs the same way, 54m across eight tasks.
- Verified by content hash across 845 artifact files: zero content changes after both runs. Row order differs in 421 files, which does not affect any result.
- Six float fields drift in the ninth or tenth digit between runs, from float32 summation order. No stored probability or figure above is affected.
- Every raw table is validated against a schema contract before any transform, including fixed-width checks that catch a lost leading zero in article and product identifiers.

## Limitations

- **Candidate recall caps everything.** 89.9% of true purchases never enter the candidate set, so MAP@12 is bounded at 0.12031 no matter how well the ranker orders.
- **The score is carried by recently active customers.** Roughly 18% of purchasers contribute about half the total. For customers with no history the system performs at the popularity baseline's level.
- **The calibrated probabilities carry little spread.** A Brier skill score of 0.002 means they cluster near the base rate everywhere except the top percent.
- **The backtest is not fully blind.** Its iteration count came from early stopping on its own evaluation week.
- **One validation week.** Nothing here shows how performance moves over a longer period, and the drift monitor needs a second scoring week to run as it would in production.
- **A fifth of the calibration evaluation is in-sample**, in the sense that those customers' probabilities were fitted on their own labels.