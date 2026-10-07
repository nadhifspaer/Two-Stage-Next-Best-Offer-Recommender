# Business Impact

**This is a simulation.** The dataset has no record of offers ever being sent: no log of who was contacted, with what, or when. Nothing here says whether customers would accept an offer or whether offers would lift sales, because neither can be measured from this data.

Every price figure is in the dataset's scaled price units. The price column has no stated unit, so none of these are currency amounts.

## Summary

- One offer goes to about 642,000 of the 1.37 million customers, roughly 47%. The rest are left alone.
- Every customer still receives 12 ranked recommendations. The layer decides only about the single pushed offer.
- For about 254,000 of those offered, the offered article is not their top recommendation. Three quarters of that is the economics overriding the ranking; one quarter is the article being off-limits.
- When it overrides an eligible top pick, the replacement costs about 26% more.
- Offers pile up: 10 articles take 41% of everything sent.
- Coverage swings from 1% to 81% depending on what a contact is assumed to cost.

## What the Layer Decides

The ranking model recommends 12 articles. The decision layer asks a different question: should this customer be contacted at all, and about which article?

For each customer it:

1. Drops articles that should not be pushed.
2. Scores the rest by what the offer is worth.
3. Offers the best one, if it is worth more than the contact costs. Otherwise, nothing.

**An article is dropped when either is true:**

- It has not sold at all in the 7 days before the scoring date, so it has stopped selling.
- The customer already bought it in the last 14 days, so they just bought it.

**What an offer is worth:**

```
value = margin_rate x price x chance_of_purchase - contact_cost
```

The chance of purchase comes from a calibrated probability model, never from the ranking score, which orders articles but is not a probability.

## The Two Assumptions

Both are chosen inputs, not measurements. Every number below depends on them.

| Assumption | Value | Meaning |
|---|---|---|
| `margin_rate` | 0.5 | Half of an article's price is assumed to be margin |
| `contact_cost` | 0.001 x median article price = 0.0000220 | What one contact is assumed to cost |

Median taken over the 29,009 articles that sold at least once in the four weeks before the scoring date.

**How the contact cost was picked:**

- It started at 0.02 of median price, set before any probability had been calibrated.
- At that value an offer needed a 4% chance of purchase to be worth sending, while the highest probability anywhere in the data was 2.66%. Only 1% of customers cleared it.
- Lowered to 0.001, which asks for 0.2% instead, inside the range the model actually produces.
- So it was set after looking at the probabilities, to a value that makes the rule do something. It says nothing about what contacting a customer really costs.

## Who Gets an Offer

| Metric | Value |
|---|---|
| Customers offered something | 642,169 |
| Customers left alone | 729,811 |
| Total customers | 1,371,980 |
| Coverage | 46.81% |

- The 53% who get nothing either have no eligible article among their 12, or none worth more than the contact cost.
- About 20% of those offered had their probabilities fitted on their own purchase history, which makes those slightly optimistic.

## Which Articles Get Offered

| Metric | Value |
|---|---|
| Distinct articles offered | 4,876 of 105,542 |
| Total offers | 642,169 |
| Share going to the 10 most offered | 41.41% |

- Roughly 266,000 customers are offered one of the same 10 articles.
- Nothing in the rule spreads offers around: an article that is expensive, still selling and well scored for many customers wins over and over.
- A real campaign has stock limits and a cap on repeat pushes. Neither is modelled here.

Share of offers by candidate source. An article can be found by more than one, so these sum to over 100%.

| Source | Share |
|---|---|
| Popular in the customer's age group | 45.13% |
| Something they bought before | 35.77% |
| Popular overall | 28.22% |
| Another colour of something they bought | 28.20% |
| Similar customers bought it | 15.08% |

## When the Offer Is Not the Top Recommendation

For 254,198 customers, 39.58% of those offered, the offered article is not the ranking model's first pick.

| Why | Customers | Share of those offered | Share of the 254,198 |
|---|---|---|---|
| Top pick was bought in the last 14 days | 63,923 | 9.95% | 25.15% |
| Top pick had not sold in the last 7 days | 35 | 0.005% | 0.01% |
| Top pick was fine, another article was worth more | 190,240 | 29.62% | 74.84% |

- Only the third case is the economics at work. In the other two the top pick was never available.
- The first case outnumbers the second by more than 1,800 to 1, which fits the ranking model's top pick often being something the customer buys repeatedly, then removed by the 14-day rule.
- In the third case, the offered article costs about 26% more than the top pick at the median.

## Offered Prices Against Top Recommendations

Scaled price units, across customers who were offered something.

| Quantile | Offered article | Top recommendation |
|---|---|---|
| p10 | 0.01668 | 0.01330 |
| p25 | 0.03307 | 0.02486 |
| p50 | 0.04143 | 0.03330 |
| p75 | 0.04156 | 0.04146 |
| p90 | 0.04992 | 0.04951 |
| mean | 0.03883 | 0.03311 |

- Offers sit at higher prices than top recommendations across the lower and middle of the range.
- That follows from the rule: between two articles with a similar chance of purchase, the pricier one is worth more.
- It says what the rule picks, not what customers would accept.

## How Much the Assumptions Matter

| `margin_rate` \ contact cost multiplier | 0.0005 | 0.001 | 0.002 | 0.005 | 0.01 | 0.02 |
|---|---|---|---|---|---|---|
| 0.3 | 55.20% | 24.47% | 9.15% | 1.42% | 0.27% | 0.01% |
| 0.4 | 63.49% | 38.23% | 13.45% | 2.74% | 0.49% | 0.05% |
| **0.5** | 80.51% | **46.81%** | 18.38% | 4.24% | 0.87% | 0.13% |
| 0.6 | 87.67% | 55.20% | 24.47% | 6.55% | 1.42% | 0.27% |

- At `margin_rate` 0.5, coverage falls from 47% to under 1% as the contact cost rises tenfold, and climbs to 81% when it halves.
- Treat coverage as a dial, not a finding.
- **Which article gets picked does not depend on either assumption.** Margin scales every article's value by the same amount and contact cost subtracts the same amount from all of them, so neither changes the winner. They only change whether that winner is worth sending, and only through the ratio between them. Cells with equal ratios give identical results, which the code checks and the tests cover.

## What This Does Not Tell You

- Whether any customer would have accepted an offer.
- Whether offers would raise sales or margin.
- Whether the layer's picks are better or worse than the ranking model's. Different questions, no outcome data for either.
- Anything about the probabilities beyond their place in the ranking. They cluster near the base rate and only separate meaningfully in the top 1% of candidates.

## Limitations

- Every figure rests on two assumed values. Change them and every figure changes.
- Offers concentrate on a handful of articles, which no real campaign would allow.
- One scoring week only. Nothing here shows how the picture moves over time.
- About a fifth of those offered had probabilities fitted partly on their own data.
- Making this real needs three things the data lacks: a record of offers actually sent, a group deliberately left alone for comparison, and a way to estimate what a different rule would have done from logged results.

## Where the Numbers Come From

| Figures | Source |
|---|---|
| Coverage, article distribution, source mix, price quantiles, sensitivity | `data/processed/nbo/_manifest.json` |
| The three divergence cases | `data/processed/nbo/_divergence_manifest.json` |
| Median price and probability quantiles | `data/processed/calibration/_offer_threshold_report.json` |
| The offer table, 642,169 rows | `data/processed/nbo/offers_2020-09-16.parquet` |
| Walkthrough | `notebooks/04_offer_decision_layer.ipynb` |

Scoring date is 2020-09-16, the one week where recommendations can be lined up against real purchases by a model whose accuracy is measured.