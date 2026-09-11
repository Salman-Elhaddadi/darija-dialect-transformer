# Results

Everything below is computed on the same 1,852-row test split (fixed seed 42, stratified,
sha256-pinned in `splits.sha256`). Every model in the comparison was trained and scored inside
a single environment, so the numbers are internally consistent — see *Reproducibility caveat*
at the bottom for why that matters.

## The test set

| | count | share |
|---|---|---|
| MSA (label 0) | 1,309 | 70.7% |
| Darija (label 1) | 543 | 29.3% |

The classes are imbalanced roughly 7:3, which is why every number here is **macro**-F1 rather
than accuracy. A degenerate model that always predicts MSA scores 70.7% accuracy but only
0.414 macro-F1, so accuracy would flatter every model by about 30 points and hide whether the
minority class is being learned at all.

## Headline comparison

All four models scored on the identical predictions persisted in `reports/preds_baselines.csv`
and `reports/preds_marbert.csv` — same rows, same order, one seed each.

| Model | macro-F1 | Errors (/1852) | Deployment artifact |
|---|---|---|---|
| Always-MSA (degenerate reference) | 0.4141 | 543 | — |
| LR, char n-gram + TF-IDF | 0.8625 | 213 | ~1 MB |
| MLP, char n-gram + TF-IDF | 0.8497 | 236 | ~1 MB |
| MARBERT, frozen encoder + linear head | 0.8983 | 155 | 5 KB head + encoder from the Hub |
| MARBERT, full fine-tune | **0.9094** | 138 | 654 MB checkpoint |

## Seed stability

The MARBERT variants were each trained with 5 seeds. The single-seed column above is the seed
whose per-example predictions are persisted for error analysis; it is the **best** of the five,
so it is not the number to quote as the expected result. The mean is:

| Variant | macro-F1, 5-seed mean ± std | per-seed |
|---|---|---|
| MARBERT frozen | 0.8964 ± 0.0017 | 0.8983, 0.8962, 0.8948, 0.8944, 0.8982 |
| MARBERT fine-tuned | 0.9063 ± 0.0025 | 0.9031, 0.9089, 0.9039, 0.9063, 0.9094 |

Read the mean as the result and the single-seed table as the artifact the error analysis is
computed on. Both MARBERT variants clear both baselines by far more than their own seed spread
(the frozen mean beats LR by 0.034, roughly twenty times its own standard deviation), so the
ranking is not seed noise. The gap *between* the two MARBERT variants is about 0.010 against
standard deviations of 0.002–0.003 — real, but an order of magnitude smaller than the gap to
the baselines.

The two classical baselines were not re-run across seeds in this repo, so they appear as a
single number each. LR is deterministic; the MLP is not, and its 0.8497 should be read with
that caveat.

## Pooling choice

The frozen variant's pooling strategy was selected on the **validation** split before touching
test: mean pooling 0.8759, CLS pooling 0.8739. Mean pooling won and is what ships in
`models/marbert_frozen_head.pt` (the file records `pooling: "mean"` alongside the weights so
serving code cannot silently disagree with training).

The margin is 0.002 — small enough that this should be read as "the two are equivalent and mean
was picked", not as evidence that mean pooling is better for this task.

## Is MARBERT a different model, or the same boundary executed better?

LR and the from-scratch MLP get **84.5%** of their errors on the same rows: two different
algorithms rediscovering one decision boundary from the same character n-gram features. If
MARBERT were merely a better-tuned version of that boundary, its errors would overlap heavily
too.

| | Share of LR's 213 errors also missed |
|---|---|
| MLP | 84.5% |
| MARBERT frozen | 45.1% |
| MARBERT fine-tuned | 43.2% |

It does not. MARBERT is wrong about substantially different examples, which is the evidence
that the pretrained encoder contributes information the n-gram features never had, rather than
just fitting the same signal more tightly.

68 rows (3.7%) are missed by all four models. That set is the practical ceiling for this
feature space and label definition — worth reading before assuming more modelling would help.

## Error rate by text length

Test rows split into quartiles by character length, Q1 shortest:

| Model | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| LR | 14.8% | 12.1% | 10.7% | 8.2% |
| MLP | 17.9% | 13.2% | 11.6% | 8.0% |
| MARBERT frozen | 11.7% | 7.0% | 7.1% | 7.4% |
| MARBERT fine-tuned | 9.1% | 8.1% | 6.7% | 5.8% |

Every model degrades on short text, which is expected — a five-word sentence carries fewer
dialect markers. The interesting part is the *slope*. LR's error rate nearly doubles from Q4 to
Q1 (8.2% → 14.8%); MARBERT fine-tuned rises far more gently (5.8% → 9.1%). Short input is where
the pretrained encoder earns the most, which is also where a real deployment lives: search
queries, comments, and chat messages are mostly Q1.

## What this costs to deploy

Full fine-tuning buys about 0.010 macro-F1 over the frozen encoder and costs a 654 MB
checkpoint instead of a 5 KB linear head. The frozen variant pulls the unmodified encoder from
the Hub at boot, so the only thing this project has to store, version, and ship is the head.

That is why the frozen variant is what gets deployed here and the fine-tuned checkpoint is not:
at roughly one point of macro-F1, the fine-tuned model is not worth a 654 MB artifact on
free-tier hosting. The tradeoff would flip if the point mattered more than the ops cost.

## Reproducibility caveat

An earlier attempt to reproduce the baseline repo's exact historical numbers failed
`splits.sha256` verification even with the seed fixed, because a different pandas/scikit-learn
version pair ordered rows differently before the stratified split. Fixing a seed pins the
*sampling*, not the row order the sampler sees.

Rather than quote numbers across that boundary, every model in the tables above was retrained
and rescored in one environment against a freshly generated `splits.sha256`. The comparison is
therefore internally valid — which is what the comparison is actually for — but the absolute
figures are not directly comparable to numbers published from earlier runs of the baseline repo.

## Not measured

- **Latency and throughput.** The tables cover accuracy and artifact size only. MARBERT is
  obviously much slower per request than a TF-IDF pipeline, but that has not been quantified
  here, so no claim is made about it.
- **Generalization beyond the MAC corpus.** One dataset, one collection process. Performance on
  YouTube comments, forum posts, or SMS is untested.
- **Latin-script Darija (Arabizi).** The corpus is Arabic-script only. Latin-script input is not
  a hard case for this model, it is an out-of-domain case, and the serving code flags it rather
  than scoring it silently.
