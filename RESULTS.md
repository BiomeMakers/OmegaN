# Results

Every number here is in the paper. Positives, ties and negatives together, because
the applicability claim only means something if the losses are visible.

---

## 1. Node classification

Platonov benchmarks, official splits. AUROC for binary targets, accuracy for
multi-class.

| feature set | minesweeper (null) | tolokers | amazon-ratings | questions | roman-empire |
|---|---|---|---|---|---|
| degree | 0.501 | 0.551 | 0.365 | 0.605 | 0.211 |
| centrality battery | 0.497 | 0.704 | 0.388 | 0.614 | 0.266 |
| ReFeX, 2-5 levels | 0.493 | 0.761-0.769 | 0.443 | 0.681 | **0.339** |
| naive 1-hop localization (4) | 0.497 | 0.693 | 0.381 | - | - |
| **Omega-N (10)** | 0.494 | **0.776** | 0.442 | **0.684** | 0.322 |
| Omega-N + battery | 0.486 | **0.791** | 0.450 | - | - |

Air-traffic graphs, twice-repeated stratified five-fold, same folds for every arm.

| | brazil (131) | europe (399) | usa (1,190) |
|---|---|---|---|
| degree alone | 0.748 | 0.564 | 0.552 |
| ReFeX, best depth | **0.786** | 0.589 | **0.685** |
| GraphWave, better of two scale settings | 0.781 | 0.571 | 0.628 |
| **Omega-N (10)** | 0.656 | 0.592 | 0.680 |

`minesweeper` is a declared null: its labels are independent of structure by
construction, and every arm returns chance, including a 100-dimensional embedding.

---

## 2. Applicability domain

Two statistics, neither computed from the descriptor's performance.

| network | mean degree (S1) | I(k;y)/H(y) (S2) | outcome |
|---|---|---|---|
| tolokers | 88.3 | 0.014 | win (0.776 vs 0.769) |
| europe-airports | 30.0 | 0.379 | tie (0.592 vs 0.589) |
| usa-airports | 22.9 | 0.328 | tie (0.680 vs 0.685) |
| amazon-ratings | 7.6 | 0.002 | tie (0.442 vs 0.443) |
| questions | 6.3 | 0.033 | tie (0.684 vs 0.681) |
| minesweeper (null) | 7.9 | 0.000 | chance for every arm |
| brazil-airports | 15.3 | **0.650** | out of domain by S2 |
| roman-empire | **2.9** | 0.087 | out of domain by S1 |

The rule "mean degree at least 4 and I(k;y)/H(y) at most 0.5" partitions the eight
networks without error, and every threshold pair in [3, 6] × [0.45, 0.60] gives the
same partition. **The two networks it excludes are the two on which the descriptor
loses.**

The rule was formulated after observing those two losses. What defends it is that
both statistics are computed without reference to performance, so it can be applied
prospectively and be wrong. That has not been tested out of sample.

---

## 3. Regression on an exogenous attribute

Twitch social graphs, audience size as target. Five-fold cross-validation, same
folds for every arm; parenthesised figure is the standard error across folds, which
is optimistic and used only as a scale.

| graph | n | degree | battery | Guimerà-Amaral | Omega-N | Omega-N + battery |
|---|---|---|---|---|---|---|
| PT-BR | 1,912 | 0.468 (.021) | 0.517 (.018) | 0.435 (.018) | **0.583** (.015) | **0.603** (.017) |
| ES | 4,648 | 0.419 (.016) | 0.448 (.013) | 0.380 (.014) | **0.539** (.016) | **0.557** (.015) |
| EN-GB | 7,126 | 0.327 (.018) | 0.361 (.024) | 0.299 (.021) | **0.410** (.017) | **0.431** (.021) |

Two of the three margins over the battery sit between roughly four and seven fold
standard errors; EN-GB is around two.

---

## 4. Drug-target prioritisation — the strongest positive

AUPRC, read as primary because positives are 5.5% to 9.3% of nodes.

| network | base rate | battery | Omega-N | margin |
|---|---|---|---|---|
| STRING, combined | 6.6% | 0.2891 | **0.4007** | +0.112 |
| STRING, no text mining | 8.9% | 0.3870 | **0.5306** | +0.144 |
| STRING, experimental only | 9.3% | 0.4147 | **0.4966** | +0.082 |
| BioPlex 293T (independent) | 5.5% | 0.0934 | **0.1668** | +0.073 |

Ten independent repetitions with per-repetition degree matching: STRING without text
mining +0.1047 (95% CI [+0.0968, +0.1127], 10/10, Wilcoxon p = 0.00195); BioPlex
+0.1030 ([+0.0955, +0.1106], 10/10, p = 0.00195).

The margin holds while the battery itself moves from 0.09 to 0.41 across
constructions sharing no experimental technique and no curation.

---

## 5. The negatives

**Omega-N does not improve on learned representations.**

| arm | features | AUROC | AUPRC |
|---|---|---|---|
| centrality battery | 6 | 0.7833 | 0.5158 |
| Node2Vec | 64 | 0.8235 | 0.5761 |
| battery + Node2Vec | 70 | **0.8463** | **0.6466** |
| Omega-N | 10 | 0.7936 | 0.5374 |
| battery + Node2Vec + Omega-N | 80 | 0.8458 | 0.6458 |

+0.0014 AUPRC over five seeds, positive on three of the five, Wilcoxon p = 0.31.

**Most of the absolute signal is curation.** On BioPlex, degree alone scores AUROC
0.5007 and the whole battery 0.5099. Omega-N reaches 0.5602: better than chance, and
not a prioritisation tool.

**Financial correlation graphs close.** On the framework's own synthetic epicentre
generator the raw triangle count beats the configuration-null score in all four
regimes, and on real data the raw count correlates +0.994 with strength. Where the
target is concentration, strength is a sufficient statistic.

The one modest positive there: partial rank correlation with forward marginal
expected shortfall, controlling past MES and beta, +0.142 on the complete graph
(16/19 windows, p = 0.0044). No lead time, no economic significance claimed.

---

## 6. Independence checks

**Not the degree.** Rank correlation of the contraction with degree, 0.48 to 0.81
across seven graphs; the reference point is that the raw per-node triangle count
correlates 0.996 with degree in the financial application of the same framework.

**Not curvature.** On seven of eight networks the rank correlation between the
triadic-excess coordinate and Ollivier-Ricci curvature does not exceed 0.18 in
absolute value and changes sign across graphs. The exception, roman-empire at +0.460,
is the sparsest graph in the set and is consistent with the applicability rule.

**Not the spectrum.** Per-node triangle counts are not spectrally determined:
cospectral graphs with different per-node counts exist. That is the formal statement
behind the whole localization argument, and the basis of the gap-matched null in
`notebooks/`.
