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
| **Omega-N (10)** | 0.503 | **0.795** | 0.470 | **0.693** | 0.322 |
| Omega-N + battery | 0.486 | 0.791 | 0.450 | - | - | <!-- PENDIENTE: regenerar -->

Air-traffic graphs, twice-repeated stratified five-fold, same folds for every arm.

| | brazil (131) | europe (399) | usa (1,190) |
|---|---|---|---|
| degree alone | 0.748 | 0.564 | 0.552 |
| ReFeX, best depth | **0.786** | 0.589 | **0.685** |
| GraphWave, better of two scale settings | 0.781 | 0.571 | 0.628 |
| **Omega-N (10)** | 0.656 | 0.592 | 0.680 |

The `Omega-N (10)` row was regenerated after a defect was found in the Fiedler
solver (see the note at the end of this file). The `degree` row reproduces the
earlier figures to three decimals on all five networks, which attributes the
change to the descriptor and not to the harness. `roman-empire` is unchanged
because its lambda_2 is degenerate and the Fiedler columns carry nothing there.

`minesweeper` is a declared null: its labels are independent of structure by
construction, and every arm returns chance, including a 100-dimensional embedding.

---

## 2. Applicability domain

Two statistics, neither computed from the descriptor's performance.

| network | mean degree (S1) | I(k;y)/H(y) (S2) | outcome |
|---|---|---|---|
| tolokers | 88.3 | 0.014 | win (0.795 vs 0.769) |
| europe-airports | 30.0 | 0.379 | tie (0.592 vs 0.589) |
| usa-airports | 22.9 | 0.328 | tie (0.680 vs 0.685) |
| amazon-ratings | 7.6 | 0.002 | tie (0.470 vs 0.443), under review |
| questions | 6.3 | 0.033 | tie (0.693 vs 0.681), under review |
| minesweeper (null) | 7.9 | 0.000 | chance for every arm |
| brazil-airports | 15.3 | **0.650** | out of domain by S2 |
| roman-empire | **2.9** | 0.087 | out of domain by S1 |

Two outcomes are marked *under review*. With the corrected descriptor, Omega-N now
sits above the ReFeX figure on `amazon-ratings` and `questions`, where the earlier
numbers were ties. They are **not** reclassified here: the ReFeX figures come from
code that is not in this repository, so it cannot be verified that rival and
descriptor were measured in the same harness. Reclassifying a tie as a win requires
re-measuring the rival, not only the descriptor.

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


> **Regenerated.** These sections were produced in a notebook that is not in this
> repository, while `h2h2.py` and `bioplex.py` read `interactions.tsv`, the earlier
> DGIdb label. `experiments/seccion11.py` now rebuilds them from the published
> label (Open Targets, approved stage only, bridged from Ensembl to STRING protein
> ids through the STRING alias file: 1,052 targets, the figure reported here), on
> the largest connected component, with the corrected Fiedler solver. Mean degree
> and base rate reproduce the published values exactly on all three networks, which
> is what certifies that the reconstruction is faithful.
>
> Two changes are worth reading before the numbers. First, **the rival is reported
> against both batteries**. The published figures were measured against the
> four-feature battery (degree, k-core, PageRank, clustering), which reproduces the
> BioPlex 0.0934 exactly; the text of the paper describes a six-feature battery,
> adding eigenvector centrality and neighbour-degree sum. Both are given, because
> the gap between them says how much of the margin depends on the strength of the
> rival, and because anyone reconstructing the rival will reach for the stronger
> one. Second, the **STRING no-text** variant is withdrawn: it required recombining
> channels to exclude text mining and the exact recipe is not recorded, so a
> reconstruction gave mean degree 22.5 against the 20.5 published, i.e. a different
> network. The remaining three come straight from a column of the source file and
> carry the same arguments: reference network, clean variant without text mining,
> and independent replicate.

| network | degree | base | vs battery of 4 | vs battery of 6 |
| --- | --- | --- | --- | --- |
| STRING, full | 29.8 | 6.6% | +0.1649 | +0.0723 |
| STRING, experimental only | 17.7 | 9.3% | +0.0704 | +0.0217 |
| BioPlex 293T | 17.0 | 5.5% | +0.1084 | +0.0560 |

Degree-matched AUPRC, ten repetitions, all 10/10 with saturated Wilcoxon p.
Against the four-feature battery the corrected solver leaves BioPlex slightly
higher than published (+0.1084 against +0.1030), so the correction does not cost
anything here.

Against the six-feature battery the three margins are +0.072, +0.022 and +0.056.
They no longer coincide, so the replication argument rests on three independent
networks agreeing in direction under the harder rival, not on the earlier
near-identity of two of the margins. The narrowest is `STRING experimental`, which
is also the cleanest network of the three, with neither text mining nor curation:
that is the honest place to look first, and it is stated here rather than left to
be found.

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


---

## Note on the v2 corrections

Three defects were found in the released code after v1 and are fixed in this
repository. All of them worked against the descriptor; none changes a verdict.

**The Fiedler solver did not converge.** LOBPCG ran unpreconditioned with the
constant vector inside the search block. On `tolokers` it returned a vector with
residual 2.7e-1, i.e. not an eigenpair, a lambda_2 19% too high, and an output that
changed with the random seed: two runs of the released code correlated 0.15.
Deflating the null space and preconditioning with diag(L)^-1 matches shift-invert
Lanczos to seven digits, is deterministic, and is faster. Six of the nine graphs
tested were affected.

**Two experiments ran on disconnected graphs.** STRING has 126 connected components
and BioPlex 17, so lambda_2 was zero and the three `fiedler_*` columns were inert.
`omega_n` warned about this and the scripts ignored the warning. Both now restrict
to the largest component.

**The Fiedler coordinate is not always defined.** Where lambda_2 is degenerate the
eigenvector is not unique, and no solver can converge to it: the three `fiedler_*`
columns are then seed-dependent. The statistic that detects this is the spectral gap
normalised by the spectral scale, (lambda_3 - lambda_2)/lambda_max, not the relative
gap, which misranks the cases. Measured: `minesweeper` 8e-19 and `roman-empire`
1.5e-07 (degenerate) against `tolokers` 5.0e-06, `questions` 9.9e-06,
`amazon-ratings` 1.3e-05, STRING 1.8e-05 and BioPlex 1.6e-05 (well defined). A
threshold of 1e-06 separates the two groups with almost two orders of magnitude of
margin. On the affected networks the effect on accuracy is negligible (+0.003 and
-0.001 from dropping the columns); the problem is reproducibility, not validity.
