# Omega-N

Reference implementation for **Omega-N: Interpretable Structural Node Descriptors
and Their Applicability Domain** (A. Acedo, Biome Makers Inc).

Paper: [arXiv:2609.01633](https://arxiv.org/abs/2609.01633)

Ten interpretable features per node, computed from the graph alone. No node
attributes, no training, no embeddings.

```python
from omega_n import omega_n, screen

X = omega_n(A)          # (n, 10) feature matrix from a scipy sparse adjacency
ok, why = screen(A, y)  # is this graph inside the applicability domain?
```

---

## What it does

The descriptor localizes each of the four factors of a composite resilience index
into a per-node quantity, and corrects the two conditioning failures that a direct
localization produces:

1. **degree share** — the node's contribution to the density
2. **triadic excess** — triangles at the node as an excess over the configuration null
3. **neighbourhood dispersion** — spread of neighbour degrees, relative to its null
4. **spectral energy** — the node's term in the Dirichlet energy of the Fiedler vector

Coordinates 2 to 4 are smoothed by a personalized-PageRank operator at two scales,
giving 4 + 3×2 = **10 features**.

---

## Run `screen()` before you use it

The descriptor has a stated applicability domain, decided by two statistics computed
from the graph and the labels and **not** from the descriptor's performance:

- **S1** — mean degree at least 4. A chain-like graph gives a triadic excess almost
  nothing to measure.
- **S2** — `I(degree; label) / H(label)` at most 0.5. Where the label is close to a
  function of the degree, the degree is the right answer and calibrating against a
  null removes signal rather than noise.

On the eight benchmark networks of the paper the rule separates the three wins, two
ties and the declared null from the two losses without error, with roughly a factor
of two of margin on both conditions. Applying it to a new dataset is one line, and the prediction it
makes is falsifiable.

---

## What it is not

Three negative results are part of the paper and belong here too.

**It does not improve on learned representations.** Node2Vec alone outscores it, and
adding Omega-N to a published pipeline of centralities plus Node2Vec changes nothing
(+0.0014 AUPRC over five seeds, p = 0.31). Where predictive performance is the only
criterion, use an embedding.

**On a clean biological graph there is little to read.** On BioPlex, an unbiased
AP-MS network, the six-feature centrality battery reaches 0.609 AUROC against 0.854
on curated STRING at a comparable base rate, and Omega-N 0.647 against 0.886. Most
of the absolute performance obtained on curated STRING is inherited from accumulated
curation, not discovered from structure.

**It closes on financial correlation graphs.** Where the target is concentration
itself, strength is a sufficient statistic and calibrating against a
degree-preserving null removes exactly what has to be detected.

---

## Reproducing the paper

```bash
pip install -r requirements.txt
python reproduce.py           # the node-classification tables
```

`reproduce.py` uses the released defaults, including the **20-term** truncation of
the personalized-PageRank smoothing; the tables were regenerated at that setting
rather than at the more favourable 25-term one.

**Pin your versions.** `requirements.txt` now fixes them rather than declaring
minima, and the reason is measurable: on `amazon-ratings` the Omega-N figure is
0.4702 under scikit-learn 1.6.1 and 0.4403 under 1.8.0, with identical features,
splits and seed. The ten descriptor columns have the same fingerprint on both, so
the difference is entirely the classifier, and it is of the same order as the
margins the paper reports. `requirements-lock.txt` records the full environment.

`experiments/` holds the scripts behind the individual results: the curvature
comparison, the applicability screen, the Twitch regression, the three protein
network constructions with their bias controls and significance tests
(`seccion11.py`), the ReFeX head-to-head that measures the rival in the same harness
(`refex_h2h.py`), and the spectral-gap-matched null used in Section 5.5.

Datasets are obtained from their original sources and none are redistributed. Each
script documents where its data comes from.

---

## Notebooks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BiomeMakers/OmegaN/blob/main/notebooks/gap_matched_null.ipynb)

`notebooks/gap_matched_null.ipynb` builds families of graphs matched on the spectral
gap but separated in local triadic organisation. It is the null model described as
missing in whole-brain modelling work where the gap and local density are
confounded, and the same construction addresses the performance-attribution problem
in graph rewiring. Runs on a free Colab CPU in about a minute. See
`notebooks/README.md`.

---

## Numerical caveats, stated

The spectral coordinate uses LOBPCG with the null space deflated and a Jacobi
preconditioner. On a disconnected graph it still degenerates into a component
indicator, so **restrict your graph to its largest connected component**;
`omega_n()` warns when the input has more than one. A convergence warning that
survives a relaxed tolerance means something different: lambda_2 is degenerate and
the Fiedler vector is not defined, so the three `fiedler_*` columns are
seed-dependent and should be dropped. The test is the spectral gap normalised by the
spectral scale, `(l3 - l2)/l_max`, with a threshold at 1e-06; Section 4.5 of the
paper gives the measured values.

---

## Citation

```
@article{acedo2026omegan,
  title  = {Omega-N: Interpretable Structural Node Descriptors and
            Their Applicability Domain},
  author = {Acedo, Alberto},
  year         = {2026},
  eprint       = {2609.01633},
  archivePrefix= {arXiv},
  primaryClass = {physics.soc-ph}
}
```

## Licence

MIT. See `LICENSE`.
