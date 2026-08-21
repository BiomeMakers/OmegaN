# Experiments

One script per result. Each fetches its own data from the original source; nothing
is redistributed here.

These are working scripts, not a packaged pipeline: paths and dataset locations at
the top of each file will need adjusting. They are included so that every number in
the paper can be traced to the code that produced it.

## Descriptor properties (Section 2)

| script | what it produces |
|---|---|
| `curv.py` | Ollivier-Ricci curvature against the triadic-excess coordinate, exact transport, four small graphs |
| `curv_grandes.py` | the same on the four large graphs, with the declared subsampling (800 edges, 40 neighbours, distances capped at 3) |
| `twitch25.py` | the Twitch regression of Section 2.5, ten features at the released 20-term setting |

## Applicability domain (Section 4.2)

| script | what it produces |
|---|---|
| `cribado_predice.py` | S1 and S2 for the eight benchmark networks, and the partition |

## Protein interaction networks (Section 4.1)

| script | what it produces |
|---|---|
| `h2h2.py` | Omega-N against the centrality battery on curated STRING |
| `notext.py` | the same with the text-mining channel removed |
| `bioplex.py` | the independent AP-MS replication |
| `bias.py` | the three bias controls, including study intensity given to the rival |
| `sweep.py` | the confidence cut-off sweep, 400 to 900 |
| `lcc.py` | the connectivity defect and its correction |
| `signif.py` | ten repetitions, per-repetition degree matching, paired Wilcoxon |
| `gnn_h2h.py` | **the negative result**: centralities plus Node2Vec, with and without Omega-N |

## Spectral-gap-matched null (Section 5.5)

| script | what it produces |
|---|---|
| `deco_nulo.py` | the 62% / 19% measurement on 600 pruned k-regular graphs |

The interactive version of the last one, which exports connectivity matrices ready
to drop into a whole-brain model, is in `../notebooks/gap_matched_null.ipynb`.
