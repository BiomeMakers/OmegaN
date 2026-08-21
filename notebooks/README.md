# Notebooks

## `gap_matched_null.ipynb`

Generates families of graphs that share a target spectral gap to a chosen
tolerance while differing in local triadic organisation. It is the matched null
model described as missing in Deco et al., *Quantum-Like Dynamics in Whole-Brain
Models of the Human Connectome*, Adv. Sci. 2026, e77103.

Why it is possible: the spectral gap is a functional of the spectrum, per-node
triangle counts are not, and cospectral graphs with different per-node triangle
counts exist. Two graphs can therefore agree on the gap and disagree on local
structure.

Why the triadic excess rather than the clustering coefficient: over 600 pruned
k-regular graphs (n=40, k=20), the fraction of variation not explained by the gap
is 62% for the triadic excess and 19% for clustering. Matching on the gap nearly
matches on clustering as well, so clustering cannot serve as the contrast. Cell 4
of the notebook reproduces that measurement at whatever size you intend to work.

Runs on a free Colab CPU in about a minute, installs nothing, and exports the
connectivity matrices as `.mat` plus a summary table.

It produces the stimulus set, not the answer: whether model fit changes at fixed
gap has to be measured inside the whole-brain model.
