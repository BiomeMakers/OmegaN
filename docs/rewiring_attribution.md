# Performance attribution in graph rewiring: a gap-matched construction

**Preliminary run, 21 August 2026. Not a result yet — see the limitations.**

Alberto Acedo, Biome Makers Inc.

---

## The question

Graph neural networks pass messages between neighbours. Where the graph has a
bottleneck, information from distant regions is compressed as it crosses and is
lost: over-squashing.

The field's response is to rewire the graph before training, and it proceeds by two
families:

- **spectral** methods add edges to increase the spectral gap
- **curvature** methods act on negatively curved edges, which in practice means
  acting on triangles

The two quantities move together. A decrease in triangle count is accompanied by an
increase in the spectral gap, so any rewiring step moves both at once, and when
accuracy improves nobody can say which one produced it.

Your 2026 survey lists this as an open problem by name — performance attribution —
and notes that reported gains often arise from favourable hyperparameter
configurations rather than from consistent improvement over the original topology.

---

## Why the confound is separable

The spectral gap is a functional of the adjacency spectrum. Per-node triangle
counts are not: cospectral graphs with different per-node counts exist.

Two graphs can therefore be constructed that **agree on the spectral gap and differ
in local triadic organisation**. Rewiring the same starting graph to the same gap by
two different routes turns the comparison the field cannot make into a controlled
one: whatever difference remains cannot be attributed to the gap.

---

## Construction

`amazon-ratings`, one of the standard heterophilous benchmarks. Connected subgraph
of 3,000 nodes, 11,548 edges, mean degree 7.70.

At a **fixed edge budget**, the two routes are mixed in varying proportion:

- **spectral route**: edges between nodes far apart in the Fiedler vector, the
  first-order approximation spectral methods use
- **triadic route**: pairs with many common neighbours, which is what curvature and
  triangulation methods do when they close triangles

Sweeping the triadic fraction from 0 to 1 and measuring both quantities at each
point, one looks for two mixtures with the **same gap** and **different excess**.

---

## Three structural findings

These are independent of the training outcome and are the most solid part of the
run.

**1. Closing triangles does not raise the spectral gap; it lowers it.**
The two families cannot be compared by simply adding edges, because they never
meet. Adding 1,192 triadic edges moves the gap from 0.00312 to 0.00294, while 200
spectral edges raise it to 0.00717.

**2. The gap is not monotone in the triadic fraction, and that is what makes
matching possible.**
Measured on the 3,000-node subgraph at a budget of 2,887 edges:

| triadic fraction | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| spectral gap | .0751 | .0776 | .0563 | .0646 | .0278 | .0466 | .0229 | .0227 | .0105 | .0136 | .0034 |
| triadic excess | +4.72 | +5.62 | +6.22 | +6.81 | +7.33 | +7.85 | +8.36 | +8.93 | +9.52 | +10.06 | +10.82 |

The triadic excess rises cleanly and monotonically. The gap **oscillates**. That is
where the room to match lies.

Practical consequence: **a fine grid is required.** At steps of 0.25 no pair is
found on this subgraph; at steps of 0.1, three are.

**3. Not every graph admits the construction.**
`roman-empire`, mean degree 2.9 with 28.6% of nodes in no triangle, yields no pair:
there are barely any triangles to close. The construction needs triadic substrate.

---

## Training outcome

The same three-layer GCN in both arms, the **ten official Platonov splits**, model
selection on validation.

| pair | edges added | triadic excess A → B | accuracy A | accuracy B | difference | favouring B |
|---|---|---|---|---|---|---|
| 1 | 577 | +5.52 → +6.47 | 0.5796 | 0.5893 | **+0.0097** | 8/10 |
| 2 | 1,385 | +6.31 → +6.86 | 0.5750 | 0.5779 | +0.0029 | 5/10 |
| 3 | 2,887 | +4.72 → +5.62 | 0.5476 | 0.5558 | **+0.0082** | 7/10 |

**Mean +0.0069, all three pairs with the same sign: at fixed spectral gap, more
triadic organisation gives higher accuracy.**

### What this is not

**There is no statistical power.** Three pairs with the same sign have probability
1 in 8 under the null. This is a direction, not evidence.

**The magnitude is small**, 0.007, comparable to seed-to-seed variation.

**And the sign reversed when the setup improved.** An earlier run with random
splits, a single pair and a smaller subgraph gave −0.0058. That is precisely why
the earlier one is not reported as a result.

---

## Why either outcome is worth having

If the direction holds with more pairs and more graphs, the spectral account is
incomplete: methods attributing their gains to the gap would be capturing, in part,
a local-structure effect. That changes the criterion new methods are designed
against.

If it does not hold, the spectral account survives an objection that until now
could not be posed as an experiment.

---

## What would be needed to make this a result

In order of importance:

1. **More pairs.** Ten or fifteen, across several graphs, for a test with meaning.
   The constraint is compute, not method.
2. **Real rewiring methods** rather than the two heuristics used here. FoSR, SDRF,
   BORF and TRIGON all have released code.
3. **Graphs where over-squashing bites**, meaning long-range tasks.
   `amazon-ratings` is heterophilous but not the extreme case.
4. **Matching the degree distribution too**, not only the edge count. The two routes
   may currently differ in degree heterogeneity, and that confound remains open.

Points 2 and 3 require knowing the field, which is why the construction is offered
rather than pursued alone.

---

## Code

Scripts that build the pairs and run the comparison are available on request. The
neuroscience version of the same construction, on pruned k-regular graphs and
addressed to a matched-null request in whole-brain modelling, is published as a
notebook at `github.com/BiomeMakers/OmegaN` under `notebooks/`.

The descriptor used to quantify triadic organisation, and the argument for why it
is not spectrally determined, are in the accompanying preprint:
[arXiv:2609.01633](https://arxiv.org/abs/2609.01633).
