"""
omega_n.py — Reference implementation of the Omega-N node descriptor.

Omega-N is the node-level counterpart of the scalar Omega-S index. It produces
ten features per node from the graph alone: no node attributes, no training, no
embeddings.

The four factors, each expressed as a local excess over the SAME configuration
null used by the scalar index, plus a multiscale smoothing:

  1. degree
  2. triadic excess over the configuration null
  3. neighbour-degree dispersion relative to its null expectation
  4. the node's term in the Dirichlet energy of the Fiedler vector
     (this one sums exactly to the algebraic connectivity)

Factors 2-4 are additionally smoothed with a personalized-PageRank operator at
alpha in {0.5, 0.9}, following the multiscale localization of Peel, Delvenne and
Lambiotte (PNAS 115(16):4057, 2018). Total: 4 + 3*2 = 10 features.

USE CONDITIONS, measured rather than assumed. Read these before applying it:

  (a) The graph must have triadic substrate. On chain-like graphs (mean degree
      around 3) the descriptor has almost nothing to measure and loses to
      recursive feature aggregation.
  (b) The target must be exogenous to connectivity. Where the label is close to
      a monotone function of the degree, the degree is the right answer and null
      calibration removes signal rather than noise.
  (c) The semantics must be co-occurrence, not paths. For reachability questions
      betweenness is the correct measure.
  (d) The descriptor contributes most when node attributes are weak or generic,
      and little when they are rich and engineered for the task.

Complexity is dominated by the triangle count. For dense graphs use
`triangles_blocked`, which never materialises a full n-by-n product.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, diags, issparse
from scipy.sparse.linalg import lobpcg

EPS = 1e-12

__all__ = ["omega_n", "feature_names", "triangles_blocked"]


def feature_names() -> list[str]:
    base = ["degree", "triadic_excess", "neighbour_dispersion", "fiedler_energy"]
    out = list(base)
    for a in (0.5, 0.9):
        out += [f"{n}_ppr{a}" for n in base[1:]]
    return out


def triangles_blocked(A: csr_matrix, block: int = 400) -> np.ndarray:
    """Per-node triangle count, computed in row blocks.

    Equivalent to ((A @ A).multiply(A)).sum(1) / 2 but bounded in memory, which
    matters on dense graphs: on a graph of mean degree 700 the full product does
    not fit, and using a sparser relation instead changes the regime entirely.
    """
    n = A.shape[0]
    out = np.zeros(n)
    for s in range(0, n, block):
        e = min(s + block, n)
        R = A[s:e]
        out[s:e] = np.asarray((R @ A).multiply(R).sum(1)).ravel()
    return out / 2.0


def _fiedler(A: csr_matrix, k: np.ndarray, seed: int = 2) -> np.ndarray:
    """Fiedler vector via LOBPCG.

    Deliberately not shift-invert: the LU factorisation of a large sparse
    Laplacian can exhaust memory. LOBPCG may not reach the requested tolerance;
    the resulting eigenvalue error is a few percent and does not move the
    downstream features materially, but it is worth knowing.
    """
    n = A.shape[0]
    L = (diags(k) - A).tocsr()
    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal((n, 3))
    X0[:, 0] = 1.0
    vals, vecs = lobpcg(L, X0, largest=False, maxiter=300, tol=1e-5)
    order = np.argsort(np.asarray(vals).ravel())
    return vecs[:, order[1]]


def _ppr(P: csr_matrix, x: np.ndarray, alpha: float, terms: int = 20) -> np.ndarray:
    """Truncated personalized-PageRank smoothing of a node signal.

    The truncation length matters at the third decimal: the paper's node
    classification tables were produced with 25 terms and this default is 20
    (on `tolokers`, 0.785 against 0.7763 AUROC). Pass ``terms=25`` through
    ``omega_n(..., ppr_terms=25)`` to reproduce those tables exactly.
    """
    acc = (1 - alpha) * x.copy()
    z = x.copy()
    for t in range(1, terms):
        z = P @ z
        acc += (1 - alpha) * (alpha ** t) * z
    return acc


def omega_n(A, block: int = 400, alphas=(0.5, 0.9), seed: int = 2,
            ppr_terms: int = 20) -> np.ndarray:
    """Compute the Omega-N descriptor.

    Parameters
    ----------
    A : scipy sparse matrix or dense array
        Symmetric adjacency matrix, unweighted, zero diagonal.
    block : int
        Row-block size for the triangle count.
    alphas : tuple of float
        Personalized-PageRank scales. Empty tuple gives the 4 raw factors only.
    ppr_terms : int
        Truncation length of the PPR smoothing. Use 25 to reproduce the paper's
        node-classification tables; 20 is the cheaper default.

    Returns
    -------
    ndarray of shape (n_nodes, 4 + 3*len(alphas))
        Column order given by ``feature_names()``.
    """
    A = csr_matrix(A) if not issparse(A) else A.tocsr()
    A = A.astype(float)
    A.setdiag(0)
    A.eliminate_zeros()
    n = A.shape[0]

    # The Fiedler coordinate is only meaningful on a connected graph: if the graph
    # has several components, lambda_2 is zero and the eigenvector degenerates into
    # a component indicator. We warn rather than silently returning a dead
    # coordinate, because that failure is invisible in the output.
    from scipy.sparse.csgraph import connected_components
    ncomp, _lab = connected_components(A, directed=False)
    if ncomp > 1:
        import warnings
        warnings.warn(
            f"graph has {ncomp} connected components: lambda_2 is zero and the "
            "Fiedler-energy coordinate will be inert. Restrict to the largest "
            "component first.", RuntimeWarning)

    k = np.asarray(A.sum(1)).ravel()
    m = k.sum() / 2.0
    if m <= 0:
        raise ValueError("empty graph")

    # --- factor 2: triadic excess over the configuration null.
    # E[t_i] = (s1_i^2 - s2_i) / 4m with s1, s2 the first two neighbour-degree
    # moments. The +1 in the denominator keeps the ratio finite on nodes whose
    # null expectation is near zero.
    tri = triangles_blocked(A, block)
    s1 = A @ k
    s2 = A @ (k ** 2)
    e_tri = (s1 ** 2 - s2) / (4.0 * m)
    triadic_excess = (tri - e_tri) / (e_tri + 1.0)

    # --- factor 3: neighbour-degree dispersion over its null expectation.
    # The literal fourth factor of the scalar index, (k_i - kbar)^2, is a
    # function of the degree alone and vanishes at the mean degree, which makes
    # the naive contraction diverge on the most ordinary nodes. This replaces it.
    mu = np.divide(s1, k, out=np.zeros(n), where=k > 0)
    var_nb = np.maximum(np.divide(s2, k, out=np.zeros(n), where=k > 0) - mu ** 2, 0.0)
    null_scale = (k ** 2).mean() / max(k.mean(), EPS)
    dispersion = np.log1p(var_nb / max(null_scale, EPS))

    # --- factor 4: Dirichlet energy of the Fiedler vector, per node.
    # Sums exactly to lambda_2, so the localization is canonical rather than
    # chosen for convenience.
    v2 = _fiedler(A, k, seed)
    energy = 0.5 * (k * v2 ** 2 - 2.0 * v2 * (A @ v2) + (A @ (v2 ** 2)))

    cols = [k, triadic_excess, dispersion, energy]
    if alphas:
        inv = np.divide(1.0, k, out=np.zeros(n), where=k > 0)
        P = diags(inv) @ A
        for a in alphas:
            cols += [_ppr(P, triadic_excess, a, ppr_terms),
                     _ppr(P, dispersion, a, ppr_terms),
                     _ppr(P, energy, a, ppr_terms)]
    return np.nan_to_num(np.column_stack(cols))


def screen(A, y=None) -> dict:
    """Cheap pre-flight check against the applicability domain.

    Run this before anything else. Two statistics decide whether the descriptor
    is expected to help, and neither needs the descriptor to be computed:

      S1  mean degree: a chain-like graph gives the triadic excess nothing to
          measure.
      S2  I(degree; label) / H(label), returned only when ``y`` is given: where
          the label is close to a function of the degree, use the degree.

    On the eight networks of the paper the rule "mean degree >= 4 and
    I(k;y)/H(y) <= 0.5" separated the two losses from the win and the four ties
    without error, with both gaps close to a factor of two. That rule was fitted
    on those eight networks and has not been confirmed out of sample: treat the
    verdict as a prior, not a guarantee.
    """
    A = csr_matrix(A) if not issparse(A) else A.tocsr()
    k = np.asarray(A.sum(1)).ravel()
    tri = triangles_blocked(A.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.divide(2 * tri, k * (k - 1), out=np.zeros(len(k)), where=k > 1)
    out = {
        "n_nodes": int(A.shape[0]),
        "n_edges": int(A.nnz // 2),
        "mean_degree": float(k.mean()),
        "frac_no_triangle": float((tri == 0).mean()),
        "mean_clustering": float(c.mean()),
        "frac_isolated": float((k == 0).mean()),
    }
    reasons = []
    if k.mean() < 4:
        reasons.append("S1 fails: mean degree below 4, no triadic substrate")
    if y is not None:
        import pandas as pd
        from sklearn.metrics import mutual_info_score
        yy = np.asarray(y).ravel()
        kb = pd.qcut(k, 12, labels=False, duplicates="drop")
        p = np.bincount(yy) / len(yy)
        h = -sum(q * np.log(q) for q in p if q > 0)
        ratio = float(mutual_info_score(kb, yy) / h) if h > 0 else 0.0
        out["degree_label_mi_ratio"] = ratio
        if ratio > 0.5:
            reasons.append("S2 fails: the label is largely a function of the "
                           "degree, so use the degree")
    out["verdict"] = "; ".join(reasons) if reasons else "inside the stated domain"
    return out
