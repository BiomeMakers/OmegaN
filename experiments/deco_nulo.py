"""
EL NULO EMPAREJADO POR HUECO ESPECTRAL QUE EL ARTICULO DECLARA QUE FALTA.

Deco y colegas (Adv Sci 2026, e77103) definen el regimen cuantico-like por el
HUECO ESPECTRAL de grafos k-regulares podados, y admiten dos veces la misma
limitacion: "la correlacion no distingue por si sola la organizacion
cuantico-like del tamano del hueco espectral" y "hace falta un modelo nulo
EMPAREJADO POR EL HUECO ESPECTRAL para establecer que la interferencia refleja
organizacion cuantico-like y no el tamano del hueco".

PREGUNTA PRE-REGISTRADA: ¿existen grafos con el MISMO hueco espectral y
estructura LOCAL distinta? Si el hueco determina la estructura local, el nulo que
piden no se puede construir y no hay nada que aportar. Si no la determina,
Omega-N es la coordenada que los separa, y la razon es teorica: diag(A^3) NO
esta determinado por el espectro (hay pares cospectrales con distinto conteo de
triangulos por nodo), mientras que el hueco espectral SI es un funcional del
espectro.

CONSTRUCCION, con SUS parametros: grafos k-regulares n=40, k=20, podados con
probabilidad p. Ellos usan p=0.2 (QL) y p=0.8 (no-QL).

FALSADOR: si dentro de una banda estrecha de hueco espectral la dispersion de
Omega-N es despreciable frente a la dispersion total, el hueco lo determina todo
y la linea se cierra.
"""
import numpy as np
import networkx as nx
import pandas as pd
from scipy.sparse import csr_matrix

import sys
sys.path.insert(0, "..")   # omega_n.py lives at the repo root
from omega_n import omega_n  # noqa: E402

SEED = 42
N, K = 40, 20
N_GRAPHS = 600
rng = np.random.default_rng(SEED)


def build(p_prune, seed):
    """k-regular podado, como en el articulo."""
    G = nx.random_regular_graph(K, N, seed=int(seed))
    edges = list(G.edges())
    drop = [e for e in edges if rng.random() < p_prune]
    G.remove_edges_from(drop)
    if G.number_of_edges() == 0:
        return None
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    if G.number_of_nodes() < 20:
        return None
    return nx.convert_node_labels_to_integers(G)


def spectral_gap(A):
    """lambda_0 - lambda_1 de la matriz de conectividad, como ellos la definen."""
    ev = np.sort(np.linalg.eigvalsh(A.toarray()))[::-1]
    return float(ev[0] - ev[1])


rows = []
for i in range(N_GRAPHS):
    p = rng.uniform(0.05, 0.85)
    G = build(p, rng.integers(0, 10 ** 6))
    if G is None:
        continue
    n = G.number_of_nodes()
    A = nx.to_scipy_sparse_array(G, nodelist=range(n), format="csr").astype(float)
    gap = spectral_gap(A)
    X = omega_n(A)
    k = np.asarray(A.sum(1)).ravel()
    rows.append(dict(p=p, n=n, gap=gap, grado=k.mean(),
                     exc_med=np.median(X[:, 1]), exc_iqr=np.subtract(*np.percentile(X[:, 1], [75, 25])),
                     disp_med=np.median(X[:, 2]), energia_med=np.median(X[:, 3]),
                     clust=nx.average_clustering(G)))

D = pd.DataFrame(rows)
print(f"grafos generados: {len(D)}")
print(f"hueco espectral: {D.gap.min():.2f} a {D.gap.max():.2f}")
print(f"   (el articulo contrasta p=0.2 frente a p=0.8)\n")

# ¿el hueco determina la estructura local?
D["banda"] = pd.qcut(D.gap, 12, labels=False, duplicates="drop")
print(f"{'coordenada':<16}{'desv TOTAL':>12}{'desv DENTRO de banda':>24}"
      f"{'% no explicado por el hueco':>30}")
for col in ("exc_med", "exc_iqr", "disp_med", "energia_med", "clust"):
    tot = D[col].std()
    dentro = D.groupby("banda")[col].std().mean()
    print(f"{col:<16}{tot:>12.4f}{dentro:>24.4f}{100*dentro/max(tot,1e-12):>29.1f}%")

# el par mas extremo con hueco casi identico
best = None
for b, g in D.groupby("banda"):
    if len(g) < 4:
        continue
    lo = g.loc[g.exc_med.idxmin()]
    hi = g.loc[g.exc_med.idxmax()]
    d = abs(hi.exc_med - lo.exc_med)
    if best is None or d > best[0]:
        best = (d, lo, hi)
if best:
    d, lo, hi = best
    print(f"\nPAR MAS EXTREMO CON HUECO CASI IDENTICO:")
    print(f"   grafo A: hueco {lo.gap:.3f}  exceso triadico mediano {lo.exc_med:+.4f}  "
          f"clustering {lo.clust:.3f}  poda {lo.p:.2f}")
    print(f"   grafo B: hueco {hi.gap:.3f}  exceso triadico mediano {hi.exc_med:+.4f}  "
          f"clustering {hi.clust:.3f}  poda {hi.p:.2f}")
    print(f"   diferencia de hueco: {abs(hi.gap-lo.gap):.4f}   "
          f"diferencia de exceso triadico: {d:.4f}")
