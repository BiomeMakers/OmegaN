"""
# DATA/ is a placeholder: point it at your own copy of the datasets.

CURVATURA EN LAS CUATRO REDES GRANDES.

La version exacta calculaba caminos minimos par a par y en tolokers eso son
decenas de millones de consultas. Atajo, y es exacto salvo en un caso declarado:

para una arista (u,v) solo hacen falta distancias entre vecinos de u y vecinos de
v, y como u~v esas distancias valen casi siempre 0, 1, 2 o 3. Se calculan por
adyacencia directa: 0 si coinciden, 1 si son adyacentes, 2 si comparten vecino, y
3 en otro caso. **El 3 es una cota: si la distancia real fuera mayor, se subestima.**
En grafos con diametro pequeno, que es el caso de los cuatro, el sesgo es minimo y
va en la misma direccion para todos los nodos.

Vecindarios acotados a 40 vecinos muestreados, aristas acotadas a 800 por red.
El transporte optimo sigue siendo exacto: programa lineal por arista.
"""
import time

import numpy as np
import networkx as nx
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr

import sys
sys.path.insert(0, "..")   # omega_n.py lives at the repo root
from omega_n import omega_n  # noqa: E402

SEED = 42
MAX_EDGES = 800
MAX_NB = 40
rng = np.random.default_rng(SEED)


def dist_matrix(A, A2, nu, nv):
    """Distancias 0/1/2/3 entre dos listas de nodos, por adyacencia."""
    n1, n2 = len(nu), len(nv)
    D = np.full((n1, n2), 3.0)
    sub1 = A[nu][:, nv].toarray()
    sub2 = A2[nu][:, nv].toarray()
    D[sub2 > 0] = 2.0
    D[sub1 > 0] = 1.0
    for i, a in enumerate(nu):
        for j, b in enumerate(nv):
            if a == b:
                D[i, j] = 0.0
    return D


def orc(A, A2, indptr, indices, u, v):
    nu = indices[indptr[u]:indptr[u + 1]]
    nv = indices[indptr[v]:indptr[v + 1]]
    if len(nu) == 0 or len(nv) == 0:
        return np.nan
    if len(nu) > MAX_NB:
        nu = rng.choice(nu, MAX_NB, replace=False)
    if len(nv) > MAX_NB:
        nv = rng.choice(nv, MAX_NB, replace=False)
    nu = np.sort(nu); nv = np.sort(nv)
    C = dist_matrix(A, A2, nu, nv)
    n1, n2 = len(nu), len(nv)
    mu = np.ones(n1) / n1
    mv = np.ones(n2) / n2
    A_eq = np.zeros((n1 + n2, n1 * n2))
    for i in range(n1):
        A_eq[i, i * n2:(i + 1) * n2] = 1
    for j in range(n2):
        A_eq[n1 + j, j::n2] = 1
    res = linprog(C.ravel(), A_eq=A_eq, b_eq=np.concatenate([mu, mv]),
                  bounds=(0, None), method="highs")
    return 1.0 - res.fun if res.success else np.nan


def load(name):
    d = np.load(f"DATA/{name}.npz")
    n = len(d["node_labels"])
    G = nx.Graph(); G.add_nodes_from(range(n))
    G.add_edges_from(map(tuple, d["edges"]))
    G.remove_edges_from(nx.selfloop_edges(G))
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    G = nx.convert_node_labels_to_integers(G)
    return G


t0 = time.time()
print(f"{'red':<16}{'n':>7}{'grado':>7}{'aristas usadas':>16}"
      f"{'rho(curv, exc)':>16}{'rho(curv, grado)':>18}")
for name in ("tolokers", "amazon_ratings", "questions", "roman_empire"):
    G = load(name)
    n = G.number_of_nodes()
    A = nx.to_scipy_sparse_array(G, nodelist=range(n), format="csr").astype(float)
    A2 = (A @ A).tocsr()
    indptr, indices = A.indptr, A.indices
    X = omega_n(A)
    edges = np.array(list(G.edges()))
    if len(edges) > MAX_EDGES:
        edges = edges[rng.choice(len(edges), MAX_EDGES, replace=False)]
    acc = np.zeros(n); cnt = np.zeros(n)
    for u, v in edges:
        k = orc(A, A2, indptr, indices, int(u), int(v))
        if np.isfinite(k):
            acc[u] += k; acc[v] += k; cnt[u] += 1; cnt[v] += 1
    ok = cnt > 0
    curv = np.divide(acc, cnt, out=np.zeros(n), where=ok)
    k = np.asarray(A.sum(1)).ravel()
    r_exc = spearmanr(curv[ok], X[ok, 1])[0]
    r_k = spearmanr(curv[ok], k[ok])[0]
    print(f"{name:<16}{n:>7}{k.mean():>7.1f}{len(edges):>16}"
          f"{r_exc:>16.3f}{r_k:>18.3f}   ({time.time()-t0:.0f}s, "
          f"{int(ok.sum())} nodos)", flush=True)
