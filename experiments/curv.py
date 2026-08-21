"""
# DATA/ is a placeholder: point it at your own copy of the datasets.

BLOQUEANTE 6: ¿es Omega-N una lectura de la CURVATURA, o son coordenadas
independientes?

Importa porque el marco del que sale Omega-N liga el conteo de triangulos con la
curvatura, asi que el primer lector que conozca ese marco preguntara si el
descriptor no es simplemente curvatura con otro nombre.

Se usan las OCHO redes disponibles en el contenedor (3 de air-traffic y 5 de
Platonov) en lugar de las seis del colab original: son mas, mas grandes y mas
variadas. Se pierde la continuidad presentacional con la figura previa y se
declara.

Curvatura de Ollivier-Ricci con distribucion uniforme sobre vecinos y coste de
transporte por distancia de camino mas corto, resuelto exactamente por programa
lineal en las redes pequenas y sobre una submuestra de aristas en las grandes.
La curvatura POR NODO es la media de la curvatura de sus aristas incidentes.
"""
import time

import numpy as np
import networkx as nx
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr

import sys
sys.path.insert(0, "..")   # omega_n.py lives at the repo root
from omega_n import omega_n  # noqa: E402

SEED = 42
MAX_EDGES = 3000       # submuestra de aristas en redes grandes
rng = np.random.default_rng(SEED)


def orc_edge(G, u, v, dist):
    """Curvatura de Ollivier-Ricci de una arista, por transporte optimo exacto."""
    nu = list(G.neighbors(u))
    nv = list(G.neighbors(v))
    if not nu or not nv:
        return 0.0
    mu = np.ones(len(nu)) / len(nu)
    mv = np.ones(len(nv)) / len(nv)
    C = np.array([[dist(a, b) for b in nv] for a in nu], float)
    n1, n2 = len(nu), len(nv)
    A_eq = np.zeros((n1 + n2, n1 * n2))
    for i in range(n1):
        A_eq[i, i * n2:(i + 1) * n2] = 1
    for j in range(n2):
        A_eq[n1 + j, j::n2] = 1
    b_eq = np.concatenate([mu, mv])
    res = linprog(C.ravel(), A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if not res.success:
        return np.nan
    return 1.0 - res.fun / max(dist(u, v), 1e-12)


def orc_nodes(G, n):
    """Curvatura media por nodo, sobre una submuestra de aristas si hace falta."""
    edges = list(G.edges())
    if len(edges) > MAX_EDGES:
        sel = rng.choice(len(edges), MAX_EDGES, replace=False)
        edges = [edges[i] for i in sel]
    cache = {}

    def dist(a, b):
        if a == b:
            return 0.0
        key = (a, b) if a < b else (b, a)
        if key not in cache:
            try:
                cache[key] = nx.shortest_path_length(G, a, b)
            except nx.NetworkXNoPath:
                cache[key] = 5.0
        return float(cache[key])

    acc = np.zeros(n)
    cnt = np.zeros(n)
    for u, v in edges:
        k = orc_edge(G, u, v, dist)
        if np.isfinite(k):
            acc[u] += k; acc[v] += k
            cnt[u] += 1; cnt[v] += 1
    return np.divide(acc, cnt, out=np.zeros(n), where=cnt > 0), cnt > 0


def load_air(name):
    G = nx.read_edgelist(f"DATA/{name}-airports.edgelist", nodetype=int)
    G.remove_edges_from(nx.selfloop_edges(G))
    G = nx.convert_node_labels_to_integers(G)
    return G


def load_plat(name):
    d = np.load(f"DATA/{name}.npz")
    n = len(d["node_labels"])
    G = nx.Graph(); G.add_nodes_from(range(n))
    G.add_edges_from(map(tuple, d["edges"]))
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


nets = [("brazil-airports", lambda: load_air("brazil")),
        ("europe-airports", lambda: load_air("europe")),
        ("usa-airports", lambda: load_air("usa")),
        ("minesweeper", lambda: load_plat("minesweeper")),
        ("tolokers", lambda: load_plat("tolokers")),
        ("amazon-ratings", lambda: load_plat("amazon_ratings")),
        ("questions", lambda: load_plat("questions")),
        ("roman-empire", lambda: load_plat("roman_empire"))]

t0 = time.time()
print(f"{'red':<18}{'n':>7}{'grado':>7}   "
      f"{'rho(curv, exc.triadico)':>25}{'rho(curv, Omega1)':>20}{'rho(curv, grado)':>18}")
rows = []
for name, fn in nets:
    G = fn()
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    G = nx.convert_node_labels_to_integers(G)
    n = G.number_of_nodes()
    A = nx.to_scipy_sparse_array(G, nodelist=range(n), format="csr").astype(float)
    X = omega_n(A)
    curv, ok = orc_nodes(G, n)
    k = np.asarray(A.sum(1)).ravel()
    r_exc = spearmanr(curv[ok], X[ok, 1])[0]      # exceso triadico
    r_om = spearmanr(curv[ok], X[ok].sum(1))[0]   # combinacion cruda de las 10
    r_k = spearmanr(curv[ok], k[ok])[0]
    rows.append((name, n, k.mean(), r_exc, r_om, r_k))
    print(f"{name:<18}{n:>7}{k.mean():>7.1f}   {r_exc:>25.3f}{r_om:>20.3f}"
          f"{r_k:>18.3f}   ({time.time()-t0:.0f}s)", flush=True)

R = pd.DataFrame(rows, columns=["red", "n", "grado", "exc", "omega", "grado_r"])
print(f"\nrho(curvatura, exceso triadico): media {R.exc.mean():+.3f}, "
      f"rango [{R.exc.min():+.3f}, {R.exc.max():+.3f}], "
      f"cambia de signo en {int((R.exc > 0).sum())} de {len(R)} redes")
print(f"rho(curvatura, grado):           media {R.grado_r.mean():+.3f}, "
      f"rango [{R.grado_r.min():+.3f}, {R.grado_r.max():+.3f}]")
