"""
# DATA/ is a placeholder: point it at your own copy of the datasets.

¿EL CRIBADO PREDICE EL RESULTADO?

Si las derrotas caen fuera de un dominio definido por criterios calculables DEL
GRAFO Y DE LA ETIQUETA, pero SIN mirar el rendimiento del descriptor, entonces no
son contraejemplos: son el alcance declarado. Si no ordena, excluirlas seria
racionalizacion a posteriori.

Dos criterios, los dos medibles antes de correr nada:
  A. SUSTRATO TRIADICO: grado medio y fraccion de nodos sin triangulo.
  B. EL GRADO NO DEBE SER LA ETIQUETA: informacion mutua entre grado y etiqueta,
     normalizada por la entropia de la etiqueta.

Se calculan para las OCHO redes y se comprueba si separan las dos derrotas de la
victoria y los cuatro empates.
"""
import numpy as np
import networkx as nx
import pandas as pd
from sklearn.metrics import mutual_info_score

RES = {"tolokers": "victoria", "europe-airports": "empate", "usa-airports": "empate",
       "amazon-ratings": "empate", "questions": "empate",
       "brazil-airports": "DERROTA", "roman-empire": "DERROTA",
       "minesweeper": "nulo"}


def plat(name):
    d = np.load(f"DATA/{name}.npz")
    y = d["node_labels"]; n = len(y)
    G = nx.Graph(); G.add_nodes_from(range(n))
    G.add_edges_from(map(tuple, d["edges"]))
    G.remove_edges_from(nx.selfloop_edges(G))
    return G, y


def air(name):
    G = nx.read_edgelist(f"DATA/{name}-airports.edgelist", nodetype=int)
    lab = pd.read_csv(f"DATA/labels-{name}-airports.txt", sep=r"\s+")
    lab.columns = [c.strip() for c in lab.columns]
    ymap = dict(zip(lab[lab.columns[0]], lab[lab.columns[1]]))
    G = G.subgraph([u for u in G.nodes() if u in ymap]).copy()
    nodes = sorted(G.nodes())
    G = nx.relabel_nodes(G, {u: i for i, u in enumerate(nodes)})
    return G, np.array([ymap[u] for u in nodes])


nets = [("tolokers", lambda: plat("tolokers")),
        ("europe-airports", lambda: air("europe")),
        ("usa-airports", lambda: air("usa")),
        ("amazon-ratings", lambda: plat("amazon_ratings")),
        ("questions", lambda: plat("questions")),
        ("brazil-airports", lambda: air("brazil")),
        ("roman-empire", lambda: plat("roman_empire")),
        ("minesweeper", lambda: plat("minesweeper"))]

print(f"{'red':<18}{'grado':>8}{'%sin tri':>10}{'clust':>8}"
      f"{'IM(k,y)/H(y)':>14}{'resultado':>12}")
rows = []
for name, fn in nets:
    G, y = fn()
    n = G.number_of_nodes()
    k = np.array([G.degree(i) for i in range(n)], float)
    tri = np.array([v for _, v in sorted(nx.triangles(G).items())], float)
    clust = np.divide(2 * tri, k * (k - 1), out=np.zeros(n), where=k > 1)
    kb = pd.qcut(k, 12, labels=False, duplicates="drop")
    im = mutual_info_score(kb, y)
    p = np.bincount(y) / len(y)
    Hy = -sum(q * np.log(q) for q in p if q > 0)
    ratio = im / Hy if Hy > 0 else 0.0
    rows.append((name, k.mean(), 100 * (tri == 0).mean(), clust.mean(), ratio, RES[name]))
    print(f"{name:<18}{k.mean():>8.1f}{100*(tri==0).mean():>9.1f}%{clust.mean():>8.3f}"
          f"{ratio:>14.3f}{RES[name]:>12}", flush=True)

R = pd.DataFrame(rows, columns=["red", "k", "notri", "clust", "im", "res"])
win = R[R.res.isin(["victoria", "empate"])]
los = R[R.res == "DERROTA"]
print(f"\nvictoria+empates: grado {win.k.min():.1f}-{win.k.max():.1f}, "
      f"IM/H {win.im.min():.3f}-{win.im.max():.3f}")
print(f"DERROTAS:         grado {los.k.min():.1f}-{los.k.max():.1f}, "
      f"IM/H {los.im.min():.3f}-{los.im.max():.3f}")
print("\n¿separa una regla de dos condiciones?")
for kmin in (3, 4, 5, 6):
    for immax in (0.45, 0.50, 0.55, 0.60):
        ok = ((win.k >= kmin) & (win.im <= immax)).all() and \
             (~((los.k >= kmin) & (los.im <= immax))).all()
        if ok:
            print(f"   SI: grado medio >= {kmin} Y IM(grado,etiqueta)/H <= {immax}")
