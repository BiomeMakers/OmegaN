"""
# DATA/ is a placeholder: point it at your own copy of the datasets.

PUNTO 9 DE LA REVISION: la tabla de Twitch de la seccion 2.5 se corrio con la
version de CUATRO coordenadas, anterior a la reformulacion, cuando el descriptor
se llamaba todavia "the field". La etiqueta "Omega-N" en esa tabla no es el objeto
de la Definicion 1.

Se rehace con el descriptor de DIEZ coordenadas y ppr_terms=20, que son los
ajustes del codigo liberado, y con los mismos rivales y el mismo protocolo.
"""
import io
import zipfile

import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix, diags
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

import sys
sys.path.insert(0, "..")   # omega_n.py lives at the repo root
from omega_n import omega_n, triangles_blocked  # noqa: E402

SEED = 42
Z = zipfile.ZipFile("DATA/musae.zip")


def load(tag):
    e = pd.read_csv(io.BytesIO(Z.read(f"MUSAE-master/input/edges/{tag}_edges.csv")))
    t = pd.read_csv(io.BytesIO(Z.read(f"MUSAE-master/input/target/{tag}_target.csv")))
    return e, t


def battery(A):
    n = A.shape[0]
    k = np.asarray(A.sum(1)).ravel()
    tri = triangles_blocked(A)
    clust = np.divide(2 * tri, k * (k - 1), out=np.zeros(n), where=k > 1)
    deg = k.copy(); core = np.zeros(n); alive = np.ones(n, bool)
    indptr, indices = A.indptr, A.indices
    kk = 0
    while alive.any():
        kk += 1
        while True:
            rm = np.where(alive & (deg < kk))[0]
            if rm.size == 0:
                break
            alive[rm] = False; core[rm] = kk - 1
            nb = np.concatenate([indices[indptr[i]:indptr[i + 1]] for i in rm])
            if nb.size:
                np.subtract.at(deg, nb, 1)
            deg[rm] = 0
    core[core == 0] = 1
    inv = np.divide(1.0, k, out=np.zeros(n), where=k > 0)
    P = (diags(inv) @ A).T.tocsr()
    pr = np.ones(n) / n
    for _ in range(60):
        pr = 0.15 / n + 0.85 * (P @ pr)
    return np.column_stack([k, core, pr, clust]), k, clust


def guimera(A, k):
    """Grado intramodular z y coeficiente de participacion, con Louvain."""
    G = nx.from_scipy_sparse_array(A)
    comms = nx.community.louvain_communities(G, seed=SEED)
    lab = np.zeros(A.shape[0], int)
    for c, nodes in enumerate(comms):
        for v in nodes:
            lab[v] = c
    indptr, indices = A.indptr, A.indices
    z = np.zeros(len(k)); p = np.zeros(len(k))
    kin = np.zeros(len(k))
    for i in range(len(k)):
        nb = indices[indptr[i]:indptr[i + 1]]
        kin[i] = (lab[nb] == lab[i]).sum()
        if len(nb):
            cnt = np.bincount(lab[nb], minlength=len(comms))
            p[i] = 1 - ((cnt / len(nb)) ** 2).sum()
    for c in range(len(comms)):
        m = lab == c
        s = kin[m].std()
        z[m] = (kin[m] - kin[m].mean()) / s if s > 0 else 0
    return np.column_stack([z, p])


def ev(X, y):
    X = np.nan_to_num(np.asarray(X, float))
    kf = KFold(5, shuffle=True, random_state=SEED)
    sc = []
    for tr, te in kf.split(X):
        m = RandomForestRegressor(300, min_samples_leaf=5, random_state=SEED,
                                  n_jobs=-1).fit(X[tr], y[tr])
        sc.append(r2_score(y[te], m.predict(X[te])))
    return float(np.mean(sc))


print(f"{'graph':<10}{'n':>8}{'degree':>9}{'battery':>10}{'Guimera':>10}"
      f"{'Omega-N':>10}{'Om+batt':>10}")
rows = []
for tag in ("PTBR", "ES", "ENGB"):
    e, t = load(tag)
    idcol = "new_id" if "new_id" in t.columns else t.columns[0]
    ycol = "views" if "views" in t.columns else t.columns[-1]
    n = int(max(e.max().max(), t[idcol].max())) + 1
    r = e.iloc[:, 0].values; c = e.iloc[:, 1].values
    A = csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    A = ((A + A.T) > 0).astype(float); A.setdiag(0); A.eliminate_zeros()
    y = np.zeros(n); y[t[idcol].values] = np.log1p(t[ycol].values)
    B, k, _ = battery(A)
    GA = guimera(A, k)
    X = omega_n(A)                       # diez coordenadas, ppr_terms=20
    r_deg, r_bat = ev(k.reshape(-1, 1), y), ev(B, y)
    r_ga, r_om = ev(GA, y), ev(X, y)
    r_un = ev(np.column_stack([B, X]), y)
    rows.append((tag, n, r_deg, r_bat, r_ga, r_om, r_un))
    print(f"{tag:<10}{n:>8}{r_deg:>9.3f}{r_bat:>10.3f}{r_ga:>10.3f}"
          f"{r_om:>10.3f}{r_un:>10.3f}", flush=True)

R = np.array([r[2:] for r in rows])
names = ["degree", "battery", "Guimera-Amaral", "Omega-N", "Omega-N + battery"]
print("\nrangos sobre los tres grafos:")
for j, nm in enumerate(names):
    print(f"   {nm:<20}{R[:,j].min():.2f}-{R[:,j].max():.2f}")
