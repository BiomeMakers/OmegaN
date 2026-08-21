"""Head-to-head Omega-N vs bateria de centralidades sobre STRING.

SIN networkx en el camino critico: era el cuello (clustering y core_number en
Python puro sobre 236k aristas se comieron media hora dos veces). Todo por
algebra dispersa.
Sin intermediacion, y esta declarado: la semantica de esta red es de
COOCURRENCIA, no de caminos, asi que no es el rival correcto.
Objetivo: ESTATUS DE DIANA, no esencialidad. Metrica principal AUPRC.
"""
import gzip
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
sys.path.insert(0, "..")   # omega_n.py lives at the repo root
from omega_n import omega_n, triangles_blocked  # noqa: E402

SEED = 42
MIN_SCORE = 700
t0 = time.time()

ids = pd.read_csv("9606_protein_info_v12_0_txt.gz", sep="\t")
id2sym = dict(zip(ids.iloc[:, 0], ids.iloc[:, 1]))

src, dst = [], []
with gzip.open("9606_protein_links_v12_0_txt.gz", "rt") as fh:
    fh.readline()
    for line in fh:
        p = line.split()
        if int(p[2]) >= MIN_SCORE:
            src.append(p[0])
            dst.append(p[1])

nodes = sorted(set(src) | set(dst))
idx = {u: i for i, u in enumerate(nodes)}
n = len(nodes)
r = np.fromiter((idx[u] for u in src), dtype=np.int64, count=len(src))
c = np.fromiter((idx[u] for u in dst), dtype=np.int64, count=len(dst))
A = csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
A = ((A + A.T) > 0).astype(float)
A.setdiag(0)
A.eliminate_zeros()
k = np.asarray(A.sum(1)).ravel()
print(f"n={n} aristas={int(A.nnz/2)} grado medio={k.mean():.1f} ({time.time()-t0:.0f}s)",
      flush=True)

# ---------- bateria, toda por algebra dispersa
tri = triangles_blocked(A)
clust = np.divide(2 * tri, k * (k - 1), out=np.zeros(n), where=k > 1)

# k-core por poda vectorizada sobre CSR
deg = k.copy()
core = np.zeros(n)
alive = np.ones(n, dtype=bool)
indptr, indices = A.indptr, A.indices
kk = 0
while alive.any():
    kk += 1
    while True:
        rm = np.where(alive & (deg < kk))[0]
        if rm.size == 0:
            break
        alive[rm] = False
        core[rm] = kk - 1
        nb = np.concatenate([indices[indptr[i]:indptr[i + 1]] for i in rm])
        if nb.size:
            np.subtract.at(deg, nb, 1)
        deg[rm] = 0
core[core == 0] = 1
print(f"k-core listo ({time.time()-t0:.0f}s)", flush=True)

# PageRank por iteracion de potencia
inv = np.divide(1.0, k, out=np.zeros(n), where=k > 0)
P = (diags(inv) @ A).T.tocsr()
pr = np.ones(n) / n
for _ in range(60):
    pr = 0.15 / n + 0.85 * (P @ pr)

B = np.column_stack([k, core, pr, clust])
print(f"bateria lista ({time.time()-t0:.0f}s)", flush=True)

X = omega_n(A)
print(f"Omega-N listo ({time.time()-t0:.0f}s)", flush=True)

d = pd.read_csv("interactions.tsv", sep="\t")
tset = set(d["gene_name"].dropna().astype(str).str.upper())
tapp = set(d[d["approved"].astype(str).str.lower() == "true"]["gene_name"]
           .dropna().astype(str).str.upper())
y_any = np.array([1 if id2sym.get(u, "").upper() in tset else 0 for u in nodes])
y_app = np.array([1 if id2sym.get(u, "").upper() in tapp else 0 for u in nodes])


def ev(M, y, tag):
    M = np.nan_to_num(np.asarray(M, float))
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    au, ap = [], []
    for tr, te in skf.split(M, y):
        m = RandomForestClassifier(300, min_samples_leaf=5, random_state=SEED,
                                   n_jobs=-1,
                                   class_weight="balanced_subsample").fit(M[tr], y[tr])
        p = m.predict_proba(M[te])[:, 1]
        au.append(roc_auc_score(y[te], p))
        ap.append(average_precision_score(y[te], p))
    print(f"   {tag:<32}{M.shape[1]:>4} feat   AUROC {np.mean(au):.4f}"
          f"   AUPRC {np.mean(ap):.4f}", flush=True)


for y, name in ((y_any, "CUALQUIER interaccion"), (y_app, "farmaco APROBADO")):
    print(f"\n=== objetivo: {name}  (base {100*y.mean():.1f}%)", flush=True)
    ev(k.reshape(-1, 1), y, "grado solo")
    ev(B, y, "bateria (grado,core,PR,clust)")
    ev(X, y, "Omega-N")
    ev(np.column_stack([B, X]), y, "bateria + Omega-N")
