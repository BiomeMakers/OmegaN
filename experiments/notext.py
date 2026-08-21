"""
LA PRUEBA DEFINITIVA: red PPI construida SIN el canal de mineria de textos.

Los dos controles anteriores atacaban el sesgo de literatura en la ETIQUETA y en
el GRADO. Quedaba el tercero y mas serio: el score combinado de STRING incluye un
canal de MINERIA DE TEXTOS, asi que parte de las ARISTAS se genera por
co-mencion en articulos, y los genes muy estudiados salen mas conectados.

Aqui la red se reconstruye usando SOLO los canales `experimental` y `database`,
que son evidencia fisica de interaccion y evidencia curada, y se descartan
`textmining`, `coexpression`, `neighborhood`, `fusion` y `cooccurence`.

Se corre el mismo protocolo y ADEMAS el emparejamiento por grado, que es el
control mas duro de los dos anteriores.

FALSADOR: si Omega-N deja de batir a la bateria en AUPRC sobre esta red, el
positivo era literatura y se cierra.
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
from omega_n import omega_n, triangles_blocked, screen  # noqa: E402

SEED = 42
MIN_SCORE = 700          # mismo umbral que antes, ahora sobre el canal combinado limpio
t0 = time.time()

info = pd.read_csv("9606_protein_info_v12_0_txt.gz", sep="\t")
info.columns = [c.lstrip("#") for c in info.columns]
id2sym = dict(zip(info["string_protein_id"], info["preferred_name"]))
id2ann = dict(zip(info["string_protein_id"],
                  info["annotation"].fillna("").astype(str).str.len()))

# canales: 0 p1, 1 p2, 2 neighborhood, 3 fusion, 4 cooccurence, 5 coexpression,
#          6 experimental, 7 database, 8 textmining, 9 combined
src, dst, w = [], [], []
with gzip.open("9606_protein_links_detailed_v12_0_txt.gz", "rt") as fh:
    fh.readline()
    for line in fh:
        p = line.split()
        exp, db = int(p[6]), int(p[7])
        # combinacion de los dos canales limpios, formula de STRING (probabilidad
        # de que al menos uno acierte), reescalada a 0-1000
        s = 1000 * (1 - (1 - exp / 1000.0) * (1 - db / 1000.0))
        if s >= MIN_SCORE:
            src.append(p[0]); dst.append(p[1]); w.append(s)

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
print(f"RED SIN MINERIA DE TEXTOS  ({time.time()-t0:.0f}s)")
print("   cribado:", {kk: (round(v, 4) if isinstance(v, float) else v)
                      for kk, v in screen(A).items()}, flush=True)

tri = triangles_blocked(A)
clust = np.divide(2 * tri, k * (k - 1), out=np.zeros(n), where=k > 1)
deg = k.copy(); core = np.zeros(n); alive = np.ones(n, bool)
indptr, indices = A.indptr, A.indices
kk_ = 0
while alive.any():
    kk_ += 1
    while True:
        rm = np.where(alive & (deg < kk_))[0]
        if rm.size == 0:
            break
        alive[rm] = False; core[rm] = kk_ - 1
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
B = np.column_stack([k, core, pr, clust])
X = omega_n(A)
lit = np.array([id2ann.get(u, 0) for u in nodes], float)
print(f"   features listas ({time.time()-t0:.0f}s)", flush=True)

d = pd.read_csv("interactions.tsv", sep="\t")
tapp = set(d[d["approved"].astype(str).str.lower() == "true"]["gene_name"]
           .dropna().astype(str).str.upper())
y = np.array([1 if id2sym.get(u, "").upper() in tapp else 0 for u in nodes])


def ev(M, yy, tag, sub=None):
    M = np.nan_to_num(np.asarray(M, float))
    if sub is not None:
        M, yy = M[sub], yy[sub]
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    au, ap = [], []
    for tr, te in skf.split(M, yy):
        m = RandomForestClassifier(300, min_samples_leaf=5, random_state=SEED,
                                   n_jobs=-1,
                                   class_weight="balanced_subsample").fit(M[tr], yy[tr])
        p = m.predict_proba(M[te])[:, 1]
        au.append(roc_auc_score(yy[te], p))
        ap.append(average_precision_score(yy[te], p))
    print(f"   {tag:<34}{M.shape[1]:>4} feat   AUROC {np.mean(au):.4f}"
          f"   AUPRC {np.mean(ap):.4f}", flush=True)


print(f"\n=== SIN textmining, objetivo farmaco APROBADO (base {100*y.mean():.1f}%)")
ev(k.reshape(-1, 1), y, "grado solo")
ev(B, y, "bateria")
ev(np.column_stack([B, lit]), y, "bateria + proxy de estudio")
ev(X, y, "Omega-N")
ev(np.column_stack([B, X]), y, "bateria + Omega-N")

rng = np.random.default_rng(SEED)
pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
order = np.argsort(k[neg]); neg_sorted = neg[order]; k_neg = k[neg_sorted]
used = np.zeros(len(neg_sorted), bool); match = []
for i in pos:
    j = np.searchsorted(k_neg, k[i]); best, bestd = -1, None
    for jj in range(max(0, j - 40), min(len(neg_sorted), j + 40)):
        if used[jj]:
            continue
        dd = abs(k_neg[jj] - k[i])
        if bestd is None or dd < bestd:
            best, bestd = jj, dd
    if best >= 0:
        used[best] = True; match.append(neg_sorted[best])
match = np.array(match)
sub = np.concatenate([pos[:len(match)], match])
print(f"\n=== SIN textmining + EMPAREJADO POR GRADO ({len(match)} pares, "
      f"grado {k[pos[:len(match)]].mean():.1f} vs {k[match].mean():.1f})")
ev(k.reshape(-1, 1), y, "grado solo (debe caer)", sub)
ev(B, y, "bateria", sub)
ev(X, y, "Omega-N", sub)
ev(np.column_stack([B, X]), y, "bateria + Omega-N", sub)
