"""
CONTROL DE SESGO DE LITERATURA sobre el positivo de STRING.

PROBLEMA: la etiqueta ("tiene interaccion farmaco-gen conocida") favorece a los
genes mas estudiados, y si la topologia correlaciona con cuanto se ha estudiado
un gen, parte del resultado es artefacto.

PROBLEMA MAYOR QUE NO SE PUEDE ARREGLAR AQUI, y hay que declararlo: el score
combinado de STRING incluye un canal de MINERIA DE TEXTOS, asi que el sesgo esta
tambien en las ARISTAS. Para quitarlo hace falta `protein.links.detailed`, que
trae los canales separados. Con el fichero que tenemos no se puede.

DOS CONTROLES QUE SI SE PUEDEN HACER:

  A. PROXY DE ESTUDIO COMO FEATURE DEL RIVAL. Se usa la longitud de la anotacion
     funcional de STRING como proxy de cuanto se conoce un gen, y se le da AL
     RIVAL (bateria + proxy). Si Omega-N sigue ganando con el rival reforzado, el
     resultado no se explica solo por intensidad de estudio.

  B. EMPAREJAMIENTO POR GRADO (case-control). El grado es el canal principal por
     el que el sesgo de estudio entra en la topologia. Se empareja cada diana con
     un no-diana de grado casi identico y se evalua SOLO sobre el conjunto
     emparejado, donde por construccion el grado ya no discrimina. Si Omega-N
     conserva ventaja ahi, la ventaja no es grado disfrazado.

FALSADOR: si en CUALQUIERA de los dos controles Omega-N deja de batir a la
bateria en AUPRC, el positivo queda en cuarentena y no se cuenta a nadie.
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

info = pd.read_csv("9606_protein_info_v12_0_txt.gz", sep="\t")
info.columns = [c.lstrip("#") for c in info.columns]
id2sym = dict(zip(info["string_protein_id"], info["preferred_name"]))
# proxy de intensidad de estudio: longitud de la anotacion funcional curada
id2ann = dict(zip(info["string_protein_id"],
                  info["annotation"].fillna("").astype(str).str.len()))

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
        nb = np.concatenate([indices[indptr[i]:indptr[i + 1]] for i in rm]) \
            if rm.size else np.array([], dtype=int)
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
print(f"features listas ({time.time()-t0:.0f}s)", flush=True)

d = pd.read_csv("interactions.tsv", sep="\t")
tapp = set(d[d["approved"].astype(str).str.lower() == "true"]["gene_name"]
           .dropna().astype(str).str.upper())
y = np.array([1 if id2sym.get(u, "").upper() in tapp else 0 for u in nodes])

print(f"\ncorrelacion Spearman(proxy de estudio, grado) = "
      f"{pd.Series(lit).corr(pd.Series(k), method='spearman'):+.3f}")
print(f"proxy de estudio medio: dianas {lit[y==1].mean():.0f} vs "
      f"no dianas {lit[y==0].mean():.0f}")


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
    print(f"   {tag:<36}{M.shape[1]:>4} feat   AUROC {np.mean(au):.4f}"
          f"   AUPRC {np.mean(ap):.4f}", flush=True)


print(f"\n=== CONTROL A: proxy de estudio DADO AL RIVAL  (base {100*y.mean():.1f}%)")
ev(lit.reshape(-1, 1), y, "solo el proxy de estudio")
ev(B, y, "bateria")
ev(np.column_stack([B, lit]), y, "bateria + proxy de estudio")
ev(X, y, "Omega-N")
ev(np.column_stack([X, lit]), y, "Omega-N + proxy")

# --- CONTROL B: emparejamiento por grado
rng = np.random.default_rng(SEED)
pos = np.where(y == 1)[0]
neg = np.where(y == 0)[0]
order = np.argsort(k[neg])
neg_sorted = neg[order]
k_neg = k[neg_sorted]
used = np.zeros(len(neg_sorted), bool)
match = []
for i in pos:
    j = np.searchsorted(k_neg, k[i])
    best, bestd = -1, None
    for jj in range(max(0, j - 40), min(len(neg_sorted), j + 40)):
        if used[jj]:
            continue
        dd = abs(k_neg[jj] - k[i])
        if bestd is None or dd < bestd:
            best, bestd = jj, dd
    if best >= 0:
        used[best] = True
        match.append(neg_sorted[best])
match = np.array(match)
sub = np.concatenate([pos[:len(match)], match])
print(f"\n=== CONTROL B: emparejado por GRADO  "
      f"({len(match)} pares; grado medio dianas {k[pos[:len(match)]].mean():.1f} "
      f"vs emparejados {k[match].mean():.1f})")
ev(k.reshape(-1, 1), y, "grado solo (debe caer a ~0.5)", sub)
ev(B, y, "bateria", sub)
ev(X, y, "Omega-N", sub)
ev(np.column_stack([B, X]), y, "bateria + Omega-N", sub)
