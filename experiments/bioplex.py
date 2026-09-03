"""
REPLICA EN GRAFO INDEPENDIENTE: BioPlex 293T.

Por que este y no HuRI: HuRI es doble hibrido (interacciones binarias) y el 58,7%
de sus nodos no tiene NINGUN triangulo, asi que no hay sustrato que medir. BioPlex
es purificacion por afinidad + espectrometria de masas, detecta CO-COMPLEJOS y por
tanto cierra triangulos: 38,3% sin triangulo, muy cerca del 35,6% de STRING
experimental. El criterio de admision se fijo ANTES de correr nada.

Por que es una replica de verdad: red totalmente independiente de STRING, obtenida
por un experimento sistematico (se marcan miles de cebos sin elegirlos por lo
estudiados que esten), sin mineria de textos y sin curacion.

Se corre el protocolo completo: bateria vs Omega-N, emparejado por grado, y 10
repeticiones para la significacion.
"""
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
sys.path.insert(0, "..")   # omega_n.py lives at the repo root
from omega_n import omega_n, triangles_blocked, screen  # noqa: E402

N_REP = 10
t0 = time.time()

d = pd.read_csv("BioPlex_293T_Network_10K_Dec_2019.tsv", sep="\t")
d = d[["SymbolA", "SymbolB"]].dropna()
d = d[d.SymbolA != d.SymbolB]
nodes = sorted(set(d.SymbolA) | set(d.SymbolB))
idx = {u: i for i, u in enumerate(nodes)}
n = len(nodes)
r = d.SymbolA.map(idx).values
c = d.SymbolB.map(idx).values
A = csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
A = ((A + A.T) > 0).astype(float)
A.setdiag(0)
A.eliminate_zeros()
k = np.asarray(A.sum(1)).ravel()

# ---------- restriccion a la componente mayor
# Sin esto lambda_2 = 0 y las tres columnas fiedler_* quedan inertes: omega_n
# avisa de ello. STRING trae 126 componentes y BioPlex 17, casi todas diminutas.
from scipy.sparse.csgraph import connected_components as _cc  # noqa: E402
_, _lab = _cc(A, directed=False)
_keep = np.where(_lab == np.bincount(_lab).argmax())[0]
if len(_keep) < A.shape[0]:
    A = csr_matrix(A[_keep][:, _keep])
    nodes = [nodes[i] for i in _keep]
    n = len(nodes)
    k = np.asarray(A.sum(1)).ravel()
    print(f"componente mayor: {n} nodos, {int(A.nnz/2)} aristas, "
          f"grado medio {k.mean():.1f}", flush=True)

print("BioPlex:", {kk: (round(v, 4) if isinstance(v, float) else v)
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
print(f"features listas ({time.time()-t0:.0f}s)", flush=True)

g = pd.read_csv("interactions.tsv", sep="\t")
tapp = set(g[g["approved"].astype(str).str.lower() == "true"]["gene_name"]
           .dropna().astype(str).str.upper())
y = np.array([1 if u.upper() in tapp else 0 for u in nodes])
print(f"dianas: {y.sum()} de {n} ({100*y.mean():.1f}%)", flush=True)


def ev(M, yy, tag, sub=None, seed=42):
    M = np.nan_to_num(np.asarray(M, float))
    if sub is not None:
        M, yy = M[sub], yy[sub]
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    au, ap = [], []
    for tr, te in skf.split(M, yy):
        m = RandomForestClassifier(300, min_samples_leaf=5, random_state=seed,
                                   n_jobs=-1,
                                   class_weight="balanced_subsample").fit(M[tr], yy[tr])
        p = m.predict_proba(M[te])[:, 1]
        au.append(roc_auc_score(yy[te], p)); ap.append(average_precision_score(yy[te], p))
    if tag:
        print(f"   {tag:<32}{M.shape[1]:>4} feat   AUROC {np.mean(au):.4f}"
              f"   AUPRC {np.mean(ap):.4f}", flush=True)
    return np.mean(au), np.mean(ap)


print(f"\n=== BioPlex completo (base {100*y.mean():.1f}%)")
ev(k.reshape(-1, 1), y, "grado solo")
ev(B, y, "bateria")
ev(X, y, "Omega-N")
ev(np.column_stack([B, X]), y, "bateria + Omega-N")


def matched(seed):
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    order = np.argsort(k[neg] + rng.normal(0, 1e-6, len(neg)))
    neg_sorted = neg[order]; k_neg = k[neg_sorted]
    used = np.zeros(len(neg_sorted), bool); match = []; keep = []
    for i in pos:
        j = np.searchsorted(k_neg, k[i]); best, bestd = -1, None
        for jj in range(max(0, j - 40), min(len(neg_sorted), j + 40)):
            if used[jj]:
                continue
            dd = abs(k_neg[jj] - k[i])
            if bestd is None or dd < bestd:
                best, bestd = jj, dd
        if best >= 0:
            used[best] = True; match.append(neg_sorted[best]); keep.append(i)
    return np.concatenate([np.array(keep), np.array(match)])


sub0 = matched(1000)
print(f"\n=== BioPlex EMPAREJADO por grado ({len(sub0)//2} pares)")
ev(k.reshape(-1, 1), y, "grado solo (debe caer)", sub0)
ev(B, y, "bateria", sub0)
ev(X, y, "Omega-N", sub0)
ev(np.column_stack([B, X]), y, "bateria + Omega-N", sub0)

rows = []
for rep in range(N_REP):
    s = matched(1000 + rep)
    _, pB = ev(B, y, None, s, 1000 + rep)
    _, pX = ev(X, y, None, s, 1000 + rep)
    rows.append((pB, pX))
    print(f"   rep {rep+1:>2}  bateria {pB:.4f}  Omega-N {pX:.4f}  dif {pX-pB:+.4f}",
          flush=True)
R = np.array(rows)
dif = R[:, 1] - R[:, 0]
w = wilcoxon(R[:, 1], R[:, 0])
print(f"\nAUPRC emparejado: bateria {R[:,0].mean():.4f}  Omega-N {R[:,1].mean():.4f}  "
      f"dif {dif.mean():+.4f}  [{dif.mean()-1.96*dif.std(ddof=1)/np.sqrt(len(dif)):+.4f}, "
      f"{dif.mean()+1.96*dif.std(ddof=1)/np.sqrt(len(dif)):+.4f}]  "
      f"gana {int((dif>0).sum())}/{len(dif)}  Wilcoxon p={w.pvalue:.5f}")
