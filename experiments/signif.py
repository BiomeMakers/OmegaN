"""
SIGNIFICACION del margen de Omega-N sobre la bateria, en el escenario MAS DURO:
red SIN mineria de textos Y emparejada por grado.

Diseno: 10 REPETICIONES independientes de validacion cruzada estratificada de 5
pliegues, cada una con su propia semilla de particion Y su propio emparejamiento
por grado (asi la aleatoriedad del emparejamiento tambien entra en el error).
Se promedia dentro de cada repeticion y se contrasta emparejado sobre las 10.

Por que asi: los pliegues dentro de una repeticion NO son independientes, asi que
contrastar sobre los 50 pliegues inflaria la significacion. Sobre 10 repeticiones
el contraste es honesto aunque tenga menos potencia.

Se reporta Wilcoxon emparejado y la diferencia media con su intervalo.
"""
import gzip
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
from omega_n import omega_n, triangles_blocked  # noqa: E402

MIN_SCORE = 700
N_REP = 10
t0 = time.time()

info = pd.read_csv("9606_protein_info_v12_0_txt.gz", sep="\t")
info.columns = [c.lstrip("#") for c in info.columns]
id2sym = dict(zip(info["string_protein_id"], info["preferred_name"]))

src, dst = [], []
with gzip.open("9606_protein_links_detailed_v12_0_txt.gz", "rt") as fh:
    fh.readline()
    for line in fh:
        p = line.split()
        exp, db = int(p[6]), int(p[7])
        s = 1000 * (1 - (1 - exp / 1000.0) * (1 - db / 1000.0))
        if s >= MIN_SCORE:
            src.append(p[0]); dst.append(p[1])

nodes = sorted(set(src) | set(dst))
idx = {u: i for i, u in enumerate(nodes)}
n = len(nodes)
r = np.fromiter((idx[u] for u in src), dtype=np.int64, count=len(src))
c = np.fromiter((idx[u] for u in dst), dtype=np.int64, count=len(dst))
A = csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
A = ((A + A.T) > 0).astype(float)
A.setdiag(0); A.eliminate_zeros()
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
B = np.column_stack([k, core, pr, clust])
X = omega_n(A)

d = pd.read_csv("interactions.tsv", sep="\t")
tapp = set(d[d["approved"].astype(str).str.lower() == "true"]["gene_name"]
           .dropna().astype(str).str.upper())
y = np.array([1 if id2sym.get(u, "").upper() in tapp else 0 for u in nodes])
print(f"red limpia: n={n} grado medio={k.mean():.1f}  dianas={y.sum()} "
      f"({100*y.mean():.1f}%)   ({time.time()-t0:.0f}s)", flush=True)


def matched(seed):
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    rng.shuffle(pos)
    order = np.argsort(k[neg] + rng.normal(0, 1e-6, len(neg)))
    neg_sorted = neg[order]; k_neg = k[neg_sorted]
    used = np.zeros(len(neg_sorted), bool); match = []
    keep = []
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


def score(M, yy, seed):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    au, ap = [], []
    for tr, te in skf.split(M, yy):
        m = RandomForestClassifier(300, min_samples_leaf=5, random_state=seed,
                                   n_jobs=-1,
                                   class_weight="balanced_subsample").fit(M[tr], yy[tr])
        p = m.predict_proba(M[te])[:, 1]
        au.append(roc_auc_score(yy[te], p)); ap.append(average_precision_score(yy[te], p))
    return np.mean(au), np.mean(ap)


rows = []
for rep in range(N_REP):
    sub = matched(1000 + rep)
    yy = y[sub]
    aB, pB = score(np.nan_to_num(B[sub]), yy, 1000 + rep)
    aX, pX = score(np.nan_to_num(X[sub]), yy, 1000 + rep)
    rows.append((aB, pB, aX, pX))
    print(f"   rep {rep+1:>2}/{N_REP}  bateria {pB:.4f}   Omega-N {pX:.4f}   "
          f"dif {pX-pB:+.4f}   ({time.time()-t0:.0f}s)", flush=True)

R = np.array(rows)
for j, name in ((1, "AUPRC"), (0, "AUROC")):
    b = R[:, j]; x = R[:, j + 2]
    dif = x - b
    w = wilcoxon(x, b)
    print(f"\n{name}: bateria {b.mean():.4f}  Omega-N {x.mean():.4f}  "
          f"dif media {dif.mean():+.4f}  [{dif.mean()-1.96*dif.std(ddof=1)/np.sqrt(len(dif)):+.4f}, "
          f"{dif.mean()+1.96*dif.std(ddof=1)/np.sqrt(len(dif)):+.4f}]  "
          f"gana {int((dif>0).sum())}/{len(dif)}  Wilcoxon p={w.pvalue:.5f}")
