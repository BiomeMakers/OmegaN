"""
DOS BLOQUEANTES DE LA SECCION 11, atacados a la vez.

BLOQUEANTE 13: el umbral de confianza de STRING se fijo en 700 y nunca se movio.
No se ajusto a ojo, pero tampoco se barrio, y un revisor preguntara si el
resultado depende de el. Se barre 400, 550, 700, 850 y 900.

BLOQUEANTE 14: LOBPCG no alcanza la tolerancia pedida en grafos de este tamano,
asi que la coordenada de energia de Fiedler arrastra error numerico. Se recalcula
con mas iteraciones y tolerancia mas floja pero declarada, y se compara el vector
resultante con el de la version original para cuantificar cuanto se movia.
"""
import gzip
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import lobpcg
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
sys.path.insert(0, "..")   # omega_n.py lives at the repo root
from omega_n import triangles_blocked  # noqa: E402

t0 = time.time()
info = pd.read_csv("9606_protein_info_v12_0_txt.gz", sep="\t")
info.columns = [c.lstrip("#") for c in info.columns]
id2sym = dict(zip(info["string_protein_id"], info["preferred_name"]))
d = pd.read_csv("interactions.tsv", sep="\t")
tapp = set(d[d["approved"].astype(str).str.lower() == "true"]["gene_name"]
           .dropna().astype(str).str.upper())

raw = []
with gzip.open("9606_protein_links_v12_0_txt.gz", "rt") as fh:
    fh.readline()
    for line in fh:
        p = line.split()
        raw.append((p[0], p[1], int(p[2])))
print(f"leidas {len(raw)} aristas ({time.time()-t0:.0f}s)", flush=True)


def fiedler(A, k, maxiter, tol, seed=2):
    n = A.shape[0]
    L = (diags(k) - A).tocsr()
    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal((n, 3)); X0[:, 0] = 1.0
    vals, vecs = lobpcg(L, X0, largest=False, maxiter=maxiter, tol=tol)
    order = np.argsort(np.asarray(vals).ravel())
    return vecs[:, order[1]], float(np.asarray(vals).ravel()[order[1]])


def build(cut):
    src = [a for a, b, s in raw if s >= cut]
    dst = [b for a, b, s in raw if s >= cut]
    nodes = sorted(set(src) | set(dst))
    idx = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)
    r = np.fromiter((idx[u] for u in src), dtype=np.int64, count=len(src))
    c = np.fromiter((idx[u] for u in dst), dtype=np.int64, count=len(dst))
    A = csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    A = ((A + A.T) > 0).astype(float)
    A.setdiag(0); A.eliminate_zeros()
    return A, nodes


def features(A, v2):
    n = A.shape[0]
    k = np.asarray(A.sum(1)).ravel(); m = k.sum() / 2
    tri = triangles_blocked(A)
    s1 = A @ k; s2 = A @ (k ** 2)
    E = (s1 ** 2 - s2) / (4 * m)
    exc = (tri - E) / (E + 1.0)
    mu = np.divide(s1, k, out=np.zeros(n), where=k > 0)
    disp = np.log1p(np.maximum(np.divide(s2, k, out=np.zeros(n), where=k > 0) - mu ** 2, 0)
                    / max((k ** 2).mean() / max(k.mean(), 1e-12), 1e-12))
    en = 0.5 * (k * v2 ** 2 - 2 * v2 * (A @ v2) + (A @ (v2 ** 2)))
    inv = np.divide(1.0, k, out=np.zeros(n), where=k > 0)
    P = diags(inv) @ A

    def ppr(x, a, T=20):
        sm = (1 - a) * x.copy(); z = x.copy()
        for t in range(1, T):
            z = P @ z; sm += (1 - a) * (a ** t) * z
        return sm
    F = [k, exc, disp, en]
    for a in (0.5, 0.9):
        F += [ppr(exc, a), ppr(disp, a), ppr(en, a)]
    X = np.nan_to_num(np.column_stack(F))
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
    pr = np.ones(n) / n
    Pt = (diags(inv) @ A).T.tocsr()
    for _ in range(60):
        pr = 0.15 / n + 0.85 * (Pt @ pr)
    B = np.nan_to_num(np.column_stack([k, core, pr, clust]))
    return B, X, k


def ev(M, y, seed=42):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    ap = []
    for tr, te in skf.split(M, y):
        m = RandomForestClassifier(300, min_samples_leaf=5, random_state=seed,
                                   n_jobs=-1,
                                   class_weight="balanced_subsample").fit(M[tr], y[tr])
        ap.append(average_precision_score(y[te], m.predict_proba(M[te])[:, 1]))
    return float(np.mean(ap))


print(f"\n{'corte':>6}{'n':>8}{'grado':>8}{'base%':>8}"
      f"{'bateria':>10}{'Omega-N':>10}{'margen':>9}   [Fiedler]")
for cut in (400, 550, 700, 850, 900):
    A, nodes = build(cut)
    k = np.asarray(A.sum(1)).ravel()
    y = np.array([1 if id2sym.get(u, "").upper() in tapp else 0 for u in nodes])
    v_lo, l_lo = fiedler(A, k, 300, 1e-5)          # como en el paper
    v_hi, l_hi = fiedler(A, k, 3000, 1e-8)         # convergencia forzada
    rho = abs(spearmanr(v_lo, v_hi)[0])
    B, X, _ = features(A, v_hi)                    # se usa el vector CONVERGIDO
    pB, pX = ev(B, y), ev(X, y)
    print(f"{cut:>6}{A.shape[0]:>8}{k.mean():>8.1f}{100*y.mean():>8.1f}"
          f"{pB:>10.4f}{pX:>10.4f}{pX-pB:>+9.4f}   "
          f"lambda2 {l_lo:.4g} -> {l_hi:.4g}, rho(v) {rho:.3f}", flush=True)
