"""
HEAD-TO-HEAD CONTRA EL RIVAL DE CAMPO.

Hasta ahora Omega-N se ha comparado con la BATERIA DE CENTRALIDADES, que no es el
estado del arte. El rival real, publicado en nov-2025, combina seis metricas de
centralidad con embeddings de Node2Vec y los mete en XGBoost. Aqui se reconstruye
ese pipeline y se hace la unica pregunta que decide si hay producto:

    ¿APORTA Omega-N POR ENCIMA del pipeline de centralidades + Node2Vec?

Sobre red CURADA (STRING combinado >= 700), que es el regimen donde el metodo es
util y donde trabaja el sector, asumiendo el sesgo de literatura de forma
declarada: la reclamacion NO es "encuentra biologia nueva" (BioPlex demostro que
sin literatura casi no hay senal), sino "ordena mejor la lista de candidatos".

BRAZOS, todos con el MISMO XGBoost y las mismas particiones:
  1. bateria de centralidades
  2. Node2Vec
  3. bateria + Node2Vec           <- EL RIVAL, el pipeline publicado
  4. Omega-N
  5. bateria + Node2Vec + Omega-N <- LA PREGUNTA

FALSADOR: si el brazo 5 no bate al brazo 3 en AUPRC, Omega-N no aporta sobre el
estado del arte y no hay producto. La comparacion 4 vs 3 es informativa pero no
decisiva: nadie sustituiria su pipeline, lo ampliaria.
"""
import gzip
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.stats import wilcoxon
from gensim.models import Word2Vec
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
sys.path.insert(0, "..")   # omega_n.py lives at the repo root
from omega_n import omega_n, triangles_blocked  # noqa: E402

MIN_SCORE = 700
DIM, WALKS, WALK_LEN = 64, 10, 40
N_REP = 5
t0 = time.time()

info = pd.read_csv("9606_protein_info_v12_0_txt.gz", sep="\t")
info.columns = [c.lstrip("#") for c in info.columns]
id2sym = dict(zip(info["string_protein_id"], info["preferred_name"]))

src, dst = [], []
with gzip.open("9606_protein_links_v12_0_txt.gz", "rt") as fh:
    fh.readline()
    for line in fh:
        p = line.split()
        if int(p[2]) >= MIN_SCORE:
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
print(f"red curada: n={n} grado medio={k.mean():.1f}  ({time.time()-t0:.0f}s)", flush=True)

# ---------- bateria (seis metricas, como el rival)
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
# fuerza y centralidad de autovector por iteracion de potencia
ev = np.ones(n) / np.sqrt(n)
for _ in range(100):
    ev = A @ ev
    ev /= (np.linalg.norm(ev) + 1e-12)
B = np.column_stack([k, core, pr, clust, ev, A @ k])
print(f"bateria lista ({time.time()-t0:.0f}s)", flush=True)

# ---------- Node2Vec (paseos aleatorios de primer orden, p=q=1)
rng = np.random.default_rng(42)
walks = []
for _ in range(WALKS):
    order = rng.permutation(n)
    for start in order:
        w = [start]
        cur = start
        for _ in range(WALK_LEN - 1):
            lo, hi = indptr[cur], indptr[cur + 1]
            if hi <= lo:
                break
            cur = indices[rng.integers(lo, hi)]
            w.append(cur)
        walks.append([str(x) for x in w])
print(f"paseos: {len(walks)} ({time.time()-t0:.0f}s)", flush=True)
w2v = Word2Vec(walks, vector_size=DIM, window=5, min_count=0, sg=1,
               workers=4, epochs=3, seed=42)
E = np.zeros((n, DIM))
for i in range(n):
    key = str(i)
    if key in w2v.wv:
        E[i] = w2v.wv[key]
print(f"Node2Vec listo ({time.time()-t0:.0f}s)", flush=True)

X = omega_n(A)
d = pd.read_csv("interactions.tsv", sep="\t")
tapp = set(d[d["approved"].astype(str).str.lower() == "true"]["gene_name"]
           .dropna().astype(str).str.upper())
y = np.array([1 if id2sym.get(u, "").upper() in tapp else 0 for u in nodes])
print(f"dianas {y.sum()} ({100*y.mean():.1f}%)", flush=True)


def ev_arm(M, yy, tag, seed=42, verbose=True):
    M = np.nan_to_num(np.asarray(M, float))
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    au, ap = [], []
    spw = (yy == 0).sum() / max((yy == 1).sum(), 1)
    for tr, te in skf.split(M, yy):
        m = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08,
                          subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=spw, random_state=seed,
                          n_jobs=4, eval_metric="logloss", tree_method="hist")
        m.fit(M[tr], yy[tr])
        p = m.predict_proba(M[te])[:, 1]
        au.append(roc_auc_score(yy[te], p)); ap.append(average_precision_score(yy[te], p))
    if verbose:
        print(f"   {tag:<38}{M.shape[1]:>4} feat   AUROC {np.mean(au):.4f}"
              f"   AUPRC {np.mean(ap):.4f}", flush=True)
    return np.mean(au), np.mean(ap)


BN = np.column_stack([B, E])
BNX = np.column_stack([B, E, X])
print(f"\n=== RED CURADA, objetivo diana APROBADA (base {100*y.mean():.1f}%), XGBoost")
ev_arm(B, y, "1. bateria (6 centralidades)")
ev_arm(E, y, "2. Node2Vec (64)")
ev_arm(BN, y, "3. bateria + Node2Vec  <- EL RIVAL")
ev_arm(X, y, "4. Omega-N (10)")
ev_arm(BNX, y, "5. rival + Omega-N  <- LA PREGUNTA")

print("\n=== significacion de 5 vs 3, 5 semillas")
rows = []
for s in range(N_REP):
    _, p3 = ev_arm(BN, y, None, 100 + s, False)
    _, p5 = ev_arm(BNX, y, None, 100 + s, False)
    rows.append((p3, p5))
    print(f"   semilla {s+1}  rival {p3:.4f}   rival+Omega-N {p5:.4f}   "
          f"dif {p5-p3:+.4f}", flush=True)
R = np.array(rows)
dif = R[:, 1] - R[:, 0]
print(f"\nAUPRC: rival {R[:,0].mean():.4f}  rival+Omega-N {R[:,1].mean():.4f}  "
      f"dif {dif.mean():+.4f}  gana {int((dif>0).sum())}/{len(dif)}  "
      f"Wilcoxon p={wilcoxon(R[:,1], R[:,0]).pvalue:.4f}")
