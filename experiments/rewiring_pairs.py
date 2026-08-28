"""Montaje corregido, acotado para terminar en una tanda.

Correcciones sobre el primer intento:
  - VARIOS pares emparejados (tres presupuestos de aristas) como unidad de analisis
  - particiones OFICIALES de Platonov en vez de aleatorias
  - mas semillas por par
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import sys, json, time

sys.path.insert(0, "..")
from omega_n import omega_n

RNG = np.random.default_rng(0)
T0 = time.time()


def gap(B):
    n = B.shape[0]
    k = np.asarray(B.sum(1)).ravel()
    di = sp.diags(1 / np.sqrt(np.maximum(k, 1e-12)))
    L = sp.eye(n) - di @ B @ di
    v = eigsh(L, k=3, sigma=-1e-4, which="LM", return_eigenvectors=False,
              maxiter=6000, tol=1e-8)
    return float(np.sort(v)[1])


def fied(B):
    n = B.shape[0]
    k = np.asarray(B.sum(1)).ravel()
    di = sp.diags(1 / np.sqrt(np.maximum(k, 1e-12)))
    L = sp.eye(n) - di @ B @ di
    w, v = eigsh(L, k=3, sigma=-1e-4, which="LM", maxiter=6000, tol=1e-8)
    return v[:, np.argsort(w)[1]]


def tri(B):
    return float(np.mean(omega_n(B)[:, 1]))


def add(B, pairs):
    if not pairs: return B
    n = B.shape[0]
    r = np.array([p[0] for p in pairs]); c = np.array([p[1] for p in pairs])
    E = sp.csr_matrix((np.ones(len(pairs)), (r, c)), shape=(n, n))
    o = sp.csr_matrix(((B + E + E.T) > 0).astype(np.float64))
    o.setdiag(0); o.eliminate_zeros(); return o


def spec_e(B, m):
    v = fied(B); n = B.shape[0]; nb = B.tolil().rows
    cand, sc = [], []
    while len(cand) < max(6000, m * 4):
        i, j = RNG.integers(0, n, 2)
        if i != j and j not in nb[i]:
            cand.append((int(i), int(j))); sc.append(abs(v[i] - v[j]))
    return [cand[t] for t in np.argsort(sc)[::-1][:m]]


def tri_e(B, m):
    B2 = (B @ B).tocoo(); msk = B2.row < B2.col
    r, c, d = B2.row[msk], B2.col[msk], B2.data[msk]
    nb = B.tolil().rows; out = []
    for t in np.argsort(d)[::-1]:
        i, j = int(r[t]), int(c[t])
        if j not in nb[i]:
            out.append((i, j))
            if len(out) >= m: break
    return out


def mixed(B, N, f):
    nt = int(N * f)
    return add(B, tri_e(B, nt) + spec_e(B, N - nt))


# ------------------------------------------------------------------ subgrafo
d = np.load("DATA/amazon_ratings.npz", allow_pickle=True)
E = d["edges"]; Y = d["node_labels"]; FE = d["node_features"]
n0 = len(Y)
B = sp.csr_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n0, n0))
B = ((B + B.T) > 0).astype(np.float64); B.setdiag(0); B.eliminate_zeros()
k = np.asarray(B.sum(1)).ravel()
s0 = int(np.argsort(k)[::-1][30]); seen = {s0}; fr = [s0]; nb = B.tolil().rows
while len(seen) < 3000 and fr:
    nx = []
    for i in fr:
        for j in nb[i]:
            if j not in seen:
                seen.add(j); nx.append(j)
                if len(seen) >= 3000: break
        if len(seen) >= 3000: break
    fr = nx
idx = np.array(sorted(seen)); B0 = sp.csr_matrix(B[idx][:, idx])
nc, lab = sp.csgraph.connected_components(B0, directed=False)
m = lab == np.argmax(np.bincount(lab)); idx = idx[m]; B0 = sp.csr_matrix(B0[m][:, m])
mm = int(B0.nnz / 2)
print(f"subgrafo {B0.shape[0]} nodos, {mm} aristas, grado {B0.sum(1).mean():.2f}",
      flush=True)
np.save("DATA/r4_idx.npy", idx)

pairs = []
for N in (int(.05 * mm), int(.12 * mm), int(.25 * mm)):
    rows = []
    for f in (0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0):
        G = mixed(B0, N, f); rows.append((f, gap(G), tri(G), G))
    best = None
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            rel = abs(rows[i][1] - rows[j][1]) / max(rows[i][1], rows[j][1])
            sep = abs(rows[i][2] - rows[j][2])
            if rel < 0.06 and (best is None or sep > best[0]):
                best = (sep, rel, rows[i], rows[j])
    print("    curva f->(hueco,exceso): " + " ".join(
        f"{r[0]:.1f}:{r[1]:.4f}/{r[2]:+.2f}" for r in rows), flush=True)
    if best:
        sep, rel, a, b = best
        print(f"N={N:>5}: f={a[0]} vs {b[0]} | hueco {a[1]:.5f} vs {b[1]:.5f} "
              f"({rel:.1%}) | exceso {a[2]:+.3f} vs {b[2]:+.3f} (sep {sep:.3f})"
              f"  [{time.time()-T0:.0f}s]", flush=True)
        sp.save_npz(f"DATA/r4_{N}_A.npz", a[3])
        sp.save_npz(f"DATA/r4_{N}_B.npz", b[3])
        pairs.append(dict(N=N, fA=a[0], fB=b[0], gA=a[1], gB=b[1],
                          tA=a[2], tB=b[2], rel=rel, sep=sep))
    else:
        print(f"N={N:>5}: sin par al 6%  [{time.time()-T0:.0f}s]", flush=True)

json.dump(pairs, open("DATA/r4_pairs.json", "w"), indent=1)
print(f"{len(pairs)} pares construidos en {time.time()-T0:.0f}s", flush=True)
