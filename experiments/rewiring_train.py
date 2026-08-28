"""Entrena en los TRES pares emparejados, con las particiones OFICIALES.

Unidad de analisis: el par. Se contrasta a traves de pares, no de semillas.
"""
import numpy as np, scipy.sparse as sp, torch, torch.nn as nn
import torch.nn.functional as Fn, json
from scipy.stats import wilcoxon

pairs = json.load(open("DATA/r4_pairs.json"))
idx = np.load("DATA/r4_idx.npy")
d = np.load("DATA/amazon_ratings.npz", allow_pickle=True)
Y = torch.tensor(d["node_labels"][idx], dtype=torch.long)
X = torch.tensor(d["node_features"][idx], dtype=torch.float)
TR = d["train_masks"][:, idx]; VA = d["val_masks"][:, idx]; TE = d["test_masks"][:, idx]
n, f = X.shape; C = int(Y.max()) + 1
print(f"{n} nodos, {f} caracteristicas, {C} clases, "
      f"{TR.shape[0]} particiones oficiales")


def na(B):
    B = B + sp.eye(B.shape[0])
    k = np.asarray(B.sum(1)).ravel()
    di = sp.diags(1 / np.sqrt(k)); S = (di @ B @ di).tocoo()
    i = torch.tensor(np.vstack([S.row, S.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(i, torch.tensor(S.data, dtype=torch.float),
                                   S.shape).coalesce()


class GCN(nn.Module):
    def __init__(s, f, h, c):
        super().__init__()
        s.a = nn.Linear(f, h); s.b = nn.Linear(h, h); s.c = nn.Linear(h, c)
    def forward(s, A, x):
        x = Fn.dropout(Fn.relu(torch.sparse.mm(A, s.a(x))), .5, s.training)
        x = Fn.dropout(Fn.relu(torch.sparse.mm(A, s.b(x))), .5, s.training)
        return torch.sparse.mm(A, s.c(x))


def run(B, split):
    torch.manual_seed(split)
    A = na(B)
    tr = torch.tensor(np.where(TR[split])[0]); va = torch.tensor(np.where(VA[split])[0])
    te = torch.tensor(np.where(TE[split])[0])
    m = GCN(f, 64, C)
    o = torch.optim.Adam(m.parameters(), lr=.01, weight_decay=5e-4)
    bv = bt = 0
    for e in range(300):
        m.train(); o.zero_grad()
        Fn.cross_entropy(m(A, X)[tr], Y[tr]).backward(); o.step()
        if e % 5 == 0:
            m.eval()
            with torch.no_grad():
                p = m(A, X).argmax(1)
                v = (p[va] == Y[va]).float().mean().item()
                t = (p[te] == Y[te]).float().mean().item()
                if v > bv: bv, bt = v, t
    return bt


print("\npar   exceso A -> B     exact. A   exact. B    dif   a favor de B")
dif_pairs = []
for p in pairs:
    A = sp.load_npz(f"DATA/r4_{p['N']}_A.npz")
    B = sp.load_npz(f"DATA/r4_{p['N']}_B.npz")
    # A es el de menor exceso por convencion
    if p["tA"] > p["tB"]:
        A, B = B, A; p["tA"], p["tB"] = p["tB"], p["tA"]
    aA = np.array([run(A, s) for s in range(10)])
    aB = np.array([run(B, s) for s in range(10)])
    dd = aB - aA
    dif_pairs.append(dd.mean())
    print(f"N={p['N']:>5}  {p['tA']:+.2f} -> {p['tB']:+.2f}   "
          f"{aA.mean():.4f}    {aB.mean():.4f}   {dd.mean():+.4f}   "
          f"{(dd>0).sum()}/10", flush=True)

dp = np.array(dif_pairs)
print(f"\nmedia sobre los {len(dp)} pares: {dp.mean():+.4f}")
print(f"pares con signo positivo: {(dp>0).sum()}/{len(dp)}")
if len(dp) >= 3:
    print("con tres pares no hay potencia para un contraste; se reporta la "
          "direccion y la magnitud, no un p-valor")
