"""SECCION 11: priorizacion de dianas con la etiqueta INDEPENDIENTE (Open Targets).

Por que existe este fichero. Las cifras de la seccion 11 se produjeron en un
cuaderno que no esta en el repositorio, mientras que `h2h2.py` y `bioplex.py`
leen `interactions.tsv`, es decir la etiqueta ANTERIOR (DGIdb). El README promete
que cada numero del articulo se puede trazar hasta el codigo que lo produjo, y esa
seccion no lo cumplia. Este script cierra ese hueco y de paso regenera las cifras
con el solver de Fiedler corregido.

Que hace, en orden:

  1. Etiqueta. Del parquet de Open Targets toma los targetId con
     maxClinicalStage == APPROVAL. Vienen en Ensembl (ENSG), asi que se traducen
     con el fichero de alias de STRING: 19.699 mapeos ENSG -> proteina, que dejan
     1.052 dianas, la cifra del articulo.
  2. Redes. STRING completa, STRING sin mineria de textos, STRING experimental y
     BioPlex. Las tres de STRING salen del fichero DETALLADO, que trae el score
     por canal.
  3. Componente conexa mayor, siempre. Sin esto lambda_2 = 0 y las tres columnas
     fiedler_* quedan inertes.
  4. DOS baterias contra Omega-N, y el emparejado por grado con diez
     repeticiones. La de CUATRO (grado, k-core, PageRank, clustering) es la que
     produjo las cifras publicadas: reproduce el 0,0934 de BioPlex exacto. La de
     SEIS anade centralidad de vector propio y suma de grados de vecinos, es la
     que declara el texto del articulo y es el rival mas exigente. Se publican
     las dos porque la diferencia entre ambas dice cuanto del margen depende de
     la dureza del rival, y porque quien reconstruya el rival por su cuenta usara
     el mas fuerte.

COMPROBACION DE QUE LA RECONSTRUCCION ES CORRECTA. El script imprime grado medio y
tasa base de cada red. Deben salir, segun la tabla publicada:

     STRING completa       grado 29,8   base 6,6%
     STRING experimental   grado 17,7   base 9,3%
     BioPlex               grado 17,0   base 5,5%

La variante "STRING sin texto" que aparecia en la tabla publicada se ha RETIRADO.
Salia de recombinar los canales excluyendo la mineria de textos, y la receta
exacta no consta: reconstruida aqui daba grado 22,5 frente al 20,7 publicado, o
sea otra red. Las otras tres no dependen de ninguna decision de este tipo, salen
directas de una columna del fichero, y sostienen los mismos argumentos: red de
referencia, variante limpia sin mineria de textos, y replica independiente.

Si cuadran, el montaje es el mismo y las diferencias en los margenes son
atribuibles al solver. Si no cuadran, PARAR: la reconstruccion no es fiel y las
cifras no son comparables con las publicadas.

FICHEROS NECESARIOS, en el directorio de trabajo:
     clinical_target.parquet                        (Open Targets)
     9606_protein_aliases_v12_0_txt.gz              (STRING)
     9606_protein_links_detailed_v12_0_txt.gz       (STRING)
     BioPlex_293T_Network_10K_Dec_2019.tsv          (BioPlex)

El detallado se baja de
https://stringdb-downloads.org/download/protein.links.detailed.v12.0/9606.protein.links.detailed.v12.0.txt.gz
Las redes que falten se saltan con un aviso; el script no se cae por ello.
"""
import gzip
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, "..")
from omega_n import omega_n, triangles_blocked, screen  # noqa: E402

SEED, N_REP, MIN_SCORE, PRIOR = 42, 10, 700, 0.041

F_PARQUET = "clinical_target.parquet"
F_ALIASES = "9606_protein_aliases_v12_0_txt.gz"
F_DETAIL = "9606_protein_links_detailed_v12_0_txt.gz"
F_BIOPLEX = "BioPlex_293T_Network_10K_Dec_2019.tsv"


# ----------------------------------------------------------------- etiqueta
def cargar_etiqueta():
    """Devuelve (dianas como proteina STRING, dianas como simbolo de gen)."""
    d = pd.read_parquet(F_PARQUET)
    ensg = set(d[d["maxClinicalStage"] == "APPROVAL"]["targetId"].dropna())
    ens2sp, sp2sym = {}, {}
    with gzip.open(F_ALIASES, "rt") as fh:
        fh.readline()
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            sp, al, so = p
            if so == "Ensembl_gene" and al.startswith("ENSG"):
                ens2sp.setdefault(al, sp)
            elif so in ("Ensembl_HGNC_symbol", "BLAST_UniProt_GN", "Ensembl_HGNC"):
                sp2sym.setdefault(sp, al)
    prot = {ens2sp[e] for e in ensg if e in ens2sp}
    sym = {sp2sym[s].upper() for s in prot if s in sp2sym}
    print(f"etiqueta Open Targets: {len(ensg)} ENSG aprobados -> {len(prot)} "
          f"proteinas STRING -> {len(sym)} simbolos   (el articulo dice 1.052)",
          flush=True)
    return prot, sym


# ----------------------------------------------------------------- redes
def redes_string():
    """Genera (nombre, nodos, A) para las tres variantes de STRING."""
    if not os.path.exists(F_DETAIL):
        print(f"AVISO: falta {F_DETAIL}, me salto las tres redes de STRING",
              flush=True)
        return
    df = pd.read_csv(F_DETAIL, sep=" ")
    variantes = [
        ("STRING completa", df["combined_score"].values),
        ("STRING experimental", df["experimental"].values.astype(float)),
    ]
    for nombre, score in variantes:
        m = score >= MIN_SCORE
        src = df["protein1"].values[m]
        dst = df["protein2"].values[m]
        nodes = sorted(set(src) | set(dst))
        idx = {u: i for i, u in enumerate(nodes)}
        n = len(nodes)
        r = np.fromiter((idx[u] for u in src), np.int64, len(src))
        c = np.fromiter((idx[u] for u in dst), np.int64, len(dst))
        A = csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
        A = ((A + A.T) > 0).astype(float)
        A.setdiag(0)
        A.eliminate_zeros()
        yield nombre, nodes, A, "prot"


def red_bioplex():
    if not os.path.exists(F_BIOPLEX):
        print(f"AVISO: falta {F_BIOPLEX}, me salto BioPlex", flush=True)
        return
    d = pd.read_csv(F_BIOPLEX, sep="\t")[["SymbolA", "SymbolB"]].dropna()
    d = d[d.SymbolA != d.SymbolB]
    nodes = sorted(set(d.SymbolA) | set(d.SymbolB))
    idx = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)
    A = csr_matrix((np.ones(len(d)), (d.SymbolA.map(idx).values,
                                      d.SymbolB.map(idx).values)), shape=(n, n))
    A = ((A + A.T) > 0).astype(float)
    A.setdiag(0)
    A.eliminate_zeros()
    yield "BioPlex", nodes, A, "sym"


def mayor_componente(nodes, A):
    _, lab = connected_components(A, directed=False)
    keep = np.where(lab == np.bincount(lab).argmax())[0]
    if len(keep) == A.shape[0]:
        return nodes, A
    return [nodes[i] for i in keep], csr_matrix(A[keep][:, keep])


# ----------------------------------------------------------------- bateria
def bateria(A):
    """Las seis centralidades, todas por algebra dispersa."""
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
            nb = np.concatenate([indices[indptr[i]:indptr[i + 1]] for i in rm]) \
                if rm.size else np.array([], int)
            if nb.size:
                np.subtract.at(deg, nb, 1)
            deg[rm] = 0
    core[core == 0] = 1

    inv = np.divide(1.0, k, out=np.zeros(n), where=k > 0)
    P = (diags(inv) @ A).T.tocsr()
    pr = np.ones(n) / n
    for _ in range(60):
        pr = 0.15 / n + 0.85 * (P @ pr)

    ev = np.ones(n) / np.sqrt(n)          # centralidad de vector propio
    for _ in range(200):
        ev = A @ ev
        nrm = np.linalg.norm(ev)
        if nrm == 0:
            break
        ev = ev / nrm

    return np.column_stack([k, core, pr, clust, ev, A @ k]), k


# ----------------------------------------------------------------- evaluacion
def ev_split(M, y, tag, sub=None, seed=SEED):
    M = np.nan_to_num(np.asarray(M, float))
    if sub is not None:
        M, y = M[sub], y[sub]
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    au, ap = [], []
    for tr, te in skf.split(M, y):
        m = RandomForestClassifier(300, min_samples_leaf=5, random_state=seed,
                                   n_jobs=-1,
                                   class_weight="balanced_subsample").fit(M[tr], y[tr])
        p = m.predict_proba(M[te])[:, 1]
        au.append(roc_auc_score(y[te], p))
        ap.append(average_precision_score(y[te], p))
    if tag:
        print(f"   {tag:<30}{M.shape[1]:>4} feat   AUROC {np.mean(au):.4f}"
              f"   AUPRC {np.mean(ap):.4f}", flush=True)
    return np.mean(au), np.mean(ap)


def emparejar(y, k, seed):
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


# ----------------------------------------------------------------- principal
def main():
    prot, sym = cargar_etiqueta()
    resumen = []
    fuentes = list(redes_string()) + list(red_bioplex())
    for nombre, nodes, A, tipo in fuentes:
        t0 = time.time()
        nodes, A = mayor_componente(nodes, A)
        diana = prot if tipo == "prot" else sym
        y = np.array([1 if (u if tipo == "prot" else str(u).upper()) in diana
                      else 0 for u in nodes])
        B, k = bateria(A)
        X = omega_n(A)
        print(f"\n### {nombre}: n={A.shape[0]}  grado medio={k.mean():.1f}  "
              f"base={100*y.mean():.1f}%  ({time.time()-t0:.0f}s)", flush=True)
        print("   screen:", {kk: (round(v, 4) if isinstance(v, float) else v)
                             for kk, v in screen(A).items()}, flush=True)
        ev_split(k.reshape(-1, 1), y, "grado solo")
        ev_split(B[:, :4], y, "bateria de 4")
        ev_split(B, y, "bateria de 6")
        ev_split(X, y, "Omega-N (10)")
        ev_split(np.column_stack([B, X]), y, "bateria de 6 + Omega-N")

        # emparejado por grado contra las DOS baterias. La de 4 es la que
        # produjo las cifras publicadas; la de 6 es la que declara el texto y
        # es el rival mas exigente. Se dan ambas para que se vea cuanto del
        # margen depende de la dureza del rival.
        filas = []
        for rep in range(N_REP):
            sub = emparejar(y, k, 1000 + rep)
            _, p4 = ev_split(B[:, :4], y, None, sub, 1000 + rep)
            _, p6 = ev_split(B, y, None, sub, 1000 + rep)
            _, pX = ev_split(X, y, None, sub, 1000 + rep)
            filas.append((p4, p6, pX))
        R = np.array(filas)
        linea = []
        for col, nom in ((0, "de 4"), (1, "de 6")):
            dif = R[:, 2] - R[:, col]
            w = wilcoxon(R[:, 2], R[:, col])
            ee = 1.96 * dif.std(ddof=1) / np.sqrt(len(dif))
            print(f"   EMPAREJADO vs bateria {nom}: rival {R[:,col].mean():.4f}  "
                  f"Omega-N {R[:,2].mean():.4f}  dif {dif.mean():+.4f} "
                  f"[{dif.mean()-ee:+.4f}, {dif.mean()+ee:+.4f}]  "
                  f"gana {int((dif>0).sum())}/{len(dif)}  p={w.pvalue:.5f}",
                  flush=True)
            linea.append(dif.mean())
        resumen.append((nombre, k.mean(), 100 * y.mean(), R[:, 0].mean(),
                        R[:, 1].mean(), R[:, 2].mean(), linea[0], linea[1]))

    print("\n" + "=" * 78)
    print(f"{'red':22} {'grado':>7} {'base':>7} {'bat.4':>8} {'bat.6':>8} "
          f"{'Omega-N':>8} {'vs 4':>8} {'vs 6':>8}")
    for r in resumen:
        print(f"{r[0]:22} {r[1]:>7.1f} {r[2]:>6.1f}% {r[3]:>8.4f} {r[4]:>8.4f} "
              f"{r[5]:>8.4f} {r[6]:>+8.4f} {r[7]:>+8.4f}")
    print("\nContrastar grado medio y base con la tabla publicada antes de usar "
          "estas cifras.")


if __name__ == "__main__":
    main()
