"""Comprueba si el defecto del solver de Fiedler afectaba a STRING y a BioPlex,
y si en esos grafos la coordenada esta bien definida.

CORRER DESDE experiments/ del repo, con los ficheros de datos donde ya los tienes
para h2h2.py y bioplex.py. No descarga nada: usa los mismos ficheros y los mismos
filtros que esos dos scripts, para que la red construida sea identica.

Lee cuatro cosas por grafo:

  residuo viejo / nuevo   ||L v - lambda v||. Si el viejo es grande (orden 1e-2 o
                          mas) el solver publicado NO devolvia un par propio en
                          ese grafo.
  estab nuevo             correlacion del vector entre dos semillas CON el solver
                          parcheado. Si es < 0,9, lambda_2 esta degenerado y la
                          coordenada no esta definida: es el caso minesweeper /
                          roman-empire, y ahi el parche no arregla nada porque no
                          hay nada que arreglar.
  corr viejo-nuevo        si es ~1, el solver viejo acertaba por suerte en ese
                          grafo y no hay que rehacer nada de lo que se midio con el.
  (l3-l2)/lmax            el criterio de degeneracion. Referencias medidas:
                          minesweeper 8e-19 y roman-empire 1,5e-07 (degenerados),
                          tolokers 5,0e-06, questions 9,9e-06, amazon 1,3e-05
                          (sanos). Umbral propuesto: 1e-06.

Si sale residuo viejo grande y corr viejo-nuevo baja, ese grafo hay que
recalcularlo. Si sale corr ~1, no.
"""
import sys, time, warnings
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import lobpcg, LinearOperator
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, "..")          # omega_n.py en la raiz del repo

MIN_SCORE = 700                   # el mismo corte que h2h2.py
STRING_LINKS = "9606_protein_links_v12_0_txt.gz"
BIOPLEX_TSV = "BioPlex_293T_Network_10K_Dec_2019.tsv"


# ---------------------------------------------------------------- solvers
def fiedler_viejo(A, k, seed=2):
    """El que estaba publicado antes del parche."""
    n = A.shape[0]
    L = (diags(k) - A).tocsr()
    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal((n, 3))
    X0[:, 0] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vals, vecs = lobpcg(L, X0, largest=False, maxiter=300, tol=1e-5)
    return vecs[:, np.argsort(np.asarray(vals).ravel())[1]]


def fiedler_nuevo(A, k, seed=2):
    """El parcheado: nucleo deflactado y precondicionador de Jacobi."""
    n = A.shape[0]
    L = (diags(k) - A).tocsr()
    d = L.diagonal().copy(); d[d <= 0] = 1.0
    M = LinearOperator((n, n), matvec=lambda x: (x.T / d).T, dtype=float)
    Y = np.ones((n, 1)) / np.sqrt(n)
    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal((n, 2)); X0 -= Y @ (Y.T @ X0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vals, vecs = lobpcg(L, X0, Y=Y, M=M, largest=False, maxiter=2000, tol=1e-8)
    return vecs[:, int(np.argmin(np.asarray(vals).ravel()))]


def residuo(A, k, v):
    L = (diags(k) - A).tocsr()
    v = v / np.linalg.norm(v)
    lam = float(v @ (L @ v))
    return np.linalg.norm(L @ v - lam * v), lam


def espectro(A, k, m=3):
    """l2, l3 y lmax, todos con el solver bueno."""
    n = A.shape[0]
    L = (diags(k) - A).tocsr()
    d = L.diagonal().copy(); d[d <= 0] = 1.0
    M = LinearOperator((n, n), matvec=lambda x: (x.T / d).T, dtype=float)
    Y = np.ones((n, 1)) / np.sqrt(n)
    rng = np.random.default_rng(2)
    X0 = rng.standard_normal((n, m + 1)); X0 -= Y @ (Y.T @ X0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vals, _ = lobpcg(L, X0, Y=Y, M=M, largest=False, maxiter=2000, tol=1e-8)
        vm, _ = lobpcg(L, rng.standard_normal((n, 1)), largest=True,
                       maxiter=600, tol=1e-6)
    lo = np.sort(np.asarray(vals).ravel())[:m]
    return lo[0], lo[1], float(np.asarray(vm).ravel()[0])


def mayor_componente(A):
    _, lab = connected_components(A, directed=False)
    keep = np.where(lab == np.bincount(lab).argmax())[0]
    return A[keep][:, keep]


# ---------------------------------------------------------------- cargadores
def cargar_string():
    import gzip
    src, dst = [], []
    with gzip.open(STRING_LINKS, "rt") as fh:
        fh.readline()
        for line in fh:
            p = line.split()
            if int(p[2]) >= MIN_SCORE:
                src.append(p[0]); dst.append(p[1])
    nodes = sorted(set(src) | set(dst))
    idx = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)
    r = np.fromiter((idx[u] for u in src), np.int64, len(src))
    c = np.fromiter((idx[u] for u in dst), np.int64, len(dst))
    A = csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    A = ((A + A.T) > 0).astype(float); A.setdiag(0); A.eliminate_zeros()
    return A


def cargar_bioplex():
    d = pd.read_csv(BIOPLEX_TSV, sep="\t")[["SymbolA", "SymbolB"]].dropna()
    d = d[d.SymbolA != d.SymbolB]
    nodes = sorted(set(d.SymbolA) | set(d.SymbolB))
    idx = {u: i for i, u in enumerate(nodes)}
    n = len(nodes)
    A = csr_matrix((np.ones(len(d)), (d.SymbolA.map(idx).values,
                                      d.SymbolB.map(idx).values)), shape=(n, n))
    A = ((A + A.T) > 0).astype(float); A.setdiag(0); A.eliminate_zeros()
    return A


# ---------------------------------------------------------------- salida
hdr = (f"{'grafo':10} {'n':>7} {'<k>':>6} {'res viejo':>10} {'res nuevo':>10} "
       f"{'estab nuevo':>11} {'corr v-n':>9} {'(l3-l2)/lmax':>13} {'veredicto':>26}")
print(hdr, flush=True)
print("-" * len(hdr), flush=True)

for nombre, cargador in (("STRING", cargar_string), ("BioPlex", cargar_bioplex)):
    try:
        t0 = time.time()
        A = mayor_componente(cargador())
    except FileNotFoundError as err:
        print(f"{nombre:10} fichero no encontrado: {err.filename}", flush=True)
        continue
    k = np.asarray(A.sum(1)).ravel()
    vv = fiedler_viejo(A, k)
    v1, v2 = fiedler_nuevo(A, k, 2), fiedler_nuevo(A, k, 7)
    rv, _ = residuo(A, k, vv)
    rn, _ = residuo(A, k, v1)
    est = abs(np.corrcoef(v1, v2)[0, 1])
    cc = abs(np.corrcoef(v1, vv)[0, 1])
    l2, l3, lmax = espectro(A, k)
    gap = (l3 - l2) / lmax

    if est < 0.9:
        ver = "degenerado: retirar columnas"
    elif cc > 0.99:
        ver = "el viejo ya acertaba"
    else:
        ver = "RECALCULAR con el parche"
    print(f"{nombre:10} {A.shape[0]:>7} {k.mean():>6.1f} {rv:>10.2e} {rn:>10.2e} "
          f"{est:>11.4f} {cc:>9.4f} {gap:>13.2e} {ver:>26}  ({time.time()-t0:.0f}s)",
          flush=True)
