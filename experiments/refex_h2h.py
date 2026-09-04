"""ReFeX contra Omega-N, en el arnes de `reproduce.py`.

POR QUE EXISTE. La tabla de dominio de aplicabilidad compara Omega-N con ReFeX,
pero ReFeX no estaba en el repositorio: sus cifras venian de codigo que no se
publico, asi que no se podia comprobar que rival y descriptor se hubieran medido
en el mismo arnes. Al regenerar Omega-N con el solver corregido, dos casos
declarados empate quedaron en duda, y un empate no se reclasifica midiendo un solo
lado. Este fichero implementa el rival y lo corre con las mismas particiones
oficiales, el mismo clasificador y la misma metrica.

ReFeX (Henderson et al., KDD 2011), recursive feature extraction:

  1. Caracteristicas locales por nodo: grado, aristas dentro del egonet y aristas
     que salen del egonet.
  2. Agregacion recursiva: en cada nivel se anaden a cada nodo la SUMA y la MEDIA
     de las caracteristicas de sus vecinos en el nivel anterior.
  3. Poda: se descartan las columnas que quedan casi perfectamente correlacionadas
     con otra ya aceptada, que es lo que evita la explosion combinatoria.

Se barre la profundidad de 2 a 5 y se reporta la MEJOR, que es como esta declarado
en el articulo ("ReFeX, best depth"). Dar al rival su mejor profundidad y quedarse
con su mejor cifra es lo correcto: si aun asi pierde, el resultado es del
descriptor y no del ajuste. Como referencia de que la implementacion no lo
debilita, en `tolokers` alcanza 0.7702, por encima del rango 0.761-0.769 que
reporta el articulo.

AVISO DE VERSIONES, y no es menor. La cifra de Omega-N en `amazon-ratings` es
0.4713 con scikit-learn 1.6.1 y 0.4403 con la 1.8.0, con identicas
caracteristicas, particiones y semilla: se comprobo que las diez columnas del
descriptor tienen la misma huella en las dos maquinas, asi que la diferencia es
enteramente del clasificador. Ese salto es del orden del margen que separa a
Omega-N de sus rivales en esa red. Correr esto con una version distinta de la que
produjo las tablas no compara lo que parece comparar.

USO:
    python3 refex_h2h.py                          # las cinco redes
    python3 refex_h2h.py tolokers questions       # solo las indicadas
    python3 refex_h2h.py --root DATA              # si los .npz estan en otra carpeta

Coste orientativo: en una sola CPU, `amazon-ratings` y `questions` pasan de media
hora cada una. Con varios nucleos baja mucho, porque el bosque paraleliza.
"""
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.insert(0, "..")
from omega_n import omega_n  # noqa: E402

PLATONOV = ["minesweeper", "tolokers", "amazon_ratings", "questions", "roman_empire"]
BINARY = {"tolokers", "questions", "minesweeper"}
DEPTHS = (2, 3, 4, 5)
TOL_PODA = 0.99          # correlacion por encima de la cual se descarta la columna
SEED = 42
N_JOBS = -1


def cargar(name, root="."):
    d = np.load(f"{root}/{name}.npz")
    y = d["node_labels"]
    n = len(y)
    e = d["edges"]
    A = csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    A = ((A + A.T) > 0).astype(float)
    A.setdiag(0)
    A.eliminate_zeros()
    return A, y, d["train_masks"], d["test_masks"]


def locales(A):
    """Grado, aristas internas del egonet y aristas salientes del egonet."""
    k = np.asarray(A.sum(1)).ravel()
    tri = np.asarray((A @ A).multiply(A).sum(1)).ravel() / 2.0
    internas = tri + k
    salientes = (A @ k) - 2 * tri - k
    return np.column_stack([k, internas, salientes])


def podar(F, tol=TOL_PODA):
    """Descarta columnas casi identicas a otra ya aceptada."""
    F = np.nan_to_num(F)
    F = F[:, F.std(0) > 0]
    if F.shape[1] <= 1:
        return F
    Z = (F - F.mean(0)) / F.std(0)
    C = np.abs(Z.T @ Z) / len(Z)
    keep = []
    for j in range(F.shape[1]):
        if all(C[j, i] < tol for i in keep):
            keep.append(j)
    return F[:, keep]


def refex(A, depth):
    k = np.asarray(A.sum(1)).ravel()
    inv = np.divide(1.0, k, out=np.zeros(len(k)), where=k > 0)
    F = podar(locales(A))
    todas = F
    for _ in range(depth):
        S = A @ F
        M = S * inv[:, None]
        F = podar(np.column_stack([S, M]))
        if F.shape[1] == 0:
            break
        todas = podar(np.column_stack([todas, F]))
    return todas


def evaluar(X, y, trm, tem, binary, seed=SEED):
    X = np.nan_to_num(np.asarray(X, dtype=float))
    sc = []
    for i in range(trm.shape[0]):
        m = RandomForestClassifier(300, min_samples_leaf=5, random_state=seed,
                                   n_jobs=N_JOBS).fit(X[trm[i]], y[trm[i]])
        if binary:
            sc.append(roc_auc_score(y[tem[i]], m.predict_proba(X[tem[i]])[:, 1]))
        else:
            sc.append(accuracy_score(y[tem[i]], m.predict(X[tem[i]])))
    return float(np.mean(sc)), float(np.std(sc) / np.sqrt(len(sc)))


def main(redes, root="."):
    import platform
    import scipy
    import sklearn
    print(f"python {platform.python_version()}   numpy {np.__version__}   "
          f"scipy {scipy.__version__}   sklearn {sklearn.__version__}", flush=True)
    print("(las tablas del articulo se produjeron con scikit-learn 1.6.1)", flush=True)
    filas = []
    for name in redes:
        A, y, trm, tem = cargar(name, root)
        binary = name in BINARY
        met = "AUROC" if binary else "acc"
        k = np.asarray(A.sum(1)).ravel().reshape(-1, 1)
        t0 = time.time()
        print(f"\n### {name}  n={A.shape[0]}  metrica {met}", flush=True)

        mu_k, se_k = evaluar(k, y, trm, tem, binary)
        print(f"   grado          {mu_k:.4f} (se {se_k:.4f})", flush=True)

        mejor = (-1, None, None)
        for d in DEPTHS:
            R = refex(A, d)
            mu, se = evaluar(R, y, trm, tem, binary)
            print(f"   ReFeX d={d}     {mu:.4f} (se {se:.4f})   {R.shape[1]} feat",
                  flush=True)
            if mu > mejor[0]:
                mejor = (mu, se, f"d={d}, {R.shape[1]} feat")

        X = omega_n(A)
        mu_x, se_x = evaluar(X, y, trm, tem, binary)
        print(f"   Omega-N        {mu_x:.4f} (se {se_x:.4f})   10 feat", flush=True)
        print(f"   >>> mejor ReFeX {mejor[0]:.4f} [{mejor[2]}]   "
              f"Omega-N {mu_x:.4f}   dif {mu_x-mejor[0]:+.4f}   "
              f"({time.time()-t0:.0f}s)", flush=True)
        filas.append((name, met, mu_k, mejor[0], mu_x, mu_x - mejor[0]))

    print("\n" + "=" * 78)
    print(f"{'red':16} {'metrica':>8} {'grado':>8} {'ReFeX':>8} {'Omega-N':>9} {'dif':>9}")
    for f in filas:
        print(f"{f[0]:16} {f[1]:>8} {f[2]:>8.4f} {f[3]:>8.4f} {f[4]:>9.4f} {f[5]:>+9.4f}")


if __name__ == "__main__":
    args = list(sys.argv[1:])
    root = "."
    if "--root" in args:
        i = args.index("--root")
        root = args[i + 1]
        args = args[:i] + args[i + 2:]
    main(args or PLATONOV, root)
