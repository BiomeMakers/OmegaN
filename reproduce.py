"""Regenerate the benchmark tables of the paper.

Datasets are fetched from their original sources and are not redistributed with
this repository. Run `python reproduce.py --list` to see what is needed.
"""
import argparse
import numpy as np
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score

from omega_n import omega_n, screen

PLATONOV = ["tolokers", "amazon_ratings", "questions", "roman_empire", "minesweeper"]
BINARY = {"tolokers", "questions", "minesweeper"}


def load_platonov(name, root="."):
    d = np.load(f"{root}/{name}.npz")
    y = d["node_labels"]
    n = len(y)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(map(tuple, d["edges"]))
    G.remove_edges_from(nx.selfloop_edges(G))
    A = nx.to_scipy_sparse_array(G, nodelist=range(n), format="csr").astype(float)
    return A, y, d["node_features"], d["train_masks"], d["test_masks"]


def evaluate(X, y, trm, tem, binary, seed=42):
    X = np.nan_to_num(np.asarray(X, dtype=float))
    scores = []
    for i in range(trm.shape[0]):
        m = RandomForestClassifier(300, min_samples_leaf=5, random_state=seed,
                                   n_jobs=-1).fit(X[trm[i]], y[trm[i]])
        if binary:
            scores.append(roc_auc_score(y[tem[i]], m.predict_proba(X[tem[i]])[:, 1]))
        else:
            scores.append(accuracy_score(y[tem[i]], m.predict(X[tem[i]])))
    return float(np.mean(scores)), float(np.std(scores) / np.sqrt(len(scores)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--datasets", nargs="*", default=PLATONOV)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        print("Platonov et al. (ICLR 2023) heterophilous-graphs release:")
        for n in PLATONOV:
            print(f"  {n}.npz")
        return
    for name in args.datasets:
        A, y, feats, trm, tem = load_platonov(name, args.root)
        s = screen(A)
        print(f"\n### {name}  n={s['n_nodes']}  mean degree={s['mean_degree']:.1f}  "
              f"clustering={s['mean_clustering']:.3f}  [{s['verdict']}]")
        X = omega_n(A)
        binary = name in BINARY
        k = np.asarray(A.sum(1)).ravel().reshape(-1, 1)
        arms = {"degree": k, "Omega-N": X, "attributes": feats,
                "attributes + Omega-N": np.column_stack([feats, X])}
        for tag, M in arms.items():
            mu, se = evaluate(M, y, trm, tem, binary)
            print(f"   {tag:<24}{M.shape[1]:>5} feat   {mu:.4f}  (se {se:.4f})")


if __name__ == "__main__":
    main()
