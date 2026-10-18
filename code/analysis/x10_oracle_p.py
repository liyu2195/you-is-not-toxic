"""Round A6 diagnosis: the I4 oracle working model saved per cell, for 'oracle' designs that use it instead of learning a model.
p_i = cross-fitted (5-fold) gradient boosting of z on the labeler's score and flag, depth, thread size, article size and the
auxiliary instruments' scores, exactly as in x10_ceiling.py's I4. Saved as results/x10_oracle_p/<pop>__<lab>.npy (float32).
usage: python3 x10_oracle_p.py POP LABELER --extra a,b"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import sys, os
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

W = _ROOT
pop, lab = sys.argv[1], sys.argv[2]
EXTRA = sys.argv[sys.argv.index("--extra") + 1].split(",") if "--extra" in sys.argv else []
P = pd.read_parquet(f"{W}/results/x10_pops/{pop}.parquet")
z = P.z.values.astype(float); y = np.nan_to_num(P[f"y:{lab}"].values.astype(float)); s = P[f"s:{lab}"].values.astype(float)
s = np.where(np.isnan(s), np.nanmin(s), s); N = len(P)
depth = P.depth.values; k = P.k.values.astype(float); xd = np.minimum(depth, 10).astype(float); logk = np.where(k >= 2, np.log2(np.maximum(k, 1)), 0.0)
X = np.column_stack([s, y, xd, logk] + ([np.log2(np.maximum(P.groupby("article").article.transform("size").values, 1))] if "article" in P else [])
                    + [np.nan_to_num(P[f"s:{e}"].values.astype(float), nan=-1.0) for e in EXTRA])
out = np.zeros(N)
for tr, te in KFold(5, shuffle=True, random_state=61).split(X):
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=61).fit(X[tr], z[tr])
    out[te] = clf.predict_proba(X[te])[:, 1]
os.makedirs(f"{W}/results/x10_oracle_p", exist_ok=True)
np.save(f"{W}/results/x10_oracle_p/{pop}__{lab.replace('@', '_at_')}.npy", np.clip(out, 1e-6, 1 - 1e-6).astype(np.float32))
print(pop, lab, "saved", N)
