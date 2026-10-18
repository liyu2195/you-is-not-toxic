"""Addendum 72: corrected populations. results/x14_pops/<pop>.parquet = results/x11_pops/<pop>.parquet with self-edges removed
(parent_row equal to the row itself becomes -1). Asserts that only CGA-Wiki changes, and only in those 84 entries.
usage: python3 x14_fix_pops.py"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
import os, glob, shutil, hashlib
import numpy as np, pandas as pd

W = _ROOT; SRC = f"{W}/results/x11_pops"; DST = f"{W}/results/x14_pops"; os.makedirs(DST, exist_ok=True)
report = []
for f in sorted(glob.glob(f"{SRC}/*.parquet")):
    name = os.path.basename(f); P = pd.read_parquet(f); pr = P.parent_row.values.copy(); n = len(P)
    self_ = np.flatnonzero(pr == np.arange(n))
    if len(self_) == 0:
        shutil.copyfile(f, f"{DST}/{name}")
        assert hashlib.sha256(open(f, "rb").read()).hexdigest() == hashlib.sha256(open(f"{DST}/{name}", "rb").read()).hexdigest()
        report.append((name, 0)); continue
    assert name == "cga.parquet" and len(self_) == 84, (name, len(self_))
    Q = P.copy(); Q.loc[self_, "parent_row"] = -1
    diff = [c for c in P.columns if not P[c].equals(Q[c])]
    assert diff == ["parent_row"] and int((P.parent_row.values != Q.parent_row.values).sum()) == 84
    Q.to_parquet(f"{DST}/{name}"); report.append((name, len(self_)))
for r in report:
    print(f"{r[0]}: {r[1]} self-edges removed")
shutil.copyfile(f"{SRC}/fresh_report.csv", f"{DST}/fresh_report.csv") if os.path.exists(f"{SRC}/fresh_report.csv") else None
