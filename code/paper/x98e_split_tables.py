"""arXiv version: split the cross-corpus law table written by x98b_size_contagion.py (tables/xb_corpora_laws.tex, 17 columns) into
two readable tables with the same cells: the depth laws (pooled, among replies, within threads; gold, Haiku, Sonnet; and the count
of settings weaker within threads) and the thread size law with the log reply risk ratio (gold, Haiku, Sonnet). No number changes.
Also splits the decomposition table written by x98a_escalation.py (tables/xa_decomp_terms.tex, 17 columns) into the depth and
thread size laws and the article size law with the log reply risk ratio.
usage: python3 x98e_split_tables.py"""
import os as _os
_ROOT = _os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))  # repository root (release: replaces the workspace path)
D = (_ROOT + "/paper_arxiv/tables")
rows = [l.rstrip() for l in open(f"{D}/xb_corpora_laws.tex") if l.strip()]
dep, srr = [], []
for r in rows:
    assert r.endswith("\\\\"), r
    c = [x.strip() for x in r[:-2].split("&")]
    assert len(c) == 17, (len(c), r)
    dep.append(" & ".join([c[0]] + c[1:10] + [c[16]]) + " \\\\")
    srr.append(" & ".join([c[0]] + c[10:16]) + " \\\\")
open(f"{D}/xb_corpora_depth.tex", "w").write("\n".join(dep) + "\n")
open(f"{D}/xb_corpora_sizerr.tex", "w").write("\n".join(srr) + "\n")
print("split", len(rows), "rows")

rows = [l.rstrip() for l in open(f"{D}/xa_decomp_terms.tex") if l.strip()]
a, b = [], []
for r in rows:
    if "&" not in r:
        a.append(r); b.append(r); continue
    c = [x.strip() for x in r[:-2].split("&")]
    assert len(c) == 17, (len(c), r)
    a.append(" & ".join([c[0]] + c[1:9]) + " \\\\")
    if any(x != "--" for x in c[9:17]):                                                 # arXiv revision: drop rows with no value (the '+ parent' rows)
        b.append(" & ".join([c[0]] + c[9:17]) + " \\\\")
open(f"{D}/xa_decomp_a.tex", "w").write("\n".join(a) + "\n")
open(f"{D}/xa_decomp_b.tex", "w").write("\n".join(b) + "\n")
print("split decomposition", len(rows), "rows")

# the earlier-draft cross-corpus table (paper_v5/tables/platforms_v6.tex, x18_tables_v5.py) with the arXiv text's corpus names; no number changes
t = open((_ROOT + "/results/reference/tables/platforms_v6.tex")).read()
t2 = t.replace("Qian et al. Reddit (hate)", "Qian et al.'s Reddit (hate)").replace("Qian et al. Gab (hate)", "Qian et al.'s Gab (hate)")
assert t2.count("Qian et al.'s") == 2
# review round 2: Civil Comments' placebo depth slope with the interval of Table 4 (results/x74_placebo, the same labels; one bootstrap run everywhere)
import json
sg_ = lambda x: (f"{x:+.2f}" if abs(x) >= 0.005 else "0.00").replace("-", "$-$")
v_ = [100 * t for t in json.load(open((_ROOT + "/results/x74_placebo/x74_placebo.json")))["cc_threads"]["labelers"]["haiku"]["placebo_depth_slope"]]
new_ = f"{sg_(v_[0])} [{sg_(v_[1])}, {sg_(v_[2])}]"
l0 = t2.split("\n")[0]; assert l0.startswith("Civil Comments (toxicity)")
cells_ = [c.strip() for c in l0.rstrip().rstrip("\\").split("&")]
assert cells_[6].split(" [")[0] == new_.split(" [")[0], (cells_[6], new_)
cells_[6] = new_
t2 = t2.replace(l0, " & ".join(cells_) + " \\\\", 1)
open(f"{D}/xe_platforms.tex", "w").write(t2); print("platforms names; Civil Comments placebo interval", new_)

