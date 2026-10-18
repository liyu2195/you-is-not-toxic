"""Rebuild the paper's population files results/x14_pops/<population>.parquet from labels/ (and, for Qian et al.'s Reddit and Gab,
CGA-Wiki and BAD, the gold labels read from the original downloads by rebuild_gold.py).

Each population file has one row per comment, in the order of labels/<population>/structure.parquet, and the columns the
analysis and paper scripts read: id, unit (thread, conversation or dialogue, numbered in order of first appearance), cluster
(article on Civil Comments, otherwise the unit), depth, parent_row (row of the parent in the file, -1 if none), z (gold), k
(comments in the unit), on Civil Comments also article and tox (toxicity fraction), then every labeler's y:<labeler> (flag) and
s:<labeler> (score) in the stored order. The result is checked against the digest in labels/manifest.json, which was computed
from the files the paper was built from; a mismatch stops the script. With cc_threads it also writes
results/f10_population_thread_items.parquet (the same comments in the layout of code/data/f10_population.py), which
code/paper/x99b_draft_numbers.py reads.

usage: python3 -I code/data/assemble_populations.py [--qian DIR] [--cga DIR] [--bad DIR] [--only pop,pop]
  root: $YOU_IS_NOT_TOXIC_ROOT (default: the repository root); writes $YOU_IS_NOT_TOXIC_ROOT/results/x14_pops/.
  --qian: the folder holding reddit.csv and gab.csv; --cga: ConvoKit's conversations-gone-awry-corpus folder;
  --bad: the folder holding train.txt, valid.txt and test.txt of bot_adversarial_dialogue_datasets_with_persona.
  A population whose original download is not given is skipped (Civil Comments and CAD need none)."""
import argparse, hashlib, json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rebuild_gold as RG                                                                    # noqa: E402

ROOT = os.environ.get("YOU_IS_NOT_TOXIC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
LAB = os.path.join(ROOT, "labels"); OUT = os.path.join(ROOT, "results", "x14_pops")


def table_digest(df):
    """sha256 over the row count and, per column, its name, dtype and little-endian values (NaN canonicalized)."""
    h = hashlib.sha256(); h.update(str(len(df)).encode())
    for c in df.columns:
        a = df[c].to_numpy()
        h.update(c.encode() + b"\0" + str(a.dtype).encode() + b"\0")
        if a.dtype.kind == "f":
            a = np.where(np.isnan(a), np.nan, a)
        h.update(np.ascontiguousarray(a.astype(a.dtype.newbyteorder("<"))).tobytes())
    return h.hexdigest()


def rows_of(keys, targets):
    """row index of each target key among keys (-1 where missing or null)"""
    pos = pd.Series(np.arange(len(keys)), index=pd.Index(keys))
    assert pos.index.is_unique
    r = pd.Series(targets).map(pos)
    return np.where(r.notna(), r.fillna(-1), -1).astype(np.int64)


def build(pop, man, a):
    d = os.path.join(LAB, pop)
    S = pd.read_parquet(os.path.join(d, "structure.parquet"))
    n = len(S); assert n == man["n_rows"], (pop, n)
    if pop in ("cc_threads", "cc_s5"):
        z = S.gold.values; unit = pd.factorize(S.thread_id.values)[0]; cluster = pd.factorize(S.article_id.values)[0]
        parent_row = rows_of(S.id.values, S.parent_id.astype("Int64").astype(object).where(S.parent_id.notna(), None).values)
        extra = {"article": cluster.astype(np.int64), "tox": S.toxicity.values.astype(np.float64)}
    elif pop == "cad":
        z = S.gold.values; unit = pd.factorize(S.thread_id.values)[0]; cluster = unit
        parent_row = rows_of(S.info_id.values, S.parent_info_id.astype(object).where(S.parent_info_id.notna(), None).values); extra = {}
    elif pop in ("qian_reddit", "qian_gab"):
        if not a.qian:
            return None
        G = RG.qian(a.qian, pop.split("_")[1]).set_index(["conversation", "post"])
        key = pd.MultiIndex.from_arrays([S.conversation.values, S.post.values])
        assert key.isin(G.index).all(), f"{pop}: posts missing from the download"
        G = G.loc[key]
        assert np.array_equal(G.id.values, S.id.values) and np.array_equal(G.depth.values, S.depth.values), f"{pop}: structure differs from the download"
        assert (G.parent_post.fillna(-1).values == S.parent_post.fillna(-1).values).all(), f"{pop}: parents differ from the download"
        z = G.gold.values.astype(np.int8); unit = pd.factorize(S.conversation.values)[0]; cluster = unit
        par_id = np.where(S.parent_post.notna(), RG.QIAN_OFFSET[pop.split("_")[1]] + 100 * S.conversation.values + S.parent_post.fillna(0).values.astype(np.int64), -1)
        parent_row = rows_of(S.id.values, np.where(par_id >= 0, par_id, None)); extra = {}
    elif pop == "cga":
        if not a.cga:
            return None
        G = RG.cga(a.cga).set_index("utterance_id")
        assert S.utterance_id.isin(G.index).all(), "cga: utterances missing from the download"
        G = G.loc[S.utterance_id.values]
        assert np.array_equal(G.id.values, S.id.values) and np.array_equal(G.position.values, S.position.values), "cga: structure differs from the download"
        assert (G.verified.values == 1).all() and (G.pair_verified.values == 1).all()
        assert (G.reply_to.fillna("").values == S.reply_to.fillna("").values).all(), "cga: reply-to links differ from the download"
        z = G.gold.values.astype(np.int8); unit = pd.factorize(S.conversation_id.values)[0]; cluster = unit
        parent_row = rows_of(S.utterance_id.values, S.reply_to.astype(object).where(S.reply_to.notna(), None).values)
        parent_row[parent_row == np.arange(n)] = -1                                           # comments that name themselves as parent get none
        extra = {}
    elif pop == "bad":
        if not a.bad:
            return None
        G = RG.bad(a.bad).set_index("id").loc[S.id.values]
        for c in ("split", "dialogue", "turn", "speaker"):
            assert (G[c].astype(str).values == S[c].astype(str).values).all(), f"bad: {c} differs from the download"
        z = G.gold.values.astype(np.int8)
        dlg = S.split.astype(str).values + "_" + S.dialogue.astype(str).values
        unit = pd.factorize(dlg)[0]; cluster = unit
        at = {(g, t): i for i, (g, t) in enumerate(zip(dlg, S.turn.values.tolist()))}       # the parent is the dialogue's previous turn
        assert len(at) == n
        parent_row = np.array([at.get((g, t - 1), -1) for g, t in zip(dlg, S.turn.values.tolist())], dtype=np.int64); extra = {}
    else:
        raise SystemExit(f"unknown population {pop}")
    unit = unit.astype(np.int64); cluster = np.asarray(cluster).astype(np.int64)
    k = pd.Series(unit).groupby(unit).transform("size").values.astype(np.int64)
    cols = {"id": S.id.values.astype(np.int64), "unit": unit, "cluster": cluster, "depth": S.depth.values.astype(np.int64),
            "parent_row": parent_row, "z": np.asarray(z).astype(np.int8), "k": k, **extra}
    L = None
    for f in [v for kk, v in man["files"].items() if kk != "structure"]:
        X = pd.read_parquet(os.path.join(d, f))
        assert np.array_equal(X.id.values, S.id.values), (pop, f)
        L = X.drop(columns="id") if L is None else pd.concat([L, X.drop(columns="id")], axis=1)
    order = [c for c, _ in man["columns"]]
    P = pd.DataFrame({c: cols[c] for c in order if c in cols})
    for c in order:
        if c not in P:
            P[c] = L[c].values.astype(np.float64)
    P = P[order]
    for c, t in man["columns"]:
        assert str(P[c].dtype) == t, (pop, c, P[c].dtype, t)
    return P


def thread_items(P, S, spec):
    """results/f10_population_thread_items.parquet (code/data/f10_population.py): the Civil Comments reply-thread items, index id"""
    F = pd.DataFrame({"root": S.thread_id.values.astype(np.int64), "article_id": S.article_id.values.astype(np.int64), "depth": P.depth.values,
                      "tk": P.k.values, "Z": P.z.values.astype(np.int8), "haiku": P["y:haiku"].values.astype(np.int8)},
                     index=pd.Index(P.id.values, name="id"))
    for c in ("haiku_ctx", "qwen7b_m", "qwen7b_m_ctx", "qwen7b"):
        F[c] = P[f"y:{c}"].values.astype(np.float64)
    assert [[c, str(F[c].dtype)] for c in F.columns] == spec["columns"]
    return F


def main():
    ap = argparse.ArgumentParser(description="rebuild results/x14_pops from labels/")
    ap.add_argument("--qian"); ap.add_argument("--cga"); ap.add_argument("--bad"); ap.add_argument("--only")
    a = ap.parse_args()
    MAN = json.load(open(os.path.join(LAB, "manifest.json"))); man = MAN["populations"]
    os.makedirs(OUT, exist_ok=True); bad_digest = []
    for pop in (a.only.split(",") if a.only else man):
        P = build(pop, man[pop], a)
        if P is None:
            print(f"{pop}: skipped (give the original download to rebuild its gold labels)"); continue
        dg = table_digest(P); ok = dg == man[pop]["digest_sha256"]
        if not ok:
            bad_digest.append(pop); print(f"{pop}: DIGEST MISMATCH, not written"); continue
        P.to_parquet(os.path.join(OUT, f"{pop}.parquet"), index=False)
        print(f"{pop}: {len(P)} rows, {P.unit.nunique()} units, {int((P.parent_row >= 0).sum())} reply edges -> results/x14_pops/{pop}.parquet (digest ok)")
        for path, spec in MAN.get("derived", {}).items():                                # files derived from this population
            if spec["from_population"] != pop:
                continue
            F = thread_items(P, pd.read_parquet(os.path.join(LAB, pop, "structure.parquet")), spec)
            if table_digest(F.reset_index()) != spec["digest_sha256"]:
                bad_digest.append(path); print(f"{path}: DIGEST MISMATCH, not written"); continue
            F.to_parquet(os.path.join(ROOT, path)); print(f"{pop}: {len(F)} rows -> {path} (digest ok)")
    if bad_digest:
        raise SystemExit(f"digest mismatch: {bad_digest}")


if __name__ == "__main__":
    main()
