"""Cached-parser streams for TRIAGE / Experiment E (2026-11-10).

  cd experiments && python log_parsing_stream.py prep [-s HDFS ...]
  -> data/stream/<sys>.json: per line group id (fine Drain, LH2 settings), coarse family id (st halved),
     ground-truth template id; per group first line index, first-line content; template strings.

Stream = Loghub-2.0 <Sys>_full.log_structured.csv lines 1..50,000 in file (= log) order. Drain clusters never
reassign lines, so the cluster a line joins when it is processed is its online group; the cluster's first line
(smallest line index) is the one the cached parser sends to the LLM.
"""
import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

SYSTEMS = ["Apache", "BGL", "HDFS", "HPC", "Hadoop", "HealthApp", "Linux", "Mac", "OpenSSH", "OpenStack",
           "Spark", "Thunderbird", "Zookeeper"]
N_LINES = 50_000
RAW = Path.home() / ".cache" / "loghub2"
OUT = EXP / "data" / "stream"
FIXED = [0]


def read_stretch(system, n=N_LINES):
    import csv, io, zipfile
    csv.field_size_limit(sys.maxsize)
    rows = []
    with zipfile.ZipFile(RAW / f"{system}.zip") as zf, zf.open(f"{system}/{system}_full.log_structured.csv") as f:
        for r in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", newline="")):
            rows.append((int(r["LineId"]), r["Content"], r["EventTemplate"]))
            if len(rows) >= n:
                break
    ids = [r[0] for r in rows]
    assert ids == list(range(ids[0], ids[0] + len(ids))), "not contiguous"
    return rows


def drain_clusters(messages, system, st_scale=1.0):
    """-> per-message cluster index (in cluster creation order)."""
    import pandas as pd
    from logparser.Drain import LogParser
    from log_parsing_drain import LH2
    s = LH2[system]
    with tempfile.TemporaryDirectory() as wd:
        class P(LogParser):
            def load_data(self):
                self.df_log = pd.DataFrame({"LineId": range(1, len(messages) + 1), "Content": messages})

            def outputResult(self, logClustL):   # keep the cluster objects; skip writing files
                self.clusters = logClustL

            def addSeqToPrefixTree(self, rn, logClust):
                """Fix of the logparser3 / Loghub-2.0 Drain bug: a message with fewer tokens than the tree depth
                creates its path but its cluster is never attached, so every identical later line opens a new
                cluster. Attach it at the leaf that treeSearch reaches."""
                super().addSeqToPrefixTree(rn, logClust)
                seq = logClust.logTemplate
                node, d = rn.childD[len(seq)], 1
                for tok in seq:
                    if d >= self.depth or d > len(seq):
                        break
                    node = node.childD[tok] if tok in node.childD else node.childD["<*>"]
                    d += 1
                if isinstance(node.childD, dict):
                    assert not node.childD
                    node.childD = [logClust]
                    FIXED[0] += 1
                elif logClust not in node.childD:
                    node.childD.append(logClust)
                    FIXED[0] += 1
        p = P(log_format="<Content>", indir=wd, outdir=wd, depth=s["depth"], st=s["st"] * st_scale,
              rex=s["regex"], keep_para=False)
        p.parse(f"{system}.log")
    lab = [None] * len(messages)
    cl = sorted(p.clusters, key=lambda c: min(c.logIDL))       # creation order = first line order
    for g, c in enumerate(cl):
        for lid in c.logIDL:
            lab[lid - 1] = g
    assert all(x is not None for x in lab)
    return lab


def prep(system):
    rows = read_stretch(system)
    msgs = [r[1] for r in rows]
    tpls = sorted(set(r[2] for r in rows))
    tid = {t: i for i, t in enumerate(tpls)}
    FIXED[0] = 0
    fine = drain_clusters(msgs, system)
    fixed_fine = FIXED[0]
    FIXED[0] = 0
    coarse = drain_clusters(msgs, system, 0.5)
    fixed_coarse = FIXED[0]
    first = {}
    for i, g in enumerate(fine):
        first.setdefault(g, i)
    ng = len(first)
    assert sorted(first) == list(range(ng)) and all(first[g] < first[g + 1] for g in range(ng - 1))
    res = {"system": system, "line_start": rows[0][0], "n_lines": len(rows), "templates": tpls,
           "gt": [tid[r[2]] for r in rows], "group": fine, "family_line": coarse,
           "first": [first[g] for g in range(ng)], "first_content": [msgs[first[g]] for g in range(ng)],
           "family": [coarse[first[g]] for g in range(ng)],
           "drain_fix_clusters": [fixed_fine, fixed_coarse]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{system}.json").write_text(json.dumps(res))
    print(f"{system}: {len(rows)} lines, {len(tpls)} templates, {ng} groups, {len(set(coarse))} families", flush=True)


def load(system):
    return json.loads((OUT / f"{system}.json").read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep"])
    ap.add_argument("-s", "--systems", nargs="+", default=SYSTEMS)
    a = ap.parse_args()
    logging.disable(logging.WARNING)
    for s in a.systems:
        if (OUT / f"{s}.json").exists():
            print(f"{s}: exists")
            continue
        prep(s)
