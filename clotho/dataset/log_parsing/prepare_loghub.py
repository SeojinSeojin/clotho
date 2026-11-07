"""Build a log-parsing test suite from Loghub-2.0 (Zenodo record 8275861, CC BY 4.0).

  python clotho/dataset/log_parsing/prepare_loghub.py --system hdfs --n 500
  -> clotho/dataset/log_parsing/loghub_hdfs_500.jsonl

Input per system: <System>.zip -> <System>/<System>_full.log_structured.csv
with columns LineId, Content, EventId, EventTemplate. `Content` is the message
with the header (timestamp, level, component...) already removed, which is
what the ground-truth template describes, so it is the model input.

Sampling:
  1. Deduplicate identical Content strings (first occurrence wins).
  2. Stratify by template (EventId) with a common cap c: each template
     contributes min(#unique messages, c), with c the smallest cap reaching n
     (ties at the cap trimmed at random). So every template appears at least
     once whenever n >= #templates, and frequent templates cannot dominate.
     Within a template, messages are a uniform random sample of its unique
     messages (reservoir sampling while streaming, so the 1.7 GB CSV is never
     held in memory).
  3. Shuffle the final sample, so any prefix (e.g. the first shards) is mixed.

Output jsonl fields: id, system, content, template, template_id.
Raw zips are cached in --raw_dir (default ~/.cache/loghub2) and never committed.
"""
import argparse
import csv
import hashlib
import io
import json
import random
import statistics
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

ZENODO = "https://zenodo.org/api/records/8275861/files/{name}.zip/content"
SYSTEMS = ["Apache", "BGL", "HDFS", "HPC", "Hadoop", "HealthApp", "Linux", "Mac",
           "OpenSSH", "OpenStack", "Proxifier", "Spark", "Thunderbird", "Zookeeper"]
HERE = Path(__file__).resolve().parent


def system_name(s):
    for name in SYSTEMS:
        if name.lower() == s.lower():
            return name
    raise SystemExit(f"unknown system {s!r}; choose from {SYSTEMS}")


def fetch(name, raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{name}.zip"
    if path.exists() and zipfile.is_zipfile(path):
        return path
    url = ZENODO.format(name=name)
    print(f"downloading {url} -> {path}", flush=True)
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(path)
    return path


def stream_rows(zip_path, name):
    csv.field_size_limit(sys.maxsize)
    with zipfile.ZipFile(zip_path) as zf:
        member = f"{name}/{name}_full.log_structured.csv"
        if member not in zf.namelist():
            raise SystemExit(f"{member} not in {zip_path}: {zf.namelist()}")
        with zf.open(member) as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", newline=""))
            missing = {"LineId", "Content", "EventId", "EventTemplate"} - set(reader.fieldnames)
            if missing:
                raise SystemExit(f"columns {missing} missing; got {reader.fieldnames}")
            yield from reader


def choose_cap(counts, n):
    """Smallest c with sum(min(k, c)) >= n (counts: unique messages per template)."""
    lo, hi = 1, max(counts)
    while lo < hi:
        mid = (lo + hi) // 2
        if sum(min(k, mid) for k in counts) >= n:
            hi = mid
        else:
            lo = mid + 1
    return lo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True)
    ap.add_argument("--n", type=int, default=500, help="500 for pilot, 10000 for full")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--raw_dir", type=Path, default=Path.home() / ".cache" / "loghub2")
    ap.add_argument("--out_dir", type=Path, default=HERE)
    a = ap.parse_args()
    name = system_name(a.system)
    rng = random.Random(a.seed)
    zip_path = fetch(name, a.raw_dir)

    seen = set()                  # 8-byte hashes of unique contents
    template_of = {}              # EventId -> EventTemplate
    n_unique = Counter()          # EventId -> #unique messages
    reservoir = {}                # EventId -> list of (LineId, Content), size <= n
    total = conflicts = 0
    first_template = {}           # hash -> EventId (to count label conflicts)
    for row in stream_rows(zip_path, name):
        total += 1
        content, eid = row["Content"], row["EventId"]
        h = hashlib.blake2b(content.encode(), digest_size=8).digest()
        if h in seen:
            if first_template[h] != eid:
                conflicts += 1
            continue
        seen.add(h)
        first_template[h] = eid
        if template_of.setdefault(eid, row["EventTemplate"]) != row["EventTemplate"]:
            raise SystemExit(f"EventId {eid} has two templates")
        n_unique[eid] += 1
        res = reservoir.setdefault(eid, [])
        item = (row["LineId"], content)
        if len(res) < a.n:
            res.append(item)
        else:
            j = rng.randrange(n_unique[eid])
            if j < a.n:
                res[j] = item
        if total % 2_000_000 == 0:
            print(f"  ... {total:,} lines, {len(seen):,} unique", flush=True)
    del seen, first_template

    n_total_unique = sum(n_unique.values())
    eids = sorted(n_unique, key=lambda e: (len(e), e))
    if n_total_unique <= a.n:
        print(f"WARNING: {name} has only {n_total_unique} unique messages (< n={a.n}); taking all.")
        take = {e: n_unique[e] for e in eids}
    elif a.n < len(eids):
        print(f"WARNING: n={a.n} < {len(eids)} templates; sampling {a.n} templates, 1 message each.")
        take = {e: 0 for e in eids}
        for e in rng.sample(eids, a.n):
            take[e] = 1
    else:
        cap = choose_cap([n_unique[e] for e in eids], a.n)
        take = {e: min(n_unique[e], cap) for e in eids}
        at_cap = [e for e in eids if take[e] == cap]
        for e in rng.sample(at_cap, sum(take.values()) - a.n):
            take[e] -= 1

    records = []
    for e in eids:
        for line_id, content in rng.sample(reservoir[e], take[e]):
            records.append({"id": f"{name.lower()}-{line_id}", "system": name, "content": content,
                            "template": template_of[e], "template_id": e})
    rng.shuffle(records)

    out = a.out_dir / f"loghub_{name.lower()}_{a.n}.jsonl"
    with open(out, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    per = [v for v in Counter(r["template_id"] for r in records).values()]
    print(f"\n== {name} ==")
    print(f"total lines            {total:,}")
    print(f"unique messages        {n_total_unique:,}")
    print(f"templates (all)        {len(eids)}")
    print(f"templates in sample    {len(per)}  ({len(eids) - len(per)} missing)")
    print(f"samples per template   min {min(per)} / median {statistics.median(per)} / max {max(per)}")
    print(f"duplicate messages with a different template than their first occurrence: {conflicts:,}")
    print(f"wrote {len(records)} records -> {out}")


if __name__ == "__main__":
    main()
