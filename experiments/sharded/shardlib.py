"""Sharded, resumable execution core (stdlib + huggingface_hub only).

A *job* = (model, task, dataset, template, stage, shard_size). Its input jsonl
is split into contiguous shards of `shard_size` lines. A shard is DONE when
`shard_XXXXX/DONE.json` exists (locally, or in RESULTS_REPO) and its input
fingerprint matches the current input lines. Each finished shard is uploaded
in one atomic commit together with the job's progress.log, so an interrupted
session loses at most the shard in flight.

Shard directories store outputs under the same relative paths the original
scripts use below `experiments/data/` (e.g. results_llama/spell_check/...),
which lets `merge()` rebuild the original files generically:
  *.json under inference_results/   -> lists concatenated in shard order
  *.pt   under input_hidden_states/ -> numpy arrays concatenated (axis 0)
  output_hidden_states.tar          -> extracted, input_<i>.pt re-indexed
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import statistics
import tarfile
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import hub_sync as hs

REPO_ROOT = hs.REPO_ROOT
EXP_DIR = hs.EXP_DIR
DATASET_DIR = REPO_ROOT / "clotho" / "dataset"
WORK_ROOT = EXP_DIR / "sharded_work"
CANONICAL_DATA = EXP_DIR / "data"
SESSION_START_FILE = Path.home() / ".clotho_session_start"

STAGES = ("hidden", "gen_hf", "gen_vllm")


def now():
    return datetime.now().isoformat(timespec="seconds")


def hms(sec):
    sec = int(sec)
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m{sec % 60:02d}s"


def fingerprint(lines):
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode())
        h.update(b"\n")
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class Job:
    model: str
    task: str
    dataset: str
    template: str
    stage: str
    shard_size: int

    @property
    def prefix(self):
        return (f"shards/{self.model}/{self.task}/{self.dataset}/{self.template}/"
                f"{self.stage}/size{self.shard_size}")

    @property
    def local(self):
        return WORK_ROOT / self.prefix

    def shard_name(self, k):
        return f"shard_{k:05d}"

    def shard_dir(self, k):
        return self.local / self.shard_name(k)

    def shard_prefix(self, k):
        return f"{self.prefix}/{self.shard_name(k)}"

    @property
    def progress_path(self):
        return self.local / "progress.log"


def load_lines(task, dataset):
    path = DATASET_DIR / task / f"{dataset}.jsonl"
    if not path.exists():
        raise SystemExit(f"Input not found: {path}")
    # The original scripts json.loads() every line, so there are no blank lines.
    return path.read_text().splitlines()


def build_arg_parser(stage_default=None):
    p = argparse.ArgumentParser(description="Sharded, resumable runner for Clotho data extraction.")
    p.add_argument("--stage", choices=STAGES, default=stage_default, required=stage_default is None,
                   help="hidden = 1-1 input hidden states (HF); gen_hf = 2-1 generation + output "
                        "hidden states (HF); gen_vllm = generation for pass/fail labels (vLLM)")
    p.add_argument("-t", "--task", required=True)
    p.add_argument("-d", "--dataset", required=True)
    p.add_argument("-m", "--model", default="llama")
    p.add_argument("-p", "--template", default="messages_template")
    p.add_argument("--shard_size", type=int, default=500)
    p.add_argument("--limit_shards", type=int, default=None,
                   help="only consider the first N shards (for testing). Merges of a limited "
                        "job go to sharded_work/merged_partial/, never to experiments/data.")
    p.add_argument("--status", action="store_true", help="print shard status and exit")
    p.add_argument("--merge_only", action="store_true", help="skip running; just merge")
    p.add_argument("--no_merge", action="store_true")
    p.add_argument("--local_only", action="store_true",
                   help="do not touch RESULTS_REPO (results stay on this machine!)")
    p.add_argument("--session_hours", type=float, default=10.0)
    p.add_argument("--safety_margin_min", type=float, default=20.0,
                   help="do not start a shard unless it is expected to finish this many "
                        "minutes before the session limit")
    return p


def job_from_args(args):
    from clotho.exp_config import supported_datasets  # validates names like the scripts do
    assert args.task in supported_datasets, f"unknown task {args.task}"
    assert args.dataset in supported_datasets[args.task], f"unknown dataset {args.dataset}"
    return Job(args.model, args.task, args.dataset, args.template, args.stage, args.shard_size)


class ShardRunner:
    def __init__(self, job, args):
        self.job = job
        self.args = args
        all_lines = load_lines(job.task, job.dataset)
        self.n_inputs = len(all_lines)
        self.shards = [all_lines[i:i + job.shard_size] for i in range(0, len(all_lines), job.shard_size)]
        self.n_total_shards = len(self.shards)
        if args.limit_shards:
            self.shards = self.shards[:args.limit_shards]
        self.repo = None if args.local_only else hs.get_results_repo(required=False)
        if self.repo is None and not args.local_only:
            print("[shard] WARNING: RESULTS_REPO not set -> results are NOT persisted off this machine.")
        self.remote_files = set()
        if self.repo:
            hs.ensure_repo(self.repo)
            self.remote_files = {f for f in hs.list_repo_files(self.repo) if f.startswith(job.prefix)}
        self.durations = []

    # ------------------------------------------------------------------ status
    @property
    def is_partial(self):
        return len(self.shards) < self.n_total_shards

    def shard_meta(self, k):
        lines = self.shards[k]
        return {"shard": k, "start": k * self.job.shard_size, "count": len(lines),
                "fingerprint": fingerprint(lines)}

    def _check_done_json(self, path, k):
        meta = json.loads(Path(path).read_text())
        if meta["fingerprint"] != fingerprint(self.shards[k]):
            raise SystemExit(
                f"[shard] {self.job.shard_prefix(k)}: DONE.json fingerprint does not match the current "
                f"input lines. The dataset changed since this shard was produced; use a new "
                f"--shard_size or delete the stale shard before continuing.")
        return meta

    def status(self, k):
        """'local' | 'remote' | 'both' | None"""
        local = (self.job.shard_dir(k) / "DONE.json").exists()
        remote = f"{self.job.shard_prefix(k)}/DONE.json" in self.remote_files
        if local:
            self._check_done_json(self.job.shard_dir(k) / "DONE.json", k)
        if remote and not local:
            p = hs.download_file(self.repo, f"{self.job.shard_prefix(k)}/DONE.json", WORK_ROOT / ".done_cache")
            self._check_done_json(p, k)
        return {(True, True): "both", (True, False): "local", (False, True): "remote"}.get((local, remote))

    def print_status(self):
        st = [self.status(k) for k in range(len(self.shards))]
        done = sum(s is not None for s in st)
        print(f"[shard] {self.job.prefix}: {done}/{len(st)} shards done"
              + (f" (limited; full job has {self.n_total_shards})" if self.is_partial else ""))
        for k, s in enumerate(st):
            print(f"   {self.job.shard_name(k)}  {s or 'pending'}")
        return st

    # ---------------------------------------------------------------- progress
    def _pull_progress_log(self):
        rp = f"{self.job.prefix}/progress.log"
        if self.repo and rp in self.remote_files and not self.job.progress_path.exists():
            hs.download_file(self.repo, rp, WORK_ROOT)

    def log(self, line):
        self.job.progress_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.job.progress_path, "a") as f:
            f.write(line + "\n")
        print(f"[progress] {line}", flush=True)

    def session_elapsed(self):
        try:
            return time.time() - int(SESSION_START_FILE.read_text().strip())
        except (OSError, ValueError):
            return None

    def fits_in_session(self):
        el = self.session_elapsed()
        if el is None or not self.durations:
            return True
        budget = self.args.session_hours * 3600 - self.args.safety_margin_min * 60
        return el + statistics.mean(self.durations) <= budget

    # ------------------------------------------------------------------ upload
    def upload_shard(self, k):
        if not self.repo:
            return
        d = self.job.shard_dir(k)
        files = {f"{self.job.shard_prefix(k)}/{p.relative_to(d).as_posix()}": p
                 for p in sorted(d.rglob("*")) if p.is_file()}
        if self.job.progress_path.exists():
            files[f"{self.job.prefix}/progress.log"] = self.job.progress_path
        t0 = time.time()
        hs.commit_files(self.repo, files, f"{self.job.shard_prefix(k)} done")
        self.remote_files.update(files)
        mb = sum(p.stat().st_size for p in files.values()) / 1e6
        print(f"[shard] uploaded {self.job.shard_name(k)} ({mb:.1f} MB) in {time.time() - t0:.0f}s", flush=True)

    # --------------------------------------------------------------------- run
    def run(self, run_fn):
        """run_fn(k, lines, out_dir) must write the shard's outputs into out_dir."""
        job = self.job
        self._pull_progress_log()
        statuses = [self.status(k) for k in range(len(self.shards))]
        for k, s in enumerate(statuses):  # finished locally but upload failed last time
            if s == "local" and self.repo:
                print(f"[shard] {job.shard_name(k)} done locally but not in RESULTS_REPO -> uploading")
                self.upload_shard(k)
        pending = [k for k, s in enumerate(statuses) if s is None]
        n_done = len(self.shards) - len(pending)
        self.log(f"# {now()} run start on {socket.gethostname()}: {job.prefix} | "
                 f"{n_done}/{len(self.shards)} shards done, {len(pending)} pending"
                 + (f" (limited to first {len(self.shards)} of {self.n_total_shards})" if self.is_partial else ""))

        for k in pending:
            if not self.fits_in_session():
                self.log(f"# {now()} STOP: next shard (~{hms(statistics.mean(self.durations))}) would not "
                         f"finish within the {self.args.session_hours}h session minus "
                         f"{self.args.safety_margin_min:.0f}min margin.")
                break
            final = job.shard_dir(k)
            tmp = final.with_name(final.name + ".partial")
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            t0 = time.time()
            run_fn(k, self.shards[k], tmp)
            dt = time.time() - t0
            meta = self.shard_meta(k) | {"stage": job.stage, "seconds": round(dt, 1),
                                         "finished_at": now(), "host": socket.gethostname()}
            (tmp / "DONE.json").write_text(json.dumps(meta, indent=2))
            shutil.rmtree(final, ignore_errors=True)
            tmp.rename(final)

            self.durations.append(dt)
            n_done += 1
            mean = statistics.mean(self.durations)
            remaining = len(self.shards) - n_done
            el = self.session_elapsed()
            self.log(f"{now()} {job.shard_name(k)} done in {hms(dt)} ({len(self.shards[k])} inputs, "
                     f"{dt / len(self.shards[k]):.2f}s/input) | {n_done}/{len(self.shards)} shards | "
                     f"mean {hms(mean)}/shard | ETA {remaining} left: {hms(remaining * mean)}"
                     + (f" | session elapsed {hms(el)}" if el is not None else ""))
            self.upload_shard(k)
        return all(self.status(k) is not None for k in range(len(self.shards)))

    # ------------------------------------------------------------------- merge
    def merge(self):
        job = self.job
        ks = list(range(len(self.shards)))
        missing = [k for k in ks if not (job.shard_dir(k) / "DONE.json").exists()]
        if missing:
            if not self.repo:
                raise SystemExit(f"[merge] shards {missing} missing locally and no RESULTS_REPO")
            print(f"[merge] downloading {len(missing)} shard(s) from {self.repo}")
            hs.download_prefix(self.repo, [f"{job.shard_prefix(k)}/*" for k in missing], WORK_ROOT)
        for k in ks:
            self._check_done_json(job.shard_dir(k) / "DONE.json", k)

        out_root = (WORK_ROOT / "merged_partial" / job.prefix) if self.is_partial else CANONICAL_DATA
        groups = defaultdict(list)
        for k in ks:
            d = job.shard_dir(k)
            for f in sorted(d.rglob("*")):
                if f.is_file() and f.name != "DONE.json":
                    groups[f.relative_to(d)].append((k, f))

        written = []
        for rel, items in sorted(groups.items()):
            if [k for k, _ in items] != ks:
                raise SystemExit(f"[merge] {rel} is missing from some shards: have {[k for k, _ in items]}")
            if rel.name == "output_hidden_states.tar":
                n = _merge_tars(items, job.shard_size, out_root)
                print(f"[merge] output hidden states: {n} files -> {out_root}")
                continue
            dest = out_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if rel.suffix == ".json":
                merged = []
                for k, f in items:
                    part = json.loads(f.read_text())
                    assert len(part) == len(self.shards[k]), f"{f}: {len(part)} != {len(self.shards[k])}"
                    merged.extend(part)
                dest.write_text(json.dumps(merged, indent=4))
                prov = dest.with_name(dest.stem + ".provenance.json")
                prov.write_text(json.dumps({"stage": job.stage, "shard_size": job.shard_size,
                                            "shards": len(ks), "merged_at": now()}, indent=2))
                written += [dest, prov]
                print(f"[merge] {len(merged)} records -> {dest}")
            elif rel.suffix == ".pt":
                import numpy as np
                import torch
                parts = []
                for k, f in items:
                    a = torch.load(f, weights_only=False)
                    assert a.shape[0] == len(self.shards[k]), f"{f}: {a.shape[0]} != {len(self.shards[k])}"
                    parts.append(a)
                merged = np.concatenate(parts, axis=0)
                torch.save(merged, dest)  # same format as 1-1: numpy array via torch.save
                written.append(dest)
                print(f"[merge] array {merged.shape} -> {dest}")
            else:
                raise SystemExit(f"[merge] don't know how to merge {rel}")

        if self.is_partial:
            print(f"[merge] partial job (limit_shards) -> wrote to {out_root}; not uploaded.")
        elif self.repo and written:
            files = {f"data/{p.relative_to(CANONICAL_DATA).as_posix()}": p for p in written}
            hs.commit_files(self.repo, files, f"merge {job.prefix}")
            print(f"[merge] uploaded {len(files)} merged file(s) to {self.repo}/data/")
        return out_root, written


_INPUT_RE = re.compile(r"^(.*/)input_(\d+)\.pt$")


def _merge_tars(items, shard_size, out_root):
    n = 0
    for k, tar_path in items:
        offset = k * shard_size
        with tarfile.open(tar_path) as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                name = m.name
                if name.startswith("/") or ".." in Path(name).parts:
                    raise SystemExit(f"unsafe tar member {name}")
                mt = _INPUT_RE.match(name)
                if not mt:
                    raise SystemExit(f"unexpected tar member {name}")
                dest = out_root / f"{mt.group(1)}input_{offset + int(mt.group(2))}.pt"
                dest.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(m) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                n += 1
    return n


def main_with_runner(args, make_run_fn):
    """Common driver: status / run / merge. make_run_fn(job) -> run_fn (lazy, so
    --status/--merge_only never load a model)."""
    job = job_from_args(args)
    runner = ShardRunner(job, args)
    if args.status:
        runner.print_status()
        return 0
    complete = True
    if not args.merge_only:
        complete = runner.run(make_run_fn(job))
    if not complete:
        print("[shard] not all shards done yet; rerun the same command in the next session to resume.")
        return 0
    if not args.no_merge:
        runner.merge()
    return 0
