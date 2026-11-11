"""R2 strict review metrics, including a certified metric-specific FTA oracle."""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
from collections import Counter
import time
import numpy as np
from log_parsing_drain_templates import fta


class ReviewMetrics:
    def __init__(self, d, cache, keep=None):
        self.d, self.cache = d, list(cache)
        self.keep = set(range(d['G'])) if keep is None else set(keep)
        self.ids = sorted(self.keep)
        self.corrected = [d['templates'][t] for t in d['major']]
        self.base = np.array([sum(n for t, n in d['cnt'][g].items() if cache[g] == d['templates'][t])
                              for g in range(d['G'])], float)
        self.gain = np.array([max(d['cnt'][g].values()) for g in range(d['G'])]) - self.base
        self.wrong = np.array([cache[g] != d['templates'][d['gtl'][d['first'][g]]]
                               for g in range(d['G'])])
        self.group_gain = self.wrong & np.array([self.corrected[g] == d['templates'][d['gtl'][d['first'][g]]]
                                                for g in range(d['G'])])
        self.bad_lines = sum(sum(d['cnt'][g].values()) - self.base[g] for g in self.ids)
        self.bad_groups = sum(self.wrong[g] for g in self.ids)
        self.lines = np.flatnonzero(np.isin(d['grp'], self.ids))
        self.gt = [d['templates'][d['gtl'][i]] for i in self.lines]

    def evaluate(self, reviewed):
        rv = set(reviewed) & self.keep
        p = [str(self.corrected[g] if g in rv else self.cache[g]) for g in self.d['grp'][self.lines]]
        return {'lines': float(sum(self.gain[g] for g in rv) / self.bad_lines) if self.bad_lines else None,
                'groups': float(sum(self.group_gain[g] for g in rv) / self.bad_groups) if self.bad_groups else None,
                'FTA': fta(p, self.gt)['FTA'] if p else None}

    def oracle(self, budget, eligible, metric):
        ids = sorted(set(eligible) & self.keep)
        if metric != 'FTA':
            gains = self.gain if metric == 'lines' else self.group_gain
            ids.sort(key=lambda g: (-float(gains[g]), -sum(self.d['cnt'][g].values()), g))
            seq = ids[:budget]
            return {'seq': seq, 'value': self.evaluate(seq)[metric], 'certified': True}
        return self.fta_oracle(budget, ids)

    def fta_oracle(self, budget, eligible, time_limit=30):
        """Maximise 2*correct/(distinct predicted + GT templates) with <= budget reviews.

        A predicted string is active iff at least one group outputs it. A GT template
        counts correct iff exactly its lines output that string. Binary constraints
        encode both conditions. Dinkelbach solves the resulting fractional MILP.
        """
        from scipy.optimize import milp, Bounds, LinearConstraint
        from scipy.sparse import coo_matrix
        ids = self.ids
        # A review that leaves the output string unchanged cannot improve FTA.
        el = [g for g in eligible if str(self.cache[g]) != self.corrected[g]]
        if not ids:
            return {'seq': [], 'value': None, 'certified': True}
        xidx = {g: i for i, g in enumerate(el)}
        predgroups, truthgroups, gtsets = {}, {}, {}
        for g in ids:
            gtsets[g] = {self.d['templates'][i] for i in self.d['cnt'][g]}
            for t in gtsets[g]:
                truthgroups.setdefault(t, set()).add(g)
            for p in {str(self.cache[g])} | ({self.corrected[g]} if g in xidx else set()):
                predgroups.setdefault(p, set()).add(g)
        strings = sorted({str(self.cache[g]) for g in ids} | {self.corrected[g] for g in el})
        truths = sorted(set(self.gt))
        yi = {s: len(el) + i for i, s in enumerate(strings)}
        zi = {s: len(el) + len(strings) + i for i, s in enumerate(truths)}
        nv = len(el) + len(strings) + len(truths)
        rows, cols, vals, lbs, ubs = [], [], [], [], []

        def con(coeff, lo=-np.inf, hi=np.inf):
            row = len(lbs)
            for k, v in coeff.items():
                if v:
                    rows.append(row); cols.append(k); vals.append(v)
            lbs.append(lo); ubs.append(hi)

        def affine(g, s):
            a = int(str(self.cache[g]) == s)
            delta = int(self.corrected[g] == s) - a if g in xidx else 0
            return a, delta

        con({i: 1 for i in xidx.values()}, hi=min(budget, len(el)))
        for s in strings:
            total_a, deltas = 0, {}
            for g in predgroups[s]:
                a, b = affine(g, s)
                if not a and not b:
                    continue
                c = {yi[s]: 1}
                if b:
                    c[xidx[g]] = -b
                    deltas[xidx[g]] = -b
                con(c, lo=a)
                total_a += a
            con({yi[s]: 1, **deltas}, hi=total_a)
        for t in truths:
            for g in truthgroups[t] | predgroups.get(t, set()):
                gtset = gtsets[g]
                if t in gtset and gtset != {t}:
                    con({zi[t]: 1}, hi=0)
                    break
                a, b = affine(g, t)
                # Pure t groups must output t; every other group must not.
                wanted = gtset == {t}
                c = {zi[t]: 1}
                if b:
                    c[xidx[g]] = -b if wanted else b
                con(c, hi=a if wanted else 1-a)
        A = coo_matrix((vals, (rows, cols)), shape=(len(lbs), nv)).tocsc()
        constraints = LinearConstraint(A, np.asarray(lbs), np.asarray(ubs))
        integrality = np.ones(nv)
        bounds = Bounds(np.zeros(nv), np.ones(nv))
        ratio = self.evaluate([])['FTA']
        best = {'seq': [], 'value': ratio, 'certified': False}
        start = time.monotonic()
        for iteration in range(30):
            remaining = time_limit - (time.monotonic() - start)
            if remaining <= 0:
                break
            obj = np.zeros(nv)
            obj[list(yi.values())] = ratio
            obj[list(zi.values())] = -2
            res = milp(obj, integrality=integrality, bounds=bounds, constraints=constraints,
                       options={'time_limit': remaining, 'mip_rel_gap': 0.0})
            if res.x is None:
                break
            seq = [g for g, i in xidx.items() if res.x[i] > .5]
            value = self.evaluate(seq)['FTA']
            y = round(sum(res.x[i] for i in yi.values()))
            z = round(sum(res.x[i] for i in zi.values()))
            assert abs(value - 2*z/(y+len(truths))) < 1e-8
            best = {'seq': seq, 'value': value, 'certified': False, 'solver_status': int(res.status),
                    'iterations': iteration+1, 'mip_gap': float(getattr(res, 'mip_gap', np.nan))}
            if res.status == 0 and abs(2*z-ratio*(y+len(truths))) < 1e-8:
                best['certified'] = True
                break
            ratio = value
            if res.status != 0:
                break
        best['wall_s'] = time.monotonic()-start
        return best
