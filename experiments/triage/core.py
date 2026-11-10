"""TRIAGE review queue."""
import numpy as np


class Triage:
    def __init__(self, prior, M=20, active=True, freq=True, seed=0):
        self.prior, self.M, self.active, self.freq, self.seed = prior, M, active, freq, seed
        self.verdicts = {}          # gid -> bool (cached template correct)
        self.n = 0

    # ---- state
    def set_groups(self, H, sizes, feats=None):
        """All groups seen so far (gid = row index, stable as rows are appended); sizes = lines per group so far."""
        self.prior.set_target(H, feats)
        self.sizes = np.asarray(sizes, dtype=float)
        self.n = len(self.sizes)

    def update(self, verdicts):
        self.verdicts.update({int(g): bool(v) for g, v in verdicts.items()})

    def p_pass(self):
        idx = sorted(self.verdicts)
        return self.prior.predict(idx, [self.verdicts[i] for i in idx], seed=self.seed)

    def risk(self):
        return 1.0 - self.p_pass()

    def priority(self, risk=None):
        r = self.risk() if risk is None else risk
        return r * self.sizes if self.freq else r

    # ---- selection
    def queue(self, b=5):
        p = self.p_pass()
        pri = self.priority(1.0 - p)
        open_ = np.array([g for g in range(self.n) if g not in self.verdicts], dtype=int)
        if not len(open_):
            return []
        order = open_[np.lexsort((-self.sizes[open_], -pri[open_]))]   # priority, then size
        if not self.active:
            return [int(g) for g in order[:b]]
        cand = list(order[:max(self.M, b)])
        picked = []
        # uncertainty half: largest binary entropy of P(PASS)
        q = np.clip(p, 1e-9, 1 - 1e-9)
        ent = -(q * np.log(q) + (1 - q) * np.log(1 - q))
        for g in sorted(cand, key=lambda g: (-ent[g], cand.index(g)))[:b // 2]:
            picked.append(int(g))
        # diversity half: greedy max-min distance to reviewed + picked, in the prior's PCA space
        Z = self.prior.space()
        anchors = [g for g in self.verdicts] + picked
        while len(picked) < min(b, len(cand)):
            rest = [g for g in cand if g not in picked]
            if not anchors:
                g = rest[0]
            else:
                A = Z[np.asarray(anchors)]
                d = np.min(((Z[np.asarray(rest)][:, None, :] - A[None, :, :]) ** 2).sum(-1), axis=1)
                g = rest[int(np.argmax(d))]          # first max wins = higher priority among equals
            picked.append(int(g))
            anchors.append(int(g))
        return picked
