"""Risk models for TRIAGE. P(PASS) per group from its first-occurrence hidden state."""
import numpy as np


def zscore(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-6)


def _pca(Xfit, p, seed=0):
    from sklearn.decomposition import PCA
    p = min(p, Xfit.shape[0] - 1, Xfit.shape[1])
    return PCA(n_components=p, random_state=seed, svd_solver="randomized").fit(Xfit)


def _fit_predict(kind, hp, X, y, Xt, seed):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    if kind == "mlp":
        from sklearn.neural_network import MLPClassifier
        m = MLPClassifier(hidden_layer_sizes=(hp["h"],), alpha=hp["alpha"], early_stopping=True,
                          random_state=seed, max_iter=300)
        m.fit(sc.transform(X), y)
    else:
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=hp["C"], class_weight="balanced", max_iter=3000).fit(sc.transform(X), y)
    return m.predict_proba(sc.transform(Xt))[:, list(m.classes_).index(True)]


class TransferPrior:
    """Experiment D's source model: per-system z-score, PCA(p) fitted label-free on sources + target, then an MLP
    ("mlp") or a logistic regression on PCA + generic log features ("probe_feats") or PCA only ("probe"),
    trained on the source systems' labels. Target verdicts are added to training repeated w times."""

    def __init__(self, source_states, source_labels, kind, hp, w, source_feats=None, seed=0, shared=None):
        """shared: optional dict reused across instances on the same sources (z-scored sources, target transforms)."""
        self.shared = {} if shared is None else shared
        if "Zs" not in self.shared:
            self.shared["Zs"] = [zscore(np.asarray(H, dtype=np.float32)) for H in source_states]
        self.Zs = self.shared["Zs"]
        self.ys = np.concatenate([np.asarray(y, dtype=bool) for y in source_labels])
        self.kind, self.hp, self.w, self.seed = kind, hp, w, seed
        self.Fs = None if source_feats is None else np.vstack(source_feats)
        self.Xs = self.Xt = None
        self._cache = {}

    def set_target(self, H, feats=None):
        """Label-free target transform over the groups seen so far (z-score and PCA refit)."""
        H = np.asarray(H, dtype=np.float32)
        key = ("T", H.shape, hash(H.tobytes()))
        if key not in self.shared:
            Zt = zscore(H)
            pca = _pca(np.vstack(self.Zs + [Zt]), self.hp["p"])
            self.shared[key] = (np.vstack([pca.transform(Z) for Z in self.Zs]), pca.transform(Zt))
        self.Xs, self.Xt = self.shared[key]
        self._tkey = key
        if self.kind == "probe_feats":
            self.Xs, self.Xt = np.hstack([self.Xs, self.Fs]), np.hstack([self.Xt, feats])
        self._cache = {}

    def space(self):
        return self.Xt[:, :self.hp["p"]]

    def predict(self, lab_idx=(), lab_y=(), seed=None):
        seed = self.seed if seed is None else seed
        key = ("P", self._tkey, tuple(lab_idx), tuple(bool(v) for v in lab_y), seed)
        self._cache = self.shared
        if key not in self._cache:
            X, y = self.Xs, self.ys
            if len(lab_idx):
                rep = np.repeat(np.asarray(lab_idx), self.w)
                X = np.vstack([X, self.Xt[rep]])
                y = np.concatenate([y, np.repeat(np.asarray(lab_y, dtype=bool), self.w)])
            self._cache[key] = _fit_predict(self.kind, self.hp, X, y, self.Xt, seed)
        return self._cache[key]


class OwnProbe:
    """Ablation 'no prior': logistic probe on the stream's own verdicts only (z-score + PCA on the stream's groups);
    P(PASS) = 0.5 until there are 2 verdicts of each class."""

    def __init__(self, p, C, seed=0):
        self.p, self.C, self.seed = p, C, seed

    def set_target(self, H, feats=None):
        Zt = zscore(np.asarray(H, dtype=np.float32))
        self.Xt = _pca(Zt, self.p).transform(Zt)

    def space(self):
        return self.Xt

    def predict(self, lab_idx=(), lab_y=(), seed=None):
        y = np.asarray(lab_y, dtype=bool)
        if y.sum() < 2 or (~y).sum() < 2:
            return np.full(len(self.Xt), 0.5)
        return _fit_predict("probe", {"C": self.C}, self.Xt[np.asarray(lab_idx)], y, self.Xt, self.seed)
