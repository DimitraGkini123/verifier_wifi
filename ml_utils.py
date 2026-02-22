# ml_utils.py
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class PerDeviceStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, n_features: int, eps: float = 1e-8):
        self.n_features = n_features
        self.eps = eps

    def fit(self, X, y=None):
        X = np.asarray(X)
        feats = X[:, :self.n_features].astype(np.float64)
        devs  = X[:, self.n_features].astype(np.int64)

        self.global_mean_ = feats.mean(axis=0)
        self.global_std_  = np.maximum(feats.std(axis=0), self.eps)

        self.stats_ = {}
        for d in np.unique(devs):
            idx = (devs == d)
            m = feats[idx].mean(axis=0)
            s = np.maximum(feats[idx].std(axis=0), self.eps)
            self.stats_[int(d)] = (m, s)
        return self

    def transform(self, X):
        X = np.asarray(X)
        feats = X[:, :self.n_features].astype(np.float64)
        devs  = X[:, self.n_features].astype(np.int64)

        out = np.empty_like(feats, dtype=np.float64)
        for i in range(feats.shape[0]):
            d = int(devs[i])
            if d in self.stats_:
                m, s = self.stats_[d]
            else:
                m, s = self.global_mean_, self.global_std_
            out[i] = (feats[i] - m) / s
        return out.astype(np.float32)
