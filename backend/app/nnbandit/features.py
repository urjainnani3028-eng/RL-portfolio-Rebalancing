"""Sequence dataset for the forecasters, built from the same Bundle the RL agent uses.

One sample = (asset i, decision bar t). The model sees a window of `lookback` bars ending at
t - exec_delay (the simulator's information set) and predicts the standardised log return of
bar t + 1, which is the return the portfolio earns for a decision taken at t.

Per time step the input has 4 market features + a one-hot asset id:
    z_i        own standardised log return
    vol_i      own standardised rolling volatility
    corr       standardised average pairwise correlation
    xs_mean    cross-sectional mean of z (a "market" factor)
    onehot(i)  so one shared network can still learn asset-specific behaviour
Every scaler was fit on the training block only (see sim._compute_features).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sim import Bundle

N_MARKET_FEATURES = 4


@dataclass
class SeqData:
    X: np.ndarray            # T x N x L x F windows (zeros where t is not a valid decision)
    y: np.ndarray            # T x N standardised next-bar log return (target)
    ret: np.ndarray          # T x N raw next-bar simple return (for directional accuracy)
    valid: np.ndarray        # T bool: decision t has a full window and a next bar
    mu: np.ndarray           # N train-mean of log returns (to de-standardise predictions)
    sd: np.ndarray           # N train-std of log returns
    lookback: int
    delay: int

    @property
    def n_assets(self) -> int:
        return self.y.shape[1]

    @property
    def n_features(self) -> int:
        return self.X.shape[-1]

    def decisions(self, a: int, b: int) -> np.ndarray:
        """Valid decision indices in [a, b)."""
        t = np.arange(a, b)
        return t[self.valid[t]]

    def flat(self, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Stack (t, asset) samples: X -> (len(ts)*N, L, F), y -> (len(ts)*N,)."""
        N = self.n_assets
        return self.X[ts].reshape(len(ts) * N, self.lookback, -1), self.y[ts].reshape(-1)


def build_sequences(b: Bundle) -> SeqData:
    p = b.pcfg
    L, d, N = p.lookback, p.exec_delay, b.n
    T = len(b.R)
    lr = np.log1p(b.R[:, :N])
    tr_b = b.segments["train"][1]
    mu, sd = lr[1:tr_b + 1].mean(0), lr[1:tr_b + 1].std(0) + 1e-8   # same train-only fit as b.z

    xs_mean = b.z.mean(1)
    feats = np.zeros((T, N, N_MARKET_FEATURES + N), dtype=np.float32)
    feats[:, :, 0] = b.z
    feats[:, :, 1] = b.vol_z
    feats[:, :, 2] = b.corr_z[:, None]
    feats[:, :, 3] = xs_mean[:, None]
    feats[:, :, N_MARKET_FEATURES:] = np.eye(N, dtype=np.float32)[None]

    X = np.zeros((T, N, L, feats.shape[-1]), dtype=np.float32)
    y = np.zeros((T, N), dtype=np.float32)
    ret = np.zeros((T, N), dtype=np.float32)
    valid = np.zeros(T, dtype=bool)
    first = max(b.warmup, L + d)
    for t in range(first, T - 1):
        end = t - d                                    # last bar the decision may see
        X[t] = feats[end - L + 1:end + 1].transpose(1, 0, 2)
        y[t] = b.z[t + 1]
        ret[t] = b.R[t + 1, :N]
        valid[t] = True
    return SeqData(X=X, y=y, ret=ret, valid=valid, mu=mu, sd=sd, lookback=L, delay=d)
