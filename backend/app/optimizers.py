"""Classical single-period allocators. In hybrid mode one of these *proposes* the target
portfolio and the RL agent decides how far to move toward it (timing + sizing)."""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform


# ---------------------------------------------------------------------------
# Structural constraints
# ---------------------------------------------------------------------------
def apply_cap(w: np.ndarray, cap: float) -> np.ndarray:
    """Project weights (last entry = cash, uncapped) so every risky weight <= cap,
    long-only and sum-to-one. Excess is water-filled into uncapped risky assets
    proportionally; if every risky asset is capped the remainder goes to cash."""
    w = np.clip(np.asarray(w, dtype=float), 0, None)
    s = w.sum()
    w = w / s if s > 0 else np.eye(len(w))[-1]
    risky = w[:-1].copy()
    cash = w[-1]
    for _ in range(len(risky) + 1):
        over = risky > cap + 1e-12
        if not over.any():
            break
        excess = float((risky[over] - cap).sum())
        risky[over] = cap
        free = risky < cap - 1e-12
        if free.any() and risky[free].sum() > 0:
            risky[free] += excess * risky[free] / risky[free].sum()
        elif free.any():
            risky[free] += excess / free.sum()
        else:
            cash += excess
    out = np.append(risky, cash)
    return out / out.sum()


def _cov(R: np.ndarray, shrink: float = 0.2) -> np.ndarray:
    S = np.cov(R, rowvar=False)
    S = np.atleast_2d(S)
    D = np.diag(np.diag(S))
    return (1 - shrink) * S + shrink * D + 1e-10 * np.eye(len(S))


# ---------------------------------------------------------------------------
# Allocators: input = window of simple returns (L x N risky), output = N weights
# ---------------------------------------------------------------------------
def equal_weight(R: np.ndarray, cap: float) -> np.ndarray:
    n = R.shape[1]
    return np.full(n, 1 / n)


def _qp(S: np.ndarray, mu: np.ndarray, gamma: float, cap: float) -> np.ndarray:
    """min 0.5*gamma*w'Sw - mu'w  s.t. 0 <= w <= cap, sum w = 1   (SLSQP with analytic gradient)."""
    n = len(mu)
    ub = max(cap, 1.0 / n)
    res = minimize(lambda w: 0.5 * gamma * w @ S @ w - mu @ w, np.full(n, 1 / n),
                   jac=lambda w: gamma * S @ w - mu, method="SLSQP", bounds=[(0, ub)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(n)}],
                   options={"maxiter": 100, "ftol": 1e-12})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def min_variance(R: np.ndarray, cap: float) -> np.ndarray:
    S = _cov(R) * 1e4  # rescale for solver tolerance
    return _qp(S, np.zeros(S.shape[0]), 1.0, cap)


def mean_variance(R: np.ndarray, cap: float, gamma: float = 5.0) -> np.ndarray:
    S = _cov(R)
    mu = R.mean(axis=0)
    mu = 0.5 * mu + 0.5 * mu.mean()  # shrink expected returns: the unstable input
    return _qp(S * 1e4, mu * 1e4, gamma, cap)


def risk_parity(R: np.ndarray, cap: float) -> np.ndarray:
    """Equal risk contribution via the convex log-barrier formulation."""
    S = _cov(R)
    n = S.shape[0]
    b = np.full(n, 1 / n)

    def obj(y):
        return 0.5 * y @ S @ y - b @ np.log(y)

    def grad(y):
        return S @ y - b / y

    y0 = 1 / np.sqrt(np.diag(S))
    res = minimize(obj, y0, jac=grad, method="L-BFGS-B", bounds=[(1e-8, None)] * n)
    w = res.x / res.x.sum()
    return w


def hrp(R: np.ndarray, cap: float) -> np.ndarray:
    """Hierarchical Risk Parity (Lopez de Prado, 2016)."""
    S = _cov(R, shrink=0.1)
    n = S.shape[0]
    if n == 1:
        return np.ones(1)
    sd = np.sqrt(np.diag(S))
    C = np.clip(S / np.outer(sd, sd), -1, 1)
    dist = np.sqrt(np.clip(0.5 * (1 - C), 0, None))
    np.fill_diagonal(dist, 0)
    order = list(leaves_list(linkage(squareform(dist, checks=False), method="single")))
    w = np.ones(n)
    clusters = [order]
    while clusters:
        clusters = [c[i:j] for c in clusters for i, j in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for k in range(0, len(clusters), 2):
            c0, c1 = clusters[k], clusters[k + 1]

            def cvar(c):
                sub = S[np.ix_(c, c)]
                ivp = 1 / np.diag(sub)
                ivp /= ivp.sum()
                return ivp @ sub @ ivp

            v0, v1 = cvar(c0), cvar(c1)
            a = 1 - v0 / (v0 + v1)
            w[c0] *= a
            w[c1] *= 1 - a
    return w / w.sum()


ALLOCATORS = {
    "equal_weight": equal_weight,
    "min_variance": min_variance,
    "mean_variance": mean_variance,
    "risk_parity": risk_parity,
    "hrp": hrp,
}


def target_path(returns: np.ndarray, method: str, window: int, cap: float, delay: int = 0,
                every: int = 1) -> np.ndarray:
    """Precompute the allocator's target (incl. cash=0 then capped) for every decision
    index t, using only returns rows <= t - delay. returns: T x (N+1) (cash last)."""
    T, n1 = returns.shape
    n = n1 - 1
    fn = ALLOCATORS[method]
    out = np.zeros((T, n1))
    last = np.append(np.full(n, 1 / n), 0.0)
    last = apply_cap(last, cap)
    for t in range(T):
        hi = t - delay + 1
        if hi - window >= 1 and (t % every == 0):
            R = returns[hi - window:hi, :n]
            try:
                w = fn(R, cap)
                last = apply_cap(np.append(w, 0.0), cap)
            except Exception:  # noqa: BLE001 - keep last good target
                pass
        out[t] = last
    return out
