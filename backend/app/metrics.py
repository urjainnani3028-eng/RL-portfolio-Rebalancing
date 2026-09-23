"""Performance metrics, all computed on net-of-cost returns."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def _safe(x):
    x = float(x)
    return None if (math.isnan(x) or math.isinf(x)) else round(x, 6)


def compute_metrics(h: dict, ppy: int, rf_per_bar: float, market_log_ret: np.ndarray | None = None) -> dict:
    lr = np.asarray(h["log_ret"], float)
    if len(lr) < 2:
        return {}
    r = np.expm1(lr)
    ex = r - rf_per_bar
    years = len(lr) / ppy
    value = np.asarray(h["value"], float)
    total = value[-1] / 1.0
    cagr = total ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std(ddof=1) * np.sqrt(ppy)
    sharpe = ex.mean() / (ex.std(ddof=1) + 1e-12) * np.sqrt(ppy)
    downside = np.sqrt(np.mean(np.minimum(ex, 0) ** 2)) * np.sqrt(ppy)
    sortino = ex.mean() * ppy / (downside + 1e-12)
    curve = np.concatenate([[1.0], value])
    dd = 1 - curve / np.maximum.accumulate(curve)
    mdd = dd.max()
    calmar = cagr / mdd if mdd > 1e-9 else np.nan
    W = np.asarray(h["weights"], float)
    risky = W[:, :-1]
    hhi = (risky ** 2).sum(1)
    turnover = np.asarray(h["turnover"], float)
    cost = np.asarray(h["cost"], float)
    out = {
        "total_return": _safe(total - 1),
        "cagr": _safe(cagr),
        "vol": _safe(vol),
        "sharpe": _safe(sharpe),
        "sortino": _safe(sortino),
        "max_drawdown": _safe(mdd),
        "calmar": _safe(calmar),
        "turnover_ann": _safe(turnover.sum() / years),
        "total_cost": _safe(cost.sum()),
        "herfindahl": _safe(hhi.mean()),
        "avg_cash": _safe(W[:, -1].mean()),
        "max_weight_seen": _safe(risky.max()),
        "n_trades": int((turnover > 1e-6).sum()),
        "bars": int(len(lr)),
        "sharpe_per_bar": _safe(ex.mean() / (ex.std(ddof=1) + 1e-12)),
        "skew": _safe(stats.skew(ex)),
        "kurt": _safe(stats.kurtosis(ex, fisher=False)),
    }
    if market_log_ret is not None and len(market_log_ret) == len(lr):
        m = np.expm1(market_log_ret) - rf_per_bar
        if m.std() > 0:
            res = stats.linregress(m, ex)
            out["alpha_ann"] = _safe(res.intercept * ppy)
            out["beta"] = _safe(res.slope)
            out["alpha_t"] = _safe(res.intercept / (res.intercept_stderr + 1e-12))
    return out


# ---------------------------------------------------------------------------
# Probabilistic / Deflated Sharpe (Bailey & Lopez de Prado, 2012/2014)
# ---------------------------------------------------------------------------
def probabilistic_sharpe(sr: float, sr_star: float, n: int, skew: float, kurt: float) -> float:
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    return float(stats.norm.cdf((sr - sr_star) * math.sqrt(max(n - 1, 1)) / denom))


def expected_max_sharpe(trial_srs: list[float]) -> float:
    """Expected maximum per-bar Sharpe of N independent trials under the null (true SR=0)."""
    n = len(trial_srs)
    if n < 2:
        return 0.0
    var = float(np.var(trial_srs, ddof=1))
    g = 0.5772156649
    return math.sqrt(var) * ((1 - g) * stats.norm.ppf(1 - 1 / n) + g * stats.norm.ppf(1 - 1 / (n * math.e)))


def deflated_sharpe(m: dict, trial_srs: list[float]) -> dict:
    sr = m.get("sharpe_per_bar")
    if sr is None:
        return {}
    sr0 = expected_max_sharpe(trial_srs)
    return {
        "psr": _safe(probabilistic_sharpe(sr, 0.0, m["bars"], m["skew"] or 0, m["kurt"] or 3)),
        "dsr": _safe(probabilistic_sharpe(sr, sr0, m["bars"], m["skew"] or 0, m["kurt"] or 3)),
        "n_trials": len(trial_srs),
    }


def summarize_seeds(metric_list: list[dict], keys=None) -> dict:
    keys = keys or ["sharpe", "cagr", "vol", "sortino", "max_drawdown", "calmar", "turnover_ann",
                    "total_cost", "herfindahl", "avg_cash", "alpha_ann"]
    out = {}
    for k in keys:
        v = np.array([m.get(k) for m in metric_list if m.get(k) is not None], float)
        if len(v):
            out[k] = {"median": _safe(np.median(v)), "q1": _safe(np.percentile(v, 25)),
                      "q3": _safe(np.percentile(v, 75)), "min": _safe(v.min()), "max": _safe(v.max())}
    return out


def drawdown_series(values: list[float]) -> list[float]:
    v = np.concatenate([[1.0], np.asarray(values, float)])
    return (v / np.maximum.accumulate(v) - 1)[1:].round(5).tolist()
