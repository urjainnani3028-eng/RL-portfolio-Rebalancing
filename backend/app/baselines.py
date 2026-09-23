"""Rule-based baselines. Every baseline runs through the *same* simulator and cost model
as the RL agent, starting from 100% cash, so comparisons are like-for-like."""
from __future__ import annotations

import numpy as np

from .config import MARKET_TICKER, CostSettings
from .metrics import compute_metrics
from .sim import Bundle, run_policy


def _bars(b: Bundle, kind: str) -> int:
    if b.ppy == 52:
        return {"monthly": 4, "quarterly": 13}[kind]
    return {"monthly": 21, "quarterly": 63}[kind]


def _ew(b: Bundle) -> np.ndarray:
    return np.append(np.full(b.n, 1 / b.n), 0.0)


def market_index(b: Bundle) -> int:
    return b.assets.index(MARKET_TICKER) if MARKET_TICKER in b.assets else 0


def strategy_specs(b: Bundle) -> list[dict]:
    m, q = _bars(b, "monthly"), _bars(b, "quarterly")
    ew = _ew(b)
    mi = market_index(b)

    def periodic(fn, every):
        return lambda t, w, k: fn(t) if k % every == 0 else None

    def threshold(tgt, band=0.05):
        return lambda t, w, k: tgt if (k == 0 or np.abs(w - tgt).max() > band) else None

    specs = [
        {"id": "market", "name": f"Market index ({b.assets[mi].replace('.NS', '')})", "family": "Passive",
         "policy": lambda t, w, k: np.eye(b.n + 1)[mi] if k == 0 else None, "cap": False},
        {"id": "bh_ew", "name": "Buy & hold (EW start)", "family": "Passive",
         "policy": lambda t, w, k: ew if k == 0 else None},
        {"id": "ew_monthly", "name": "Equal-weight, monthly", "family": "Calendar",
         "policy": periodic(lambda t: ew, m)},
        {"id": "ew_quarterly", "name": "Equal-weight, quarterly", "family": "Calendar",
         "policy": periodic(lambda t: ew, q)},
        {"id": "ew_threshold", "name": "Equal-weight, 5% threshold", "family": "Threshold",
         "policy": threshold(ew)},
    ]
    for method, label in [("min_variance", "Min-variance"), ("mean_variance", "Mean-variance"),
                          ("risk_parity", "Risk parity"), ("hrp", "HRP")]:
        specs.append({"id": f"{method}_monthly", "name": f"{label}, rolling monthly", "family": "Optimiser",
                      "policy": periodic(lambda t, _m=method: b.targets(_m)[t], m)})
    return specs


def target_specs(b: Bundle, method: str) -> list[dict]:
    """Benchmarks for a hybrid agent: its own optimiser applied mechanically."""
    m = _bars(b, "monthly")
    tp = b.targets(method)
    return [
        {"id": f"tgt_every", "name": f"Target ({method}) every bar", "family": "Hybrid ref",
         "policy": lambda t, w, k: tp[t]},
        {"id": f"tgt_monthly", "name": f"Target ({method}) monthly", "family": "Hybrid ref",
         "policy": lambda t, w, k: tp[t] if k % m == 0 else None},
    ]


def run_specs(b: Bundle, cost: CostSettings, segment: str, specs: list[dict], market_lr=None) -> dict:
    out = {}
    for s in specs:
        h = run_policy(b, cost, segment, s["policy"], enforce_cap=s.get("cap", True))
        out[s["id"]] = {"id": s["id"], "name": s["name"], "family": s["family"], "history": h}
    if market_lr is None and "market" in out:
        market_lr = np.asarray(out["market"]["history"]["log_ret"])
    for v in out.values():
        v["metrics"] = compute_metrics(v["history"], b.ppy, b.rf_per_bar, market_lr)
    return out


def market_log_ret(b: Bundle, cost: CostSettings, segment: str) -> np.ndarray:
    mi = market_index(b)
    h = run_policy(b, cost, segment, lambda t, w, k: np.eye(b.n + 1)[mi] if k == 0 else None, enforce_cap=False)
    return np.asarray(h["log_ret"])


def run_baselines(b: Bundle, cost: CostSettings, segment: str) -> dict:
    return run_specs(b, cost, segment, strategy_specs(b))
