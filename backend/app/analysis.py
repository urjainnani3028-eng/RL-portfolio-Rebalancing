"""Robustness analysis: cost sweep, regime-sliced performance, and the headline
return-vs-drawdown scatter across all runs and baselines."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import state
from .agents import evaluate, load_model
from .baselines import run_specs, strategy_specs, target_specs
from .config import Settings
from .costs import flat_cost_settings
from .runs import list_runs, load_curves, read_meta, run_dataset, run_dir, run_settings

NAMED_WINDOWS = [
    ("Taper tantrum", "2013-05-15", "2013-09-30"),
    ("Demonetisation", "2016-11-01", "2017-01-31"),
    ("IL&FS / NBFC stress", "2018-08-15", "2018-12-31"),
    ("COVID crash", "2020-02-15", "2020-04-30"),
    ("COVID recovery", "2020-05-01", "2021-02-28"),
    ("2022 rate hikes", "2022-01-01", "2022-06-30"),
    ("Adani / Jan-Mar 2023", "2023-01-20", "2023-03-31"),
    ("Oct 2024 - Mar 2025 correction", "2024-10-01", "2025-03-31"),
]


def cost_sweep(run_id: str, segment: str, bps_list=(0, 5, 10, 20, 30, 50)) -> dict:
    meta = read_meta(run_id)
    if meta.get("kind") != "train":
        raise ValueError("Cost sweep is available for standard runs.")
    if segment == "test" and not meta["test"].get("touched"):
        raise PermissionError("Test block is locked for this run. Evaluate on test first.")
    s, agent = run_settings(meta)
    ds = run_dataset(meta)
    b = state.get_bundle(ds, s.portfolio)
    models = [load_model(run_dir(run_id) / "models" / f"seed{r['seed']}.zip", agent) for r in meta["seeds"]]
    out = {"bps": list(bps_list), "rl": [], "baselines": {}}
    base_specs = [x for x in strategy_specs(b) if x["id"] in ("market", "ew_quarterly", "min_variance_monthly",
                                                               "risk_parity_monthly")]
    if agent.mode == "hybrid":
        base_specs += target_specs(b, agent.target)
    for bps in bps_list:
        c = flat_cost_settings(bps)
        sh = [evaluate(m, b, agent, c, segment)[1] for m in models]
        vals = np.array([x["sharpe"] for x in sh], float)
        turn = np.array([x["turnover_ann"] for x in sh], float)
        out["rl"].append({"bps": bps, "median": float(np.median(vals)), "q1": float(np.percentile(vals, 25)),
                          "q3": float(np.percentile(vals, 75)), "turnover": float(np.median(turn))})
        res = run_specs(b, c, segment, base_specs)
        for k, v in res.items():
            out["baselines"].setdefault(k, {"name": v["name"], "sharpe": []})["sharpe"].append(v["metrics"]["sharpe"])
    return out


def _regime_masks(b, dates: list[str], s: Settings, events: bool = True) -> list[tuple[str, str, np.ndarray]]:
    """Label each evaluated bar using information available before it (trailing market stats)."""
    from .baselines import market_index

    all_d = pd.DatetimeIndex(b.dates)
    idx = np.array([all_d.get_loc(pd.Timestamp(d)) for d in dates])
    mi = market_index(b)
    lr = np.log1p(b.R[:, mi])
    w = 26 if b.ppy == 52 else 126
    trail_ret = pd.Series(lr).rolling(w, min_periods=w // 2).sum().shift(1).values
    trail_vol = pd.Series(lr).rolling(w // 2, min_periods=4).std().shift(1).values
    tr_b = b.segments["train"][1]
    thresh = np.nanpercentile(trail_vol[: tr_b + 1], 75)
    tr, tv = trail_ret[idx], trail_vol[idx]
    masks = [("Bull (trailing return > 0)", "state", tr > 0), ("Bear (trailing return ≤ 0)", "state", tr <= 0),
             ("High volatility (> train 75th pct)", "state", tv > thresh),
             ("Calm (≤ train 75th pct)", "state", tv <= thresh)]
    dt = pd.DatetimeIndex(dates)
    for name, a, e in (NAMED_WINDOWS if events else []):
        m = (dt >= pd.Timestamp(a)) & (dt <= pd.Timestamp(e))
        if m.sum() >= 4:
            masks.append((name, "event", np.asarray(m)))
    return masks


def _slice_stats(log_ret: np.ndarray, mask: np.ndarray, ppy: int, rf: float) -> dict:
    x = log_ret[mask]
    if len(x) < 3:
        return {}
    r = np.expm1(x)
    ex = r - rf
    curve = np.exp(np.cumsum(x))
    dd = 1 - curve / np.maximum.accumulate(np.concatenate([[1.0], curve]))[1:]
    return {"ann_return": float(np.exp(x.mean() * ppy) - 1), "vol": float(r.std(ddof=1) * np.sqrt(ppy)),
            "sharpe": float(ex.mean() / (ex.std(ddof=1) + 1e-12) * np.sqrt(ppy)),
            "cum_return": float(np.exp(x.sum()) - 1), "max_drawdown": float(dd.max()), "bars": int(len(x))}


def regimes(run_id: str, segment: str) -> dict:
    meta = read_meta(run_id)
    s, agent = run_settings(meta)
    if meta.get("kind") == "walkforward":
        segment = "oos"
    if segment == "test" and not meta["test"].get("touched"):
        raise PermissionError("Test block is locked for this run. Evaluate on test first.")
    ds = run_dataset(meta)
    b = state.get_bundle(ds, s.portfolio)
    hs = [load_curves(run_id, segment, r["seed"]) for r in meta["seeds"]]
    hs = [h for h in hs if h]
    if not hs:
        raise FileNotFoundError("No curves for this segment")
    dates = hs[0]["dates"]
    seg_arg = tuple(meta["oos_span"]) if segment == "oos" else segment
    specs = [x for x in strategy_specs(b) if x["id"] in ("market", "ew_quarterly", "min_variance_monthly",
                                                          "risk_parity_monthly", "hrp_monthly")]
    if agent.mode == "hybrid":
        specs += target_specs(b, agent.target)[1:]
    base = run_specs(b, s.costs, seg_arg, specs)
    rows = []
    real = ds.meta.get("source") != "synthetic"   # named historical events only make sense on real data
    for name, kind, mask in _regime_masks(b, dates, s, events=real):
        row = {"regime": name, "kind": kind, "bars": int(mask.sum()), "strategies": {}}
        rl = [_slice_stats(np.asarray(h["log_ret"]), mask, b.ppy, b.rf_per_bar) for h in hs]
        rl = [x for x in rl if x]
        if rl:
            row["strategies"]["RL (median of seeds)"] = {k: float(np.median([x[k] for x in rl])) for k in rl[0]}
        for k, v in base.items():
            st = _slice_stats(np.asarray(v["history"]["log_ret"]), mask, b.ppy, b.rf_per_bar)
            if st:
                row["strategies"][v["name"]] = st
        rows.append(row)
    return {"segment": segment, "rows": rows}


def scatter(segment: str) -> dict:
    """Headline chart: annual return vs max drawdown. Baselines = points; each run = a seed cloud."""
    s = state.load_settings()
    ds = state.get_active()
    points = []
    if ds is not None:
        for v in state.get_baselines(ds, s, segment).values():
            m = v["metrics"]
            points.append({"kind": "baseline", "id": v["id"], "name": v["name"], "cagr": m.get("cagr"),
                           "max_drawdown": m.get("max_drawdown"), "sharpe": m.get("sharpe")})
    for r in list_runs():
        meta = read_meta(r["id"])
        if not meta or meta.get("kind") != "train":
            continue
        if meta.get("dataset", {}).get("id") != (ds.id if ds else None):
            continue
        for rec in meta.get("seeds", []):
            m = rec["metrics"].get(segment)
            if m:
                points.append({"kind": "rl", "run_id": meta["id"], "name": meta["name"], "seed": rec["seed"],
                               "cagr": m.get("cagr"), "max_drawdown": m.get("max_drawdown"),
                               "sharpe": m.get("sharpe")})
    return {"segment": segment, "points": points}
