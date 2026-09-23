"""FastAPI backend for the RL portfolio rebalancing lab."""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import analysis, runs, sanity, state
from .baselines import strategy_specs
from .config import CASH, CATALOGUE, MARKET_TICKER, AgentSettings, Settings
from .costs import CostModel
from .data import asset_info, dataset_from_csv, split_indices
from .jobs import Job, manager
from .metrics import drawdown_series

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="RL Portfolio Rebalancing Lab", version="1.0")
SANITY: dict = {"fast": None, "slow": None, "dataset": None}
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _require_ds():
    ds = state.get_active()
    if ds is None:
        raise HTTPException(409, "No dataset prepared yet. Open the Data page and click 'Prepare data'.")
    return ds


def _err(e: Exception, code=400):
    raise HTTPException(code, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Meta & settings
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/meta")
def meta():
    return {
        "catalogue": CATALOGUE, "market": MARKET_TICKER,
        "algos": [
            {"id": "PPO", "label": "PPO", "note": "Stable, forgiving - start here", "modes": ["hybrid", "direct"]},
            {"id": "A2C", "label": "A2C", "note": "Fast, noisier on-policy", "modes": ["hybrid", "direct"]},
            {"id": "SAC", "label": "SAC", "note": "Sample-efficient, slower per step", "modes": ["hybrid", "direct"]},
            {"id": "TD3", "label": "TD3", "note": "Deterministic off-policy", "modes": ["hybrid", "direct"]},
            {"id": "DQN", "label": "DQN", "note": "Discrete timing: 0-100% move in 25% steps", "modes": ["hybrid"]},
        ],
        "archs": [
            {"id": "mlp", "label": "MLP", "note": "Flat baseline network"},
            {"id": "eiie", "label": "EIIE", "note": "Weight-shared per-asset evaluator (Jiang et al. 2017)"},
            {"id": "attention", "label": "Cross-asset attention", "note": "Transformer layer across assets"},
        ],
        "targets": [
            {"id": "risk_parity", "label": "Risk parity (ERC)"}, {"id": "min_variance", "label": "Min-variance"},
            {"id": "hrp", "label": "HRP"}, {"id": "mean_variance", "label": "Mean-variance"},
            {"id": "equal_weight", "label": "Equal-weight"},
        ],
        "defaults": {"settings": Settings().model_dump(), "agent": AgentSettings().model_dump()},
    }


@app.get("/api/settings")
def get_settings():
    return state.load_settings().model_dump()


@app.put("/api/settings")
def put_settings(s: Settings):
    state.save_settings(s)
    ds = state.get_active()
    if ds is not None:
        ds.meta.update({"train_end": s.data.train_end, "val_end": s.data.val_end, "embargo_bars": s.data.embargo_bars})
    return s.model_dump()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class PrepareReq(BaseModel):
    settings: Optional[Settings] = None
    refresh: bool = False


@app.post("/api/data/prepare")
def prepare(req: PrepareReq):
    s = req.settings or state.load_settings()
    if len(s.data.tickers) < 2:
        raise HTTPException(400, "Select at least two ETFs.")
    try:
        state.save_settings(s)
        ds = state.ensure_dataset(s, refresh=req.refresh)
    except Exception as e:  # noqa: BLE001
        _err(e, 502)
    return data_summary()


@app.post("/api/data/upload")
async def upload(file: UploadFile = File(...)):
    s = state.load_settings()
    try:
        ds = dataset_from_csv(await file.read(), s.data, s.portfolio.rf_annual)
        s.data.tickers = ds.assets
        state.set_active(ds, s)
    except Exception as e:  # noqa: BLE001
        _err(e)
    return data_summary()


@app.get("/api/data/summary")
def data_summary():
    ds = state.get_active()
    if ds is None:
        return {"ready": False}
    s = state.load_settings()
    p = ds.prices
    segs = split_indices(ds)
    dates = [str(d.date()) for d in p.index]
    norm = (p / p.iloc[0] * 100).round(3)
    r = ds.returns
    ppy = ds.periods_per_year()
    stats = []
    for c in p.columns:
        row = {"ticker": c}
        for seg, (a, b) in segs.items():
            x = np.log1p(r[c].values[a + 1:b + 1])
            if len(x) > 2:
                row[seg] = {"ann_return": float(np.exp(x.mean() * ppy) - 1), "vol": float(x.std() * np.sqrt(ppy))}
        stats.append(row)
    tr = r.iloc[segs["train"][0] + 1:segs["train"][1] + 1][ds.assets]
    corr = tr.corr().round(3).values.tolist()
    info = asset_info(ds)
    cm = CostModel(s.costs, [i["equity"] for i in info[:-1]])
    return {
        "ready": True, "meta": ds.meta, "assets": info, "dates": dates,
        "prices": {c: norm[c].tolist() for c in p.columns},
        "segments": {k: {"start": dates[max(a, 0)], "end": dates[min(b, len(dates) - 1)], "bars": b - a}
                     for k, (a, b) in segs.items()},
        "stats": stats, "corr": {"assets": ds.assets, "matrix": corr},
        "costs": {"assets": ds.assets, **cm.breakdown()}, "settings": s.model_dump(),
    }


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
@app.get("/api/baselines")
def baselines(segment: str = "val"):
    ds = _require_ds()
    s = state.load_settings()
    try:
        res = state.get_baselines(ds, s, segment)
    except Exception as e:  # noqa: BLE001
        _err(e, 500)
    out = []
    for v in res.values():
        h = v["history"]
        out.append({"id": v["id"], "name": v["name"], "family": v["family"], "metrics": v["metrics"],
                    "curve": {"dates": h["dates"], "value": [round(x, 6) for x in h["value"]],
                              "drawdown": drawdown_series(h["value"])},
                    "weights_last": h["weights"][-1] if h["weights"] else None})
    bar = next((x["metrics"].get("sharpe") for x in out if x["id"] == "ew_quarterly"), None)
    return {"segment": segment, "strategies": out, "bar_to_beat": bar, "assets": ds.assets + [CASH]}


# ---------------------------------------------------------------------------
# Sanity tests
# ---------------------------------------------------------------------------
@app.get("/api/sanity")
def sanity_state():
    ds = state.get_active()
    return SANITY if (ds is not None and SANITY.get("dataset") == ds.id) else {"fast": None, "slow": None}


@app.post("/api/sanity/fast")
def sanity_fast():
    ds = _require_ds()
    s = state.load_settings()
    out = []
    for f in sanity.FAST:
        try:
            out.append(f(ds, s))
        except Exception as e:  # noqa: BLE001
            out.append({"id": f.__name__, "name": f.__name__, "passed": False, "detail": f"Error: {e}", "numbers": {}})
    if SANITY.get("dataset") != ds.id:
        SANITY["slow"] = None
    SANITY.update(fast=out, dataset=ds.id)
    return {"results": out}


@app.post("/api/sanity/slow")
def sanity_slow():
    ds = _require_ds()
    s = state.load_settings()

    def fn(job: Job):
        results = []
        for i, f in enumerate(sanity.SLOW):
            job.update(i / len(sanity.SLOW), f"Running: {f.__name__.replace('test_', '')}")

            def prog(step, _r, _i=i):
                job.update((_i + step / 20000) / len(sanity.SLOW))

            results.append(f(ds, s, progress=prog, stop=lambda: job.cancel_requested))
            job.result = {"results": list(results)}
        if SANITY.get("dataset") != ds.id:
            SANITY["fast"] = None
        SANITY.update(slow=results, dataset=ds.id)
        return {"results": results}

    j = manager.submit(Job("sanity", "Sanity tests 4-5 (shuffled / look-ahead)", fn))
    return j.to_dict()


# ---------------------------------------------------------------------------
# Training & jobs
# ---------------------------------------------------------------------------
@app.post("/api/train")
def train(agent: AgentSettings):
    ds = _require_ds()
    s = state.load_settings()
    if agent.algo == "DQN" and agent.mode != "hybrid":
        raise HTTPException(400, "DQN is only available in hybrid (timing-only) mode.")
    agent.timesteps = int(max(2000, min(agent.timesteps, 2_000_000)))
    agent.seeds = int(max(1, min(agent.seeds, 20)))
    kind = "walkforward" if agent.walk_forward else "train"
    fn = (lambda job: runs.execute_walkforward(job, s, agent, ds)) if agent.walk_forward else \
        (lambda job: runs.execute_training(job, s, agent, ds))
    title = ("Walk-forward · " if agent.walk_forward else "") + (agent.name or runs._default_name(agent))
    j = manager.submit(Job(kind, title, fn, params={"agent": agent.model_dump()}))
    return j.to_dict()


@app.get("/api/jobs")
def jobs():
    return manager.list()


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    j = manager.get(job_id)
    if not j:
        raise HTTPException(404, "Job not found")
    return j.to_dict()


@app.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str):
    return {"ok": manager.cancel(job_id)}


# ---------------------------------------------------------------------------
# Runs & analysis
# ---------------------------------------------------------------------------
@app.get("/api/runs")
def list_runs():
    return runs.list_runs()


@app.get("/api/runs/{run_id}")
def get_run(run_id: str, segment: str = "val"):
    try:
        return runs.run_results(run_id, segment)
    except FileNotFoundError as e:
        _err(e, 404)
    except Exception as e:  # noqa: BLE001
        _err(e, 500)


@app.delete("/api/runs/{run_id}")
def del_run(run_id: str):
    return {"ok": runs.delete_run(run_id)}


@app.post("/api/runs/{run_id}/test")
def test_run(run_id: str):
    try:
        runs.evaluate_test(run_id)
        return runs.run_results(run_id, "test")
    except Exception as e:  # noqa: BLE001
        _err(e)


@app.get("/api/analysis/cost-sweep/{run_id}")
def cost_sweep(run_id: str, segment: str = "val"):
    try:
        return analysis.cost_sweep(run_id, segment)
    except Exception as e:  # noqa: BLE001
        _err(e)


@app.get("/api/analysis/regimes/{run_id}")
def regimes(run_id: str, segment: str = "val"):
    try:
        return analysis.regimes(run_id, segment)
    except Exception as e:  # noqa: BLE001
        _err(e)


@app.get("/api/analysis/scatter")
def scatter(segment: str = "val"):
    try:
        return analysis.scatter(segment)
    except Exception as e:  # noqa: BLE001
        _err(e)


@app.get("/api/overview")
def overview():
    ds = state.get_active()
    s = state.load_settings()
    out = {"dataset": ds.meta if ds else None, "runs": len(runs.list_runs()), "bar_to_beat": None, "best": None}
    if ds is not None:
        try:
            res = state.get_baselines(ds, s, "val")
            out["bar_to_beat"] = res["ew_quarterly"]["metrics"].get("sharpe")
        except Exception:  # noqa: BLE001
            pass
    best = None
    for r in runs.list_runs():
        v = ((r.get("summary") or {}).get("val") or {}).get("sharpe", {}).get("median")
        if v is not None and (best is None or v > best["sharpe"]):
            best = {"run_id": r["id"], "name": r["name"], "sharpe": v}
    out["best"] = best
    out["active_jobs"] = [j for j in manager.list() if j["status"] in ("queued", "running")]
    out["sanity"] = SANITY if (ds is not None and SANITY.get("dataset") == ds.id) else {"fast": None, "slow": None}
    return out


# ---------------------------------------------------------------------------
# Serve the built frontend (production mode: one command, one port)
# ---------------------------------------------------------------------------
DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = DIST / path
        if path and f.exists() and f.is_file():
            return FileResponse(f)
        return FileResponse(DIST / "index.html")
