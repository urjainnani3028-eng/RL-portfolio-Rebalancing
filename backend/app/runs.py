"""Experiment runs: multi-seed training, walk-forward, persistence, one-shot test evaluation,
and assembling results (with baselines + deflated Sharpe) for the UI."""
from __future__ import annotations

import json
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .agents import evaluate, load_model, train_one_seed
from .baselines import market_log_ret, run_specs, target_specs
from .config import RUNS_DIR, AgentSettings, Settings
from .data import Dataset
from .jobs import Job
from .metrics import compute_metrics, deflated_sharpe, drawdown_series, summarize_seeds
from .sim import make_bundle
from . import state


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def run_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def read_meta(run_id: str) -> Optional[dict]:
    p = run_dir(run_id) / "meta.json"
    return json.loads(p.read_text()) if p.exists() else None


def write_meta(meta: dict) -> None:
    d = run_dir(meta["id"])
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=1, default=_json_default))
    tmp.replace(d / "meta.json")


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_curves(run_id: str, segment: str, seed: int, h: dict) -> None:
    d = run_dir(run_id) / "curves"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{segment}_seed{seed}.json").write_text(json.dumps(h, default=_json_default))


def load_curves(run_id: str, segment: str, seed: int) -> Optional[dict]:
    p = run_dir(run_id) / "curves" / f"{segment}_seed{seed}.json"
    return json.loads(p.read_text()) if p.exists() else None


def list_runs() -> list[dict]:
    out = []
    for d in sorted(RUNS_DIR.glob("*/meta.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            m = json.loads(d.read_text())
        except Exception:  # noqa: BLE001
            continue
        out.append({k: m.get(k) for k in ("id", "name", "created", "status", "kind", "agent", "summary",
                                          "dataset", "test", "error")})
    return out


def delete_run(run_id: str) -> bool:
    d = run_dir(run_id)
    if d.exists() and d.parent == RUNS_DIR:
        shutil.rmtree(d)
        return True
    return False


def _default_name(agent: AgentSettings) -> str:
    if agent.mode == "hybrid":
        return f"{agent.algo}-{agent.arch.upper()} · hybrid[{agent.target}] · {agent.reward.risk}"
    return f"{agent.algo}-{agent.arch.upper()} · direct · {agent.reward.risk}"


def _new_meta(kind: str, s: Settings, agent: AgentSettings, ds: Dataset) -> dict:
    rid = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    return {"id": rid, "kind": kind, "name": agent.name or _default_name(agent), "created": datetime.now().isoformat(
        timespec="seconds"), "status": "running", "settings": s.model_dump(), "agent": agent.model_dump(),
        "dataset": {k: ds.meta.get(k) for k in ("id", "source", "sha", "start", "end", "freq", "bars")},
        "seeds": [], "summary": {}, "test": {"touched": 0, "history": []}}


def run_settings(meta: dict) -> tuple[Settings, AgentSettings]:
    return Settings(**meta["settings"]), AgentSettings(**meta["agent"])


def run_dataset(meta: dict) -> Dataset:
    s, _ = run_settings(meta)
    ds = state.dataset_for(meta["dataset"]["id"], s)
    if ds is None:
        raise FileNotFoundError("The dataset snapshot used by this run no longer exists.")
    return ds


# ---------------------------------------------------------------------------
# Standard multi-seed training job
# ---------------------------------------------------------------------------
def execute_training(job: Job, s: Settings, agent: AgentSettings, ds: Dataset) -> dict:
    meta = _new_meta("train", s, agent, ds)
    job.params["run_id"] = meta["id"]
    write_meta(meta)
    b = state.get_bundle(ds, s.portfolio)
    if agent.mode == "hybrid":
        job.update(0.0, f"Precomputing {agent.target} targets")
        b.targets(agent.target)
    mlr = {seg: market_log_ret(b, s.costs, seg) for seg in ("train", "val")}
    (run_dir(meta["id"]) / "models").mkdir(parents=True, exist_ok=True)
    n = agent.seeds
    try:
        for i in range(n):
            seed = agent.seed_offset + i
            job.log(f"Seed {seed}: training {agent.algo}/{agent.arch} for {agent.timesteps:,} steps")
            t0 = time.time()

            def on_prog(step, ep_r, _i=i, _seed=seed):
                job.update((_i + step / agent.timesteps) / n,
                           f"Seed {_seed} ({_i + 1}/{n}) · step {step:,}/{agent.timesteps:,}")
                if ep_r is not None:
                    job.series.append({"seed": _seed, "step": step, "ep_reward": round(ep_r, 4)})

            def on_eval(step, met, _seed=seed):
                job.evals.append({"seed": _seed, "step": step, "sharpe": met.get("sharpe"),
                                  "cagr": met.get("cagr"), "max_drawdown": met.get("max_drawdown"),
                                  "turnover_ann": met.get("turnover_ann")})

            path = run_dir(meta["id"]) / "models" / f"seed{seed}.zip"
            info = train_one_seed(b, agent, s.costs, seed, path, on_prog, on_eval, lambda: job.cancel_requested)
            if info.get("cancelled"):
                meta["status"] = "cancelled"
                write_meta(meta)
                return {"run_id": meta["id"], "cancelled": True}
            model = load_model(path, agent)
            rec = {"seed": seed, "best_step": info["best_step"], "train_seconds": round(time.time() - t0, 1),
                   "metrics": {}}
            for seg in ("train", "val"):
                h, met = evaluate(model, b, agent, s.costs, seg, market_lr=mlr[seg])
                rec["metrics"][seg] = met
                save_curves(meta["id"], seg, seed, h)
            meta["seeds"].append(rec)
            meta["summary"] = {seg: summarize_seeds([r["metrics"][seg] for r in meta["seeds"]])
                               for seg in ("train", "val")}
            write_meta(meta)
            job.log(f"Seed {seed}: val Sharpe {rec['metrics']['val'].get('sharpe'):.3f} "
                    f"(best checkpoint @ {info['best_step']:,})")
        meta["status"] = "done"
    except Exception as e:  # noqa: BLE001
        meta["status"], meta["error"] = "failed", str(e)
        write_meta(meta)
        raise
    write_meta(meta)
    return {"run_id": meta["id"]}


# ---------------------------------------------------------------------------
# Purged walk-forward: retrain every year, evaluate on the next, stitch OOS curve
# ---------------------------------------------------------------------------
def execute_walkforward(job: Job, s: Settings, agent: AgentSettings, ds: Dataset) -> dict:
    meta = _new_meta("walkforward", s, agent, ds)
    meta["test"]["note"] = "Walk-forward OOS spans validation AND test years."
    job.params["run_id"] = meta["id"]
    write_meta(meta)
    dates = pd.DatetimeIndex(ds.dates)
    emb = s.data.embargo_bars
    first_year = pd.Timestamp(s.data.train_end).year + 1
    years = sorted({d.year for d in dates if d.year >= first_year})
    folds = []
    for y in years:
        idx_start = int(np.searchsorted(dates, pd.Timestamp(f"{y}-01-01")))
        idx_end = int(np.searchsorted(dates, pd.Timestamp(f"{y}-12-31"), side="right")) - 1
        tr_end = idx_start - 1 - emb
        if idx_end - idx_start < 8 or tr_end < 60:
            continue
        folds.append({"year": y, "train_end": str(dates[tr_end].date()), "val_end": str(dates[idx_end].date())})
    meta["folds"] = folds
    (run_dir(meta["id"]) / "models").mkdir(parents=True, exist_ok=True)
    n_total = len(folds) * agent.seeds
    stitched = {agent.seed_offset + i: None for i in range(agent.seeds)}
    final_w = {k: None for k in stitched}
    done = 0
    for f in folds:
        fds = Dataset(id=ds.id, prices=ds.prices, meta={**ds.meta, "train_end": f["train_end"],
                                                         "val_end": f["val_end"], "embargo_bars": emb})
        b = make_bundle(fds, s.portfolio)
        job.log(f"Fold {f['year']}: train ≤ {f['train_end']} (embargo {emb} bars), test {f['year']}")
        for i in range(agent.seeds):
            seed = agent.seed_offset + i

            def on_prog(step, ep_r, _d=done, _seed=seed, _y=f["year"]):
                job.update((_d + step / agent.timesteps) / n_total, f"Fold {_y} · seed {_seed} · step {step:,}")
                if ep_r is not None:
                    job.series.append({"seed": _seed, "step": _d * agent.timesteps + step, "ep_reward": round(ep_r, 4)})

            path = run_dir(meta["id"]) / "models" / f"fold{f['year']}_seed{seed}.zip"
            info = train_one_seed(b, agent, s.costs, seed, path, on_prog, lambda *_: None,
                                  lambda: job.cancel_requested, val_segment=None)
            if info.get("cancelled"):
                meta["status"] = "cancelled"
                write_meta(meta)
                return {"run_id": meta["id"], "cancelled": True}
            model = load_model(path, agent)
            from .sim import PortfolioEnv, rollout_env
            env = PortfolioEnv(b, agent, s.costs, segment="val", training=False)
            h = rollout_env(env, lambda o: model.predict(o, deterministic=True)[0], w0=final_w[seed])
            final_w[seed] = h.pop("final_w")
            prev = stitched[seed]
            if prev is None:
                stitched[seed] = h
            else:
                scale = prev["value"][-1]
                for k in ("dates", "log_ret", "cost", "turnover", "weights", "a"):
                    prev[k] += h[k]
                prev["value"] += [v * scale for v in h["value"]]
            done += 1
    # metrics on the stitched out-of-sample curve
    b_full = state.get_bundle(ds, s.portfolio)
    span = _span_from_dates(b_full, stitched[agent.seed_offset]["dates"]) if folds else None
    mlr = market_log_ret(b_full, s.costs, span) if span else None
    for seed, h in stitched.items():
        if h is None:
            continue
        met = compute_metrics(h, b_full.ppy, b_full.rf_per_bar, mlr)
        save_curves(meta["id"], "oos", seed, h)
        meta["seeds"].append({"seed": seed, "metrics": {"oos": met}})
    meta["summary"] = {"oos": summarize_seeds([r["metrics"]["oos"] for r in meta["seeds"]])}
    meta["oos_span"] = list(span) if span else None
    meta["status"] = "done"
    write_meta(meta)
    return {"run_id": meta["id"]}


def _span_from_dates(b, dates: list[str]) -> tuple[int, int]:
    """Decision-index span whose earned bars are exactly `dates`."""
    all_d = pd.DatetimeIndex(b.dates)
    first = int(all_d.get_loc(pd.Timestamp(dates[0])))
    last = int(all_d.get_loc(pd.Timestamp(dates[-1])))
    return first - 1, last


# ---------------------------------------------------------------------------
# One-shot test evaluation (the test block is touched once, and we log it)
# ---------------------------------------------------------------------------
def evaluate_test(run_id: str) -> dict:
    meta = read_meta(run_id)
    if meta is None:
        raise FileNotFoundError(run_id)
    if meta.get("kind") != "train":
        raise ValueError("Test evaluation applies to standard runs (walk-forward already reports OOS).")
    s, agent = run_settings(meta)
    ds = run_dataset(meta)
    b = state.get_bundle(ds, s.portfolio)
    mlr = market_log_ret(b, s.costs, "test")
    for rec in meta["seeds"]:
        model = load_model(run_dir(run_id) / "models" / f"seed{rec['seed']}.zip", agent)
        h, met = evaluate(model, b, agent, s.costs, "test", market_lr=mlr)
        rec["metrics"]["test"] = met
        save_curves(run_id, "test", rec["seed"], h)
    meta["summary"]["test"] = summarize_seeds([r["metrics"]["test"] for r in meta["seeds"]])
    meta["test"]["touched"] = meta["test"].get("touched", 0) + 1
    meta["test"]["history"].append(datetime.now().isoformat(timespec="seconds"))
    write_meta(meta)
    return meta


# ---------------------------------------------------------------------------
# Results assembly
# ---------------------------------------------------------------------------
def all_trial_sharpes(segment: str) -> list[float]:
    """Per-bar Sharpe of every seed of every run on `segment` -> multiplicity for deflated Sharpe."""
    out = []
    for p in RUNS_DIR.glob("*/meta.json"):
        try:
            m = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        for r in m.get("seeds", []):
            v = r.get("metrics", {}).get(segment, {}).get("sharpe_per_bar")
            if v is not None:
                out.append(v)
    return out


def _thin(h: dict, keep_weights=True) -> dict:
    out = {"dates": h["dates"], "value": [round(v, 6) for v in h["value"]],
           "drawdown": drawdown_series(h["value"]),
           "turnover": [round(v, 5) for v in h["turnover"]]}
    if keep_weights:
        out["weights"] = [[round(x, 4) for x in w] for w in h["weights"]]
        out["a"] = [None if (a is None or (isinstance(a, float) and np.isnan(a))) else round(a, 4) for a in h["a"]]
    return out


def run_results(run_id: str, segment: str) -> dict:
    meta = read_meta(run_id)
    if meta is None:
        raise FileNotFoundError(run_id)
    s, agent = run_settings(meta)
    if meta.get("kind") == "walkforward":
        segment = "oos"
    if segment == "test" and not meta.get("test", {}).get("touched"):
        return {"meta": _public_meta(meta), "locked": True, "segment": segment}
    ds = run_dataset(meta)
    b = state.get_bundle(ds, s.portfolio)
    seg_arg = tuple(meta["oos_span"]) if segment == "oos" and meta.get("oos_span") else segment
    trials = all_trial_sharpes(segment)
    seeds = []
    curves = {}
    for rec in meta["seeds"]:
        met = rec["metrics"].get(segment)
        if not met:
            continue
        seeds.append({"seed": rec["seed"], "best_step": rec.get("best_step"), "metrics": met,
                      "dsr": deflated_sharpe(met, trials)})
        h = load_curves(run_id, segment, rec["seed"])
        if h:
            curves[rec["seed"]] = _thin(h)
    specs_res = state.get_baselines(ds, s, segment) if isinstance(seg_arg, str) else None
    if specs_res is None:
        from .baselines import strategy_specs
        specs_res = run_specs(b, s.costs, seg_arg, strategy_specs(b))
    baselines = [{"id": v["id"], "name": v["name"], "family": v["family"], "metrics": v["metrics"],
                  "curve": _thin(v["history"], keep_weights=False)} for v in specs_res.values()]
    if agent.mode == "hybrid":
        mlr = np.asarray(specs_res["market"]["history"]["log_ret"]) if "market" in specs_res else None
        ref = run_specs(b, s.costs, seg_arg, target_specs(b, agent.target), market_lr=mlr)
        baselines += [{"id": v["id"], "name": v["name"], "family": v["family"], "metrics": v["metrics"],
                       "curve": _thin(v["history"], keep_weights=False)} for v in ref.values()]
    bar = next((x["metrics"].get("sharpe") for x in baselines if x["id"] == "ew_quarterly"), None)
    return {"meta": _public_meta(meta), "segment": segment, "locked": False, "assets": b.assets + ["CASH"],
            "seeds": seeds, "summary": meta.get("summary", {}).get(segment, {}), "curves": curves,
            "baselines": baselines, "bar_to_beat": bar, "n_trials": len(trials)}


def _public_meta(meta: dict) -> dict:
    return {k: meta.get(k) for k in ("id", "name", "kind", "created", "status", "agent", "settings", "dataset",
                                     "test", "folds", "error")}
