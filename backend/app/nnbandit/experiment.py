"""Orchestration of the model-selection study.

1. Live bandit   : the chosen algorithm spends `budget` pulls. A pull of arm k trains the next seed of
                   model k (cached; after `max_seeds` seeds it re-tests an existing seed) and scores it on
                   a random `window`-bar slice of the validation block. Training only happens when the
                   bandit asks for it, so the compute that is saved is real.
2. Ground truth  : every model is trained for all `max_seeds` seeds; the exhaustive mean reward over
                   (seed, window) defines the true best arm.
3. Replay study  : every bandit algorithm (and uniform allocation) is replayed `replay_runs` times on the
                   precomputed reward table -> P(identify best arm) and simple regret vs. number of pulls.
4. Comparison    : validation metrics per model (forecast + portfolio), learning curves, lag saliency,
                   Diebold-Mariano tests, baselines.
5. Test (locked) : only with evaluate_test=True. All models + baselines on the 2022+ block, regime slices.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field
from scipy import stats

from ..baselines import market_log_ret, run_specs, strategy_specs
from ..config import STORAGE, CostSettings
from ..metrics import compute_metrics, summarize_seeds
from ..sim import Bundle
from .bandit import ALGORITHMS, make_bandit
from .features import SeqData, build_sequences
from .models import MODEL_IDS, MODEL_INFO
from .portfolio import prediction_metrics, run_forecast_portfolio, sharpe_and_mdd, window_reward
from .training import train_one

NN_DIR = STORAGE / "nnbandit"
CACHE = NN_DIR / "models"
RESULTS = NN_DIR / "results"
for _d in (CACHE, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)

BASELINE_IDS = ("market", "ew_monthly", "ew_quarterly", "min_variance_monthly", "risk_parity_monthly", "hrp_monthly")


class BanditSettings(BaseModel):
    models: list[str] = Field(default_factory=lambda: list(MODEL_IDS))
    algorithm: Literal["uniform", "epsilon_greedy", "ucb1", "ucb1_tuned", "thompson", "successive_halving"] = \
        "successive_halving"   # pure best-arm identification: the goal here is to pick a model
    budget: int = 40                   # pulls the live bandit may spend
    max_seeds: int = 8                 # distinct trained seeds per model (also the ground-truth pool)
    window: int = 26                   # bars per validation evaluation window
    reward_weights: dict[str, float] = Field(
        default_factory=lambda: {"sharpe": 0.5, "accuracy": 0.3, "drawdown": 0.2})
    reward_reference: Literal["ew_quarterly", "none"] = "ew_quarterly"   # score Sharpe / drawdown vs this
    replay_runs: int = 300
    compare_algorithms: list[str] = Field(default_factory=lambda: list(ALGORITHMS))
    # network / training (the project plan: 2 layers x 64 units, dropout 0.1, L2, early stopping)
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 60
    patience: int = 10
    inner_val_bars: int = 52
    # portfolio construction from forecasts
    tilt: float = 5.0
    band: float = 0.02
    seed: int = 0
    evaluate_test: bool = False

    def train_kwargs(self) -> dict:
        return dict(hidden=self.hidden, layers=self.layers, dropout=self.dropout, lr=self.learning_rate,
                    weight_decay=self.weight_decay, batch_size=self.batch_size, max_epochs=self.max_epochs,
                    patience=self.patience, inner_bars=self.inner_val_bars)


def _r(x, nd=5):
    if x is None:
        return None
    x = float(x)
    return None if (math.isnan(x) or math.isinf(x)) else round(x, nd)


# ---------------------------------------------------------------------------
# A trained (model, seed) and its validation scores
# ---------------------------------------------------------------------------
class Record:
    def __init__(self, info: dict, pred: np.ndarray, saliency: np.ndarray, ctx: "Context"):
        self.info, self.pred, self.saliency = info, pred, saliency
        a, bnd = ctx.b.segment("val")
        self.h_val = run_forecast_portfolio(ctx.b, ctx.cost, "val", pred, ctx.cfg.tilt, ctx.cfg.band)
        ts = ctx.data.decisions(a, bnd)
        assert len(ts) == len(self.h_val["log_ret"]), "decision / history misalignment"
        raw = pred[ts] * ctx.data.sd + ctx.data.mu
        self.hits = (np.sign(raw) == np.sign(ctx.data.ret[ts])).mean(1)
        self.log_ret = np.asarray(self.h_val["log_ret"])
        self.ctx = ctx

    def reward(self, start: int) -> tuple[float, dict]:
        w = self.ctx.cfg.window
        ref = self.ctx.ref_windows[start] if self.ctx.ref_windows is not None else None
        return window_reward(self.log_ret[start:start + w], self.hits[start:start + w], self.ctx.b.ppy,
                             self.ctx.b.rf_per_bar, self.ctx.cfg.reward_weights, ref)

    @property
    def n_windows(self) -> int:
        return len(self.log_ret) - self.ctx.cfg.window + 1


class Context:
    def __init__(self, b: Bundle, cost: CostSettings, cfg: BanditSettings, stop: Callable[[], bool]):
        self.b, self.cost, self.cfg, self.stop = b, cost, cfg, stop
        self.data: SeqData = build_sequences(b)
        tk = json.dumps([b.key, b.pcfg.model_dump(), cfg.train_kwargs()], sort_keys=True, default=str)
        self.train_key = hashlib.sha1(tk.encode()).hexdigest()[:12]
        self.records: dict[tuple[str, int], Record] = {}
        self.trained_now: set[tuple[str, int]] = set()
        self.ref_windows = None
        if cfg.reward_reference != "none":
            spec = [x for x in strategy_specs(b) if x["id"] == cfg.reward_reference]
            ref_lr = np.asarray(run_specs(b, cost, "val", spec)[cfg.reward_reference]["history"]["log_ret"])
            self.ref_windows = [sharpe_and_mdd(ref_lr[i:i + cfg.window], b.ppy, b.rf_per_bar)
                                for i in range(len(ref_lr) - cfg.window + 1)]

    def get(self, model_id: str, seed: int) -> Record:
        key = (model_id, seed)
        if key in self.records:
            return self.records[key]
        stem = CACHE / f"{self.train_key}_{model_id}_s{seed}"
        if stem.with_suffix(".json").exists() and stem.with_suffix(".npz").exists():
            info = json.loads(stem.with_suffix(".json").read_text())
            z = np.load(stem.with_suffix(".npz"))
            pred, sal = z["pred"], z["saliency"]
        else:
            out = train_one(model_id, self.data, self.b.segments["train"], seed, stop=self.stop,
                            **self.cfg.train_kwargs())
            pred, sal = out.pop("pred"), out.pop("saliency")
            info = out
            np.savez_compressed(stem.with_suffix(".npz"), pred=pred, saliency=sal)
            stem.with_suffix(".json").write_text(json.dumps(info))
            self.trained_now.add(key)
        rec = Record(info, pred, sal, self)
        self.records[key] = rec
        return rec


# ---------------------------------------------------------------------------
# 1. Live bandit
# ---------------------------------------------------------------------------
def run_live(ctx: Context, progress: Callable) -> dict:
    cfg = ctx.cfg
    K = len(cfg.models)
    bandit = make_bandit(cfg.algorithm, K, cfg.budget, cfg.seed)
    rng = np.random.default_rng(cfg.seed + 1)
    pulls = np.zeros(K, int)
    log, train_secs = [], np.zeros(K)
    for i in range(cfg.budget):
        if ctx.stop():
            break
        k = bandit.select()
        m = cfg.models[k]
        seed = int(pulls[k] % cfg.max_seeds)
        fresh = (m, seed) not in ctx.records
        rec = ctx.get(m, seed)
        if fresh:
            train_secs[k] += rec.info["seconds"]
        start = int(rng.integers(rec.n_windows))
        r, parts = rec.reward(start)
        bandit.update(k, r)
        pulls[k] += 1
        log.append({"pull": i + 1, "arm": m, "seed": seed, "window_start": start, "reward": _r(r),
                    "new_training": fresh, "recommend": cfg.models[bandit.recommend()],
                    **{k2: _r(v) for k2, v in parts.items()}})
        progress(0.05 + 0.35 * (i + 1) / cfg.budget,
                 f"Bandit pull {i + 1}/{cfg.budget}: {MODEL_INFO[m]['label']} seed {seed} -> reward {r:.3f}")
    means = bandit.means
    return {
        "algorithm": cfg.algorithm, "budget": cfg.budget, "winner": cfg.models[bandit.recommend()],
        "pulls": {m: int(pulls[i]) for i, m in enumerate(cfg.models)},
        "mean_reward": {m: _r(means[i]) if np.isfinite(means[i]) else None for i, m in enumerate(cfg.models)},
        "trainings": {m: int(min(pulls[i], cfg.max_seeds)) for i, m in enumerate(cfg.models)},
        "train_seconds": {m: _r(train_secs[i], 2) for i, m in enumerate(cfg.models)},
        "log": log,
    }


# ---------------------------------------------------------------------------
# 2 + 3. Ground truth and replay study
# ---------------------------------------------------------------------------
def reward_table(ctx: Context) -> np.ndarray:
    """R[arm, seed, window] for every model, seed and validation window."""
    cfg = ctx.cfg
    recs = [[ctx.get(m, s) for s in range(cfg.max_seeds)] for m in cfg.models]
    W = recs[0][0].n_windows
    return np.array([[[r.reward(w)[0] for w in range(W)] for r in row] for row in recs])


def replay_study(R: np.ndarray, cfg: BanditSettings) -> dict:
    K, S, W = R.shape
    mu = R.reshape(K, -1).mean(1)
    best = int(np.argmax(mu))
    B = cfg.budget
    out = {}
    for name in cfg.compare_algorithms:
        correct = np.zeros(B)
        regret = np.zeros(B)
        best_share = np.zeros(B)
        final_pulls = np.zeros(K)
        for run in range(cfg.replay_runs):
            rng = np.random.default_rng(10_000 + run)
            bd = make_bandit(name, K, B, 20_000 + run)
            on_best = 0
            for t in range(B):
                k = bd.select()
                bd.update(k, float(R[k, rng.integers(S), rng.integers(W)]))
                on_best += k == best
                rec = bd.recommend()
                correct[t] += rec == best
                regret[t] += mu[best] - mu[rec]
                best_share[t] += on_best / (t + 1)
            final_pulls += bd.n
        n = cfg.replay_runs
        pc = correct / n
        hit90 = np.flatnonzero(pc >= 0.9)
        out[name] = {"p_correct": [_r(x, 4) for x in pc], "simple_regret": [_r(x, 5) for x in regret / n],
                     "share_on_best": [_r(x, 4) for x in best_share / n],
                     "final_p_correct": _r(pc[-1], 4), "pulls_to_90pct": int(hit90[0] + 1) if len(hit90) else None,
                     "avg_pulls": {m: _r(final_pulls[i] / n, 2) for i, m in enumerate(cfg.models)}}
    return {"true_mean_reward": {m: _r(mu[i]) for i, m in enumerate(cfg.models)},
            "true_std_reward": {m: _r(R[i].std()) for i, m in enumerate(cfg.models)},
            "true_best": cfg.models[best],
            "gap_to_second": _r(np.sort(mu)[-1] - np.sort(mu)[-2]) if K > 1 else None,
            "algorithms": out}


# ---------------------------------------------------------------------------
# 4. Model comparison
# ---------------------------------------------------------------------------
def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> dict:
    """DM test on absolute-error loss differentials (positive stat -> model 1 has larger errors)."""
    d = np.abs(e1) - np.abs(e2)
    n = len(d)
    gamma0 = d.var()
    lrv = gamma0 + 2 * sum((1 - k / h) * np.cov(d[k:], d[:-k])[0, 1] for k in range(1, h)) if h > 1 else gamma0
    dm = d.mean() / math.sqrt(lrv / n + 1e-18)
    return {"stat": _r(dm, 3), "p_value": _r(2 * (1 - stats.norm.cdf(abs(dm))), 4)}


def _curve_mean(curves: list[list[float]]) -> list[float]:
    L = max(len(c) for c in curves)
    arr = np.full((len(curves), L), np.nan)
    for i, c in enumerate(curves):
        arr[i, :len(c)] = c
    return [_r(x) for x in np.nanmean(arr, 0)]


def _thin_history(h: dict) -> dict:
    return {"dates": h["dates"], "value": [round(v, 5) for v in h["value"]]}


def compare_segment(ctx: Context, segment: str, market_lr: np.ndarray, include_curves: bool = True) -> dict:
    cfg, b, data = ctx.cfg, ctx.b, ctx.data
    a, bnd = b.segment(segment)
    ts = data.decisions(a, bnd)
    rows, ens_err, ens_h = {}, {}, {}
    for m in cfg.models:
        recs = [ctx.get(m, s) for s in range(cfg.max_seeds)]
        per_seed = []
        for r in recs:
            h = r.h_val if segment == "val" else run_forecast_portfolio(b, ctx.cost, segment, r.pred, cfg.tilt,
                                                                        cfg.band)
            pm = compute_metrics(h, b.ppy, b.rf_per_bar, market_lr)
            pm.update(prediction_metrics(data, r.pred, ts))
            per_seed.append(pm)
        ens_pred = np.mean([r.pred for r in recs], 0)
        h_ens = run_forecast_portfolio(b, ctx.cost, segment, ens_pred, cfg.tilt, cfg.band)
        em = compute_metrics(h_ens, b.ppy, b.rf_per_bar, market_lr)
        em.update(prediction_metrics(data, ens_pred, ts))
        ens_err[m] = (ens_pred[ts] - data.y[ts]).mean(1)
        ens_h[m] = h_ens
        keys = ["sharpe", "cagr", "vol", "max_drawdown", "turnover_ann", "total_cost", "alpha_ann",
                "dir_acc", "mae", "rmse", "ic"]
        row = {"label": MODEL_INFO[m]["label"], "family": MODEL_INFO[m]["family"], "source": MODEL_INFO[m]["source"],
               "seeds": summarize_seeds(per_seed, keys),
               "ensemble": {k: _r(em.get(k)) for k in keys},
               "positive_sharpe_share": _r(np.mean([(p.get("sharpe") or 0) > 0 for p in per_seed]), 3)}
        if include_curves:
            row.update({
                "n_params": recs[0].info["n_params"],
                "train_seconds_median": _r(np.median([r.info["seconds"] for r in recs]), 2),
                "epochs_median": _r(np.median([r.info["epochs"] for r in recs]), 1),
                "best_epoch_median": _r(np.median([r.info["best_epoch"] for r in recs]), 1),
                "train_loss_curve": _curve_mean([r.info["train_loss"] for r in recs]),
                "val_loss_curve": _curve_mean([r.info["val_loss"] for r in recs]),
                "lag_saliency": [_r(x, 4) for x in np.mean([r.saliency for r in recs], 0)],
            })
        rows[m] = row
    # Diebold-Mariano: best ensemble (by median seed Sharpe) vs each of the others
    order = sorted(cfg.models, key=lambda m: -(rows[m]["seeds"].get("sharpe", {}).get("median") or -9))
    top = order[0]
    dm = {m: diebold_mariano(ens_err[top], ens_err[m]) for m in cfg.models if m != top}
    specs = [s for s in strategy_specs(b) if s["id"] in BASELINE_IDS]
    base = run_specs(b, ctx.cost, segment, specs, market_lr)
    baselines = {k: {"name": v["name"], **{x: _r(v["metrics"].get(x)) for x in
                                           ("sharpe", "cagr", "vol", "max_drawdown", "turnover_ann", "alpha_ann")}}
                 for k, v in base.items()}
    curves = {m: _thin_history(h) for m, h in ens_h.items()}
    curves.update({k: _thin_history(v["history"]) for k, v in base.items() if k in ("market", "ew_quarterly")})
    return {"segment": segment, "bars": len(ts), "models": rows, "ranking_by_sharpe": order,
            "diebold_mariano_vs_top": {"top": top, "tests": dm}, "baselines": baselines, "equity_curves": curves,
            "_ens_h": ens_h, "_base": base}


def regime_table(ctx: Context, comp: dict) -> list[dict]:
    from ..analysis import _regime_masks, _slice_stats
    b = ctx.b
    dates = next(iter(comp["_ens_h"].values()))["dates"]
    rows = []
    for name, kind, mask in _regime_masks(b, dates, None, events=False):
        row = {"regime": name, "bars": int(mask.sum()), "sharpe": {}}
        for m, h in comp["_ens_h"].items():
            st = _slice_stats(np.asarray(h["log_ret"]), mask, b.ppy, b.rf_per_bar)
            row["sharpe"][MODEL_INFO[m]["label"]] = _r(st.get("sharpe"), 3)
        for k in ("market", "ew_quarterly"):
            st = _slice_stats(np.asarray(comp["_base"][k]["history"]["log_ret"]), mask, b.ppy, b.rf_per_bar)
            row["sharpe"][comp["_base"][k]["name"]] = _r(st.get("sharpe"), 3)
        rows.append(row)
    return rows


def _public(comp: dict) -> dict:
    return {k: v for k, v in comp.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Full study
# ---------------------------------------------------------------------------
def run_study(b: Bundle, cost: CostSettings, cfg: BanditSettings, dataset_meta: dict,
              progress: Callable[[float, str], None] = lambda *_: None, stop: Callable[[], bool] = lambda: False,
              log: Callable[[str], None] = print) -> dict:
    t0 = time.perf_counter()
    unknown = [m for m in cfg.models if m not in MODEL_INFO]
    if unknown:
        raise ValueError(f"Unknown models {unknown}")
    ctx = Context(b, cost, cfg, stop)
    progress(0.02, "Building sequences")
    log(f"Sequences: {int(ctx.data.valid.sum())} decisions x {ctx.data.n_assets} assets, "
        f"window {ctx.data.lookback}, {ctx.data.n_features} features")

    live = run_live(ctx, progress)
    log(f"Live {cfg.algorithm}: winner {live['winner']}, pulls {live['pulls']}")
    live_trainings = len(ctx.trained_now)

    total = len(cfg.models) * cfg.max_seeds
    for i, (m, s) in enumerate([(m, s) for m in cfg.models for s in range(cfg.max_seeds)]):
        if stop():
            raise RuntimeError("Cancelled")
        ctx.get(m, s)
        progress(0.40 + 0.35 * (i + 1) / total, f"Ground-truth pool: {MODEL_INFO[m]['label']} seed {s}")
    R = reward_table(ctx)
    progress(0.78, "Replaying bandit algorithms")
    replay = replay_study(R, cfg)
    log(f"True best arm: {replay['true_best']}")

    # compute accounting: seconds of training the live bandit triggered vs uniform with the same budget
    mean_secs = {m: float(np.mean([ctx.get(m, s).info["seconds"] for s in range(cfg.max_seeds)])) for m in cfg.models}
    per_arm_uniform = min(cfg.max_seeds, math.ceil(cfg.budget / len(cfg.models)))
    live["compute"] = {
        "bandit_trainings": int(sum(live["trainings"].values())),
        "uniform_trainings": per_arm_uniform * len(cfg.models),
        "bandit_train_seconds_est": _r(sum(live["trainings"][m] * mean_secs[m] for m in cfg.models), 1),
        "uniform_train_seconds_est": _r(per_arm_uniform * sum(mean_secs.values()), 1),
        "exhaustive_trainings": total,
        "trained_in_this_run": live_trainings,
    }
    live["correct"] = live["winner"] == replay["true_best"]

    progress(0.85, "Comparing models on validation")
    mkt_val = market_log_ret(b, cost, "val")
    val = compare_segment(ctx, "val", mkt_val)
    result = {
        "id": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "created": datetime.now().isoformat(timespec="seconds"),
        "dataset": {k: dataset_meta.get(k) for k in ("id", "source", "start", "end", "bars", "sha", "tickers",
                                                     "train_end", "val_end")},
        "settings": cfg.model_dump(), "models": {m: MODEL_INFO[m] for m in cfg.models},
        "live": live, "replay": replay, "validation": _public(val),
        "validation_regimes": regime_table(ctx, val),
        "test": {"locked": True},
    }
    if cfg.evaluate_test:
        progress(0.93, "Evaluating on the locked test block (one-shot)")
        mkt_test = market_log_ret(b, cost, "test")
        test = compare_segment(ctx, "test", mkt_test, include_curves=False)
        w = live["winner"]
        ew = test["baselines"]["ew_quarterly"]["sharpe"]
        ws = test["models"][w]["seeds"].get("sharpe", {}).get("median")
        result["test"] = {"locked": False, "unlocked_at": datetime.now().isoformat(timespec="seconds"),
                          **_public(test), "regimes": regime_table(ctx, test),
                          "winner": w, "winner_sharpe_median": ws, "ew_quarterly_sharpe": ew,
                          "winner_beats_ew_quarterly": bool(ws is not None and ew is not None and ws > ew),
                          "winner_positive_sharpe": bool(ws is not None and ws > 0)}
    result["elapsed_seconds"] = _r(time.perf_counter() - t0, 1)
    save_result(result)
    progress(1.0, f"Done: bandit picked {MODEL_INFO[live['winner']]['label']}")
    return result


def save_result(result: dict) -> Path:
    p = RESULTS / f"{result['id']}.json"
    txt = json.dumps(result, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    p.write_text(txt)
    (RESULTS / "latest.json").write_text(txt)
    return p


def list_results() -> list[dict]:
    out = []
    for p in sorted(RESULTS.glob("2*.json"), reverse=True):
        r = json.loads(p.read_text())
        out.append({"id": r["id"], "created": r["created"], "winner": r["live"]["winner"],
                    "algorithm": r["live"]["algorithm"], "source": r["dataset"].get("source"),
                    "test_unlocked": not r["test"].get("locked", True)})
    return out


def load_result(result_id: str = "latest") -> Optional[dict]:
    p = RESULTS / f"{result_id}.json"
    return json.loads(p.read_text()) if p.exists() else None
