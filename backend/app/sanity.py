"""The five sanity tests that must pass before any training result is trusted."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

from .config import AgentSettings, CostSettings, RewardSettings, Settings
from .costs import flat_cost_settings
from .data import Dataset
from .metrics import compute_metrics
from .sim import PortfolioEnv, make_bundle, rollout_env, run_policy


def _res(tid, name, passed, detail, **numbers):
    return {"id": tid, "name": name, "passed": bool(passed), "detail": detail,
            "numbers": {k: (round(float(v), 8) if v is not None else None) for k, v in numbers.items()}}


def test_buy_and_hold(ds: Dataset, st: Settings) -> dict:
    b = make_bundle(ds, st.portfolio)
    zero = flat_cost_settings(0.0)
    a, e = b.segment("train")
    w0 = np.append(np.full(b.n, 1 / b.n), 0.0)
    h = run_policy(b, zero, "train", lambda t, w, k: w0 if k == 0 else None, enforce_cap=False)
    by_hand = float(w0 @ np.prod(1 + b.R[a + 1:e + 1], axis=0))
    err = abs(h["value"][-1] - by_hand) / by_hand
    return _res("bh", "Zero-cost buy-and-hold matches hand calculation", err < 1e-9,
                f"Simulator final value {h['value'][-1]:.6f} vs hand-computed {by_hand:.6f} "
                f"(relative error {err:.2e}).", sim=h["value"][-1], hand=by_hand, rel_error=err)


def test_forced_rotation(ds: Dataset, st: Settings, bps: float = 10.0) -> dict:
    b = make_bundle(ds, st.portfolio)
    cost = flat_cost_settings(bps)
    c = bps / 1e4
    a, e = b.segment("val")
    e0, e1 = np.eye(b.n + 1)[0], np.eye(b.n + 1)[1]
    h = run_policy(b, cost, "val", lambda t, w, k: e0 if k % 2 == 0 else e1, enforce_cap=False)
    v, paid = 1.0, 0.0
    for k, t in enumerate(range(a, e)):
        traded = 1.0 if k == 0 else 2.0            # cash->A first, then A<->B (sell 1 + buy 1)
        paid += c * traded
        asset = 0 if k % 2 == 0 else 1
        v *= (1 - c * traded) * (1 + b.R[t + 1, asset])
    err = abs(h["value"][-1] - v) / v
    sim_paid = float(np.sum(h["cost"]))
    ok = err < 1e-9 and abs(sim_paid - paid) < 1e-9
    return _res("rotation", f"Forced rotation pays exactly turnover x {bps:g} bps", ok,
                f"{len(h['cost'])} forced rotations. Cost paid {sim_paid:.4%} vs expected {paid:.4%}; "
                f"final value error {err:.2e}. Confirms costs are charged before the market move.",
                sim_cost=sim_paid, expected_cost=paid, rel_error=err)


def test_one_over_n(ds: Dataset, st: Settings, bps: float = 10.0) -> dict:
    """Env with a uniform action (softmax of zeros) must reproduce an independently coded 1/n."""
    b = make_bundle(ds, st.portfolio)
    cost = flat_cost_settings(bps)
    agent = AgentSettings(mode="direct", algo="PPO", reward=RewardSettings(risk="none"))
    env = PortfolioEnv(b, agent, cost, segment="val", training=False)
    h = rollout_env(env, lambda o: np.zeros(b.n + 1, dtype=np.float32))
    a, e = b.segment("val")
    n1 = b.n + 1
    tgt = np.full(n1, 1 / n1)
    w = np.eye(n1)[-1]
    v = 1.0
    c = bps / 1e4
    for t in range(a, e):                          # independent vectorised re-implementation
        fee = c * np.abs(tgt[:-1] - w[:-1]).sum()
        gross = tgt * (1 + b.R[t + 1])
        v *= (1 - fee) * gross.sum()
        w = gross / gross.sum()
    err = abs(h["value"][-1] - v) / v
    return _res("one_over_n", "1/n action reproduces the equal-weight portfolio", err < 1e-9,
                f"Env final value {h['value'][-1]:.6f} vs independent 1/n loop {v:.6f} (relative error {err:.2e}).",
                env=h["value"][-1], independent=v, rel_error=err)


def _timing_edge(model, agent, b, cost, leak=False) -> tuple[float, float]:
    """Agent Sharpe vs a constant mix of its own average weights, over val+test (out of sample).
    The gap isolates timing skill from a static tilt."""
    from .agents import evaluate

    span = (b.segments["val"][0], b.segments["test"][1])
    h_rl, met = evaluate(model, b, agent, cost, span, leak=leak)
    avg_w = np.asarray(h_rl["weights"]).mean(0)
    h = run_policy(b, cost, span, lambda t, w, k: avg_w, enforce_cap=False)
    return met["sharpe"], compute_metrics(h, b.ppy, b.rf_per_bar)["sharpe"]


def _quick_agent(ds: Dataset, st: Settings, shuffle_seed=None, leak=False, steps=20000,
                 progress: Callable = lambda *_: None, stop=lambda: False, n_eval_shuffles: int = 0):
    from .agents import load_model, train_one_seed

    b = make_bundle(ds, st.portfolio, shuffle_seed=shuffle_seed)
    cost = flat_cost_settings(5.0)
    agent = AgentSettings(mode="direct", algo="PPO", arch="mlp", timesteps=steps, eval_every=steps,
                          reward=RewardSettings(risk="none"), learning_rate=3e-4)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.zip"
        train_one_seed(b, agent, cost, 0, p, progress, lambda *_: None, stop, val_segment=None, leak=leak)
        model = load_model(p, agent)
    if n_eval_shuffles:
        # average over several fresh permutations of the out-of-sample period to beat estimation noise
        res = [_timing_edge(model, agent, make_bundle(ds, st.portfolio, shuffle_seed=1000 + k), cost)
               for k in range(n_eval_shuffles)]
        return float(np.mean([r[0] for r in res])), float(np.mean([r[1] for r in res]))
    return _timing_edge(model, agent, b, cost, leak=leak)


def test_shuffled(ds: Dataset, st: Settings, progress=lambda *_: None, stop=lambda: False) -> dict:
    rl, stat = _quick_agent(ds, st, shuffle_seed=123, progress=progress, stop=stop, n_eval_shuffles=8)
    edge = rl - stat
    return _res("shuffled", "Shuffled returns are NOT profitable", edge < 0.25,
                f"PPO trained on time-shuffled returns, evaluated out-of-sample on 8 fresh shuffles: mean Sharpe "
                f"{rl:.2f} vs {stat:.2f} for a constant mix of its own average weights (timing edge {edge:+.2f}). "
                f"No temporal structure -> no timing edge (pass if edge < 0.25).",
                agent_sharpe=rl, static_sharpe=stat, edge=edge)


def test_lookahead(ds: Dataset, st: Settings, progress=lambda *_: None, stop=lambda: False) -> dict:
    rl, stat = _quick_agent(ds, st, leak=True, progress=progress, stop=stop)
    edge = rl - stat
    return _res("lookahead", "Injected look-ahead improves results sharply", edge > 1.0,
                f"PPO given next-bar returns in its state: out-of-sample Sharpe {rl:.2f} vs {stat:.2f} for a constant "
                f"mix of its average weights (timing edge {edge:+.2f}). Proves the pipeline learns signal when it "
                f"exists (pass if edge > 1.0).", agent_sharpe=rl, static_sharpe=stat, edge=edge)


FAST = [test_buy_and_hold, test_forced_rotation, test_one_over_n]
SLOW = [test_shuffled, test_lookahead]
