"""Portfolio simulator + Gymnasium environment.

Per-step order (non-negotiable):
    1. agent decides target weights using information up to t - exec_delay
    2. transaction cost is charged on turnover (drifted -> target)
    3. THEN the market moves: bar t+1 returns are applied to the target weights
    4. weights drift with prices; the drifted weights are carried to the next step
    5. reward = log net growth - risk penalty
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import CACHE_DIR, AgentSettings, CostSettings, PortfolioSettings, RewardSettings
from .costs import CostModel
from .data import Dataset, asset_info, split_indices
from .optimizers import apply_cap, target_path

HYBRID_LEVELS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])


# ---------------------------------------------------------------------------
# Bundle: everything derived from a dataset for one experiment
# ---------------------------------------------------------------------------
@dataclass
class Bundle:
    dates: np.ndarray
    R: np.ndarray                    # T x (N+1) simple returns, cash last
    assets: list[str]                # risky asset tickers
    equity_flags: list[bool]
    segments: dict
    ppy: int
    pcfg: PortfolioSettings
    rf_per_bar: float
    z: np.ndarray = None             # T x N standardised log returns (train-fit scaler)
    vol_z: np.ndarray = None         # T x N standardised rolling vol
    corr_z: np.ndarray = None        # T average pairwise correlation (standardised)
    warmup: int = 0
    key: str = ""
    _targets: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.R.shape[1] - 1

    def targets(self, method: str) -> np.ndarray:
        if method not in self._targets:
            p = self.pcfg
            path = CACHE_DIR / f"tgt_{self.key}_{method}_{p.opt_window}_{p.max_weight}_{p.exec_delay}.npy"
            if self.key and path.exists():
                self._targets[method] = np.load(path)
            else:
                self._targets[method] = target_path(self.R, method, p.opt_window, p.max_weight, delay=p.exec_delay)
                if self.key:
                    np.save(path, self._targets[method])
        return self._targets[method]

    def segment(self, name) -> tuple[int, int]:
        a, b = self.segments[name] if isinstance(name, str) else name
        return max(a, self.warmup), b


def make_bundle(ds: Dataset, pcfg: PortfolioSettings, shuffle_seed: Optional[int] = None) -> Bundle:
    R = ds.returns.values.astype(float)
    segs = split_indices(ds)
    if shuffle_seed is not None:
        # Destroy temporal structure inside each segment (sanity test: must NOT be profitable)
        rng = np.random.default_rng(shuffle_seed)
        R = R.copy()
        for a, b in segs.values():
            idx = np.arange(a + 1, b + 1)
            R[idx] = R[rng.permutation(idx)]
    info = asset_info(ds)
    b = Bundle(dates=np.array(ds.dates), R=R, assets=ds.assets, equity_flags=[i["equity"] for i in info[:-1]],
               segments=segs, ppy=ds.periods_per_year(), pcfg=pcfg,
               rf_per_bar=float(np.mean(R[1:, -1])),
               key=f"{ds.id}_{ds.meta.get('sha', '')}_{ds.meta.get('train_end')}_{ds.meta.get('val_end')}_"
                   f"{ds.meta.get('embargo_bars')}_s{shuffle_seed}")
    _compute_features(b)
    return b


def _compute_features(b: Bundle) -> None:
    p = b.pcfg
    N = b.n
    lr = np.log1p(b.R[:, :N])
    tr_a, tr_b = b.segments["train"]
    train_rows = slice(1, tr_b + 1)
    mu, sd = lr[train_rows].mean(0), lr[train_rows].std(0) + 1e-8      # fit on TRAIN only
    b.z = (lr - mu) / sd
    T = len(lr)
    w = p.risk_window
    vol = np.zeros((T, N))
    corr = np.zeros(T)
    for t in range(T):
        lo = max(1, t - w + 1)
        win = lr[lo:t + 1]
        if len(win) >= 3:
            vol[t] = win.std(0) * np.sqrt(b.ppy)
            if N > 1:
                c = np.corrcoef(win, rowvar=False)
                c = np.nan_to_num(c)
                corr[t] = (c.sum() - N) / (N * (N - 1))
    vm, vs = vol[train_rows].mean(0), vol[train_rows].std(0) + 1e-8
    cm, cs = corr[train_rows].mean(), corr[train_rows].std() + 1e-8
    b.vol_z = (vol - vm) / vs
    b.corr_z = (corr - cm) / cs
    b.warmup = max(p.lookback, p.risk_window, p.opt_window) + p.exec_delay + 1


# ---------------------------------------------------------------------------
# Core simulator (shared by RL env and every baseline -> identical accounting)
# ---------------------------------------------------------------------------
class PortfolioSim:
    def __init__(self, bundle: Bundle, cost: CostModel):
        self.b = bundle
        self.cost_model = cost

    def reset(self, t0: int, w0: Optional[np.ndarray] = None):
        n1 = self.b.n + 1
        self.t = t0
        self.w = np.eye(n1)[-1] if w0 is None else np.asarray(w0, float) / np.sum(w0)
        self.value = 1.0
        self.peak = 1.0
        self.dd = 0.0
        self.log_rets: list[float] = []
        self.bars_since_trade = 0
        return self

    def step(self, target: np.ndarray) -> dict:
        target = np.asarray(target, float)
        w_before = self.w
        turnover = 0.5 * float(np.abs(target - w_before).sum())
        cost = self.cost_model.cost(w_before, target) if turnover > 1e-12 else 0.0   # (2) cost first
        r = self.b.R[self.t + 1]                                                    # (3) then market moves
        growth = float(target @ (1 + r))
        net = (1 - cost) * growth
        self.value *= net
        self.w = target * (1 + r) / growth                                          # (4) drift
        self.peak = max(self.peak, self.value)
        prev_dd = self.dd
        self.dd = 1 - self.value / self.peak
        lr = float(np.log(net))
        self.log_rets.append(lr)
        self.bars_since_trade = 0 if turnover > 1e-6 else self.bars_since_trade + 1
        self.t += 1
        return {"cost": cost, "turnover": turnover, "growth": growth, "log_ret": lr, "net": net,
                "target": target, "w_before": w_before, "dd": self.dd, "dd_inc": max(0.0, self.dd - prev_dd)}


# ---------------------------------------------------------------------------
# Gymnasium environment
# ---------------------------------------------------------------------------
class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, bundle: Bundle, agent: AgentSettings, cost: CostSettings, segment: str = "train",
                 training: bool = True, leak: bool = False, seed: Optional[int] = None):
        super().__init__()
        self.b = bundle
        self.agent = agent
        self.rw: RewardSettings = agent.reward
        self.seg = bundle.segment(segment)
        self.training = training
        self.leak = leak
        self.sim = PortfolioSim(bundle, CostModel(cost, bundle.equity_flags))
        self.rng = np.random.default_rng(seed)
        self.hybrid = agent.mode == "hybrid"
        self.discrete = agent.algo == "DQN"
        if self.discrete and not self.hybrid:
            raise ValueError("DQN is only available in hybrid (timing-only) mode")
        self.targets = bundle.targets(agent.target) if self.hybrid else None
        N = bundle.n
        L = bundle.pcfg.lookback
        self.Fa = L + 1 + 1 + (2 if self.hybrid else 0) + (1 if leak else 0)
        self.G = 5 + (2 if self.hybrid else 0)
        self.observation_space = spaces.Box(-10, 10, shape=(N * self.Fa + self.G,), dtype=np.float32)
        if self.discrete:
            self.action_space = spaces.Discrete(len(HYBRID_LEVELS))
        elif self.hybrid:
            self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(-1, 1, shape=(N + 1,), dtype=np.float32)
        self.obs_spec = {"n_assets": N, "per_asset": self.Fa, "global_dim": self.G}

    # -- observation ---------------------------------------------------------
    def _obs(self) -> np.ndarray:
        b, t, s = self.b, self.sim.t, self.sim
        f = t - b.pcfg.exec_delay
        L = b.pcfg.lookback
        N = b.n
        cols = [b.z[f - L + 1:f + 1].T, b.vol_z[f][:, None], (s.w[:-1] * N)[:, None]]
        g = [b.corr_z[f], s.w[-1], s.dd * 10,
             (np.std(s.log_rets[-13:]) * np.sqrt(b.ppy) * 5) if len(s.log_rets) > 2 else 0.0,
             min(s.bars_since_trade, 26) / 13.0]
        if self.hybrid:
            tgt = self.targets[t]
            cols += [(tgt[:-1] * N)[:, None], ((tgt[:-1] - s.w[:-1]) * N)[:, None]]
            g += [tgt[-1], 0.5 * float(np.abs(tgt - s.w).sum()) * 5]
        if self.leak:  # deliberately leak next-bar returns (sanity test only!)
            cols.append(b.z[min(t + 1, len(b.z) - 1)][:, None])
        per_asset = np.concatenate(cols, axis=1).reshape(-1)
        obs = np.concatenate([per_asset, np.array(g, dtype=float)])
        return np.clip(np.nan_to_num(obs), -10, 10).astype(np.float32)

    # -- action -> target weights ---------------------------------------------
    def action_to_target(self, action) -> tuple[np.ndarray, float]:
        w = self.sim.w
        cap = self.b.pcfg.max_weight
        if self.hybrid:
            if self.discrete:
                a = float(HYBRID_LEVELS[int(action)])
            else:
                a = float(np.clip((np.asarray(action).reshape(-1)[0] + 1) / 2, 0, 1))
            tgt = w + a * (self.targets[self.sim.t] - w)
            tgt = apply_cap(tgt, cap) if a > 0 else w
            return tgt, a
        x = np.asarray(action, float).reshape(-1) * 3.0
        e = np.exp(x - x.max())
        tgt = apply_cap(e / e.sum(), cap)
        return tgt, float("nan")

    # -- gym API ---------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        w0_override = (options or {}).get("w0")
        a, bnd = self.seg
        n1 = self.b.n + 1
        if self.training:
            L = self.agent.episode_len
            t0 = int(self.rng.integers(a, max(a + 1, bnd - L)))
            self.end = min(bnd, t0 + L)
            k = self.rng.integers(4)
            if k == 0:
                w0 = np.eye(n1)[-1]
            elif k == 1:
                w0 = apply_cap(np.append(np.full(n1 - 1, 1 / (n1 - 1)), 0), self.b.pcfg.max_weight)
            elif k == 2 and self.hybrid:
                w0 = self.targets[t0]
            else:
                w0 = apply_cap(self.rng.dirichlet(np.ones(n1)), self.b.pcfg.max_weight)
        else:
            t0, self.end = a, bnd
            w0 = np.eye(n1)[-1] if w0_override is None else np.asarray(w0_override, float)
        self.sim.reset(t0, w0)
        return self._obs(), {}

    def step(self, action):
        tgt, a = self.action_to_target(action)
        band = self.agent.no_trade_band
        if band > 0 and 0.5 * np.abs(tgt - self.sim.w).sum() < band:
            tgt = self.sim.w.copy()
        info = self.sim.step(tgt)
        info["a"] = a
        rw = self.rw
        lr = info["log_ret"]
        pen = 0.0
        if rw.risk == "variance":
            pen = rw.risk_lambda * lr ** 2
        elif rw.risk == "downside":
            pen = rw.risk_lambda * min(lr, 0.0) ** 2
        elif rw.risk == "drawdown":
            pen = rw.risk_lambda * info["dd_inc"]
        if rw.concentration_kappa:
            risky = tgt[:-1]
            pen += rw.concentration_kappa * max(0.0, float((risky ** 2).sum()) - 1 / self.b.n)
        if rw.turnover_penalty:
            pen += rw.turnover_penalty * info["turnover"]
        reward = rw.scale * (lr - pen)
        truncated = self.sim.t >= self.end
        return self._obs(), float(reward), False, bool(truncated), info


# ---------------------------------------------------------------------------
# Deterministic rollouts -> history dict (used for RL eval and all baselines)
# ---------------------------------------------------------------------------
def empty_history():
    return {"dates": [], "value": [], "log_ret": [], "cost": [], "turnover": [], "weights": [], "a": []}


def record(h: dict, b: Bundle, sim: PortfolioSim, info: dict):
    h["dates"].append(str(np.datetime_as_string(b.dates[sim.t], unit="D")))
    h["value"].append(sim.value)
    h["log_ret"].append(info["log_ret"])
    h["cost"].append(info["cost"])
    h["turnover"].append(info["turnover"])
    h["weights"].append(info["target"].tolist())
    h["a"].append(info.get("a", float("nan")))


def rollout_env(env: PortfolioEnv, act: Callable, w0=None) -> dict:
    obs, _ = env.reset(options={"w0": w0} if w0 is not None else None)
    h = empty_history()
    h["start"] = str(np.datetime_as_string(env.b.dates[env.sim.t], unit="D"))
    done = False
    while not done:
        obs, _, term, trunc, info = env.step(act(obs))
        record(h, env.b, env.sim, info)
        done = term or trunc
    h["final_w"] = env.sim.w.tolist()
    return h


def run_policy(b: Bundle, cost: CostSettings, segment: str,
               policy: Callable[[int, np.ndarray, int], Optional[np.ndarray]], enforce_cap: bool = True) -> dict:
    """policy(t, current_drifted_w, step_no) -> target weights, or None to hold."""
    a, bnd = b.segment(segment)
    sim = PortfolioSim(b, CostModel(cost, b.equity_flags)).reset(a)
    h = empty_history()
    h["start"] = str(np.datetime_as_string(b.dates[a], unit="D"))
    k = 0
    while sim.t < bnd:
        tgt = policy(sim.t, sim.w.copy(), k)
        if tgt is None:
            tgt = sim.w.copy()
        else:
            tgt = np.clip(np.asarray(tgt, float), 0, None)
            tgt = apply_cap(tgt, b.pcfg.max_weight) if enforce_cap else tgt / tgt.sum()
        info = sim.step(tgt)
        record(h, b, sim, info)
        k += 1
    return h
