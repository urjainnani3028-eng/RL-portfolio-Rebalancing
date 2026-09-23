"""Stable-Baselines3 agents, custom feature extractors (MLP / EIIE / cross-asset attention),
training with validation-based checkpoint selection, and deterministic evaluation."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .config import AgentSettings, CostSettings
from .metrics import compute_metrics
from .sim import Bundle, PortfolioEnv, rollout_env

torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

ALGOS = {"PPO": PPO, "A2C": A2C, "SAC": SAC, "TD3": TD3, "DQN": DQN}


# ---------------------------------------------------------------------------
# Feature extractors. Observation layout = [N assets x per_asset features | global]
# ---------------------------------------------------------------------------
class _Split(nn.Module):
    def __init__(self, n, fa, g):
        super().__init__()
        self.n, self.fa, self.g = n, fa, g

    def forward(self, x):
        per = x[:, : self.n * self.fa].reshape(-1, self.n, self.fa)
        glob = x[:, self.n * self.fa:]
        return per, glob


class EIIEExtractor(BaseFeaturesExtractor):
    """Ensemble of Identical Independent Evaluators (Jiang, Xu & Liang, 2017): one small
    network with shared weights scores every asset from its own history."""

    def __init__(self, observation_space, n_assets: int, per_asset: int, global_dim: int, emb: int = 16):
        super().__init__(observation_space, features_dim=n_assets * emb + global_dim)
        self.split = _Split(n_assets, per_asset, global_dim)
        self.evaluator = nn.Sequential(nn.Linear(per_asset, 64), nn.ReLU(), nn.Linear(64, emb), nn.ReLU())

    def forward(self, obs):
        per, glob = self.split(obs)
        e = self.evaluator(per)                       # (B, N, emb) - same weights for every asset
        return torch.cat([e.flatten(1), glob], dim=1)


class AttentionExtractor(BaseFeaturesExtractor):
    """Per-asset embeddings + one transformer layer across assets (cross-asset attention)."""

    def __init__(self, observation_space, n_assets: int, per_asset: int, global_dim: int, d: int = 32):
        super().__init__(observation_space, features_dim=n_assets * d + global_dim)
        self.split = _Split(n_assets, per_asset, global_dim)
        self.embed = nn.Linear(per_asset, d)
        self.asset_id = nn.Parameter(torch.randn(1, n_assets, d) * 0.02)
        self.attn = nn.TransformerEncoderLayer(d_model=d, nhead=4, dim_feedforward=64, batch_first=True,
                                               dropout=0.0)

    def forward(self, obs):
        per, glob = self.split(obs)
        h = self.attn(self.embed(per) + self.asset_id)
        return torch.cat([h.flatten(1), glob], dim=1)


def build_model(env, obs_spec: dict, agent: AgentSettings, seed: int):
    algo = ALGOS[agent.algo]
    pk: dict = {}
    if agent.arch == "eiie":
        pk.update(features_extractor_class=EIIEExtractor, features_extractor_kwargs=obs_spec)
    elif agent.arch == "attention":
        pk.update(features_extractor_class=AttentionExtractor, features_extractor_kwargs=obs_spec)
    kw: dict = dict(learning_rate=agent.learning_rate, seed=seed, verbose=0, device="cpu")
    if agent.algo == "PPO":
        pk["net_arch"] = dict(pi=[64, 64], vf=[64, 64])
        kw.update(n_steps=1024, batch_size=128, n_epochs=10, gamma=0.99, gae_lambda=0.95, ent_coef=0.0,
                  clip_range=0.2)
    elif agent.algo == "A2C":
        pk["net_arch"] = dict(pi=[64, 64], vf=[64, 64])
        kw.update(n_steps=16, gamma=0.99)
    elif agent.algo in ("SAC", "TD3"):
        pk["net_arch"] = [128, 128]
        kw.update(buffer_size=100_000, learning_starts=1000, batch_size=128, gamma=0.99, train_freq=1)
        if agent.algo == "TD3":
            n = env.action_space.shape[0]
            kw["action_noise"] = NormalActionNoise(np.zeros(n), 0.2 * np.ones(n))
    elif agent.algo == "DQN":
        pk["net_arch"] = [128, 128]
        kw.update(buffer_size=50_000, learning_starts=1000, batch_size=64, gamma=0.99, train_freq=4,
                  target_update_interval=1000, exploration_fraction=0.3, exploration_final_eps=0.05)
    return algo("MlpPolicy", env, policy_kwargs=pk, **kw)


def load_model(path: str | Path, agent: AgentSettings):
    return ALGOS[agent.algo].load(str(path), device="cpu")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, bundle: Bundle, agent: AgentSettings, cost: CostSettings, segment,
             market_lr: Optional[np.ndarray] = None, leak: bool = False) -> tuple[dict, dict]:
    env = PortfolioEnv(bundle, agent, cost, segment=segment, training=False, leak=leak)
    h = rollout_env(env, lambda o: model.predict(o, deterministic=True)[0])
    return h, compute_metrics(h, bundle.ppy, bundle.rf_per_bar, market_lr)


# ---------------------------------------------------------------------------
# Training with progress reporting + validation checkpointing
# ---------------------------------------------------------------------------
class ProgressCallback(BaseCallback):
    def __init__(self, total: int, eval_every: int, on_progress: Callable, on_eval: Callable,
                 should_stop: Callable[[], bool]):
        super().__init__()
        self.total, self.eval_every = total, max(1000, eval_every)
        self.on_progress, self.on_eval, self.should_stop = on_progress, on_eval, should_stop
        self._last_eval = 0
        self._last_report = 0

    def _on_step(self) -> bool:
        n = self.num_timesteps
        if n - self._last_report >= max(256, self.total // 200):
            self._last_report = n
            buf = [e["r"] for e in self.model.ep_info_buffer] if self.model.ep_info_buffer else []
            self.on_progress(n, float(np.mean(buf)) if buf else None)
        if n - self._last_eval >= self.eval_every:
            self._last_eval = n
            self.on_eval(n, self.model)
        return not self.should_stop()


def train_one_seed(bundle: Bundle, agent: AgentSettings, cost: CostSettings, seed: int, out_path: Path,
                   on_progress: Callable, on_eval_point: Callable, should_stop: Callable[[], bool],
                   val_segment="val", leak: bool = False) -> dict:
    base = PortfolioEnv(bundle, agent, cost, segment="train", training=True, leak=leak, seed=seed)
    env = Monitor(base)
    env.reset(seed=seed)
    model = build_model(env, base.obs_spec, agent, seed)
    best = {"sharpe": -np.inf, "step": 0}

    def do_eval(step, m):
        if val_segment is None:
            return
        _, met = evaluate(m, bundle, agent, cost, val_segment, leak=leak)
        sh = met.get("sharpe") if met.get("sharpe") is not None else -np.inf
        on_eval_point(step, met)
        if sh > best["sharpe"]:
            best.update(sharpe=sh, step=step)
            m.save(str(out_path))

    cb = ProgressCallback(agent.timesteps, agent.eval_every, on_progress, do_eval, should_stop)
    model.learn(total_timesteps=agent.timesteps, callback=cb, progress_bar=False)
    if should_stop():
        return {"cancelled": True}
    do_eval(agent.timesteps, model)
    if val_segment is None or not out_path.exists():
        model.save(str(out_path))
        best["step"] = agent.timesteps
    return {"best_step": best["step"], "best_val_sharpe": None if best["sharpe"] == -np.inf else best["sharpe"]}
