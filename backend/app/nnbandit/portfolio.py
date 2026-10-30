"""Turn forecasts into a rebalancing policy, and score forecasts and portfolios.

Policy (identical for every model, so only forecast quality differs):
    w_t = softmax(tilt * pred_t) over the risky ETFs, capped at max_weight per ETF
    trade only if some weight is more than `band` away from the drifted weight
pred_t is in volatility units (standardised log return), so a model takes a big bet only when it
forecasts a big move; a model with flat (no-information) forecasts holds equal weight. We do NOT
re-standardise across assets: that would blow tiny noise up into full-size bets.
It runs through the shared simulator, so costs and timing match the RL agent and the baselines.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ..config import CostSettings
from ..optimizers import apply_cap
from ..sim import Bundle, run_policy
from .features import SeqData


def target_weights(pred_row: np.ndarray, tilt: float, cap: float) -> np.ndarray:
    z = np.clip(tilt * pred_row, -6.0, 6.0)
    e = np.exp(z - z.max())
    return apply_cap(np.append(e / e.sum(), 0.0), cap)


def run_forecast_portfolio(b: Bundle, cost: CostSettings, segment, pred: np.ndarray, tilt: float = 5.0,
                           band: float = 0.02) -> dict:
    cap = b.pcfg.max_weight

    def policy(t, w, k):
        if np.isnan(pred[t]).any():
            return None
        tgt = target_weights(pred[t], tilt, cap)
        return tgt if (k == 0 or np.abs(tgt - w).max() > band) else None

    return run_policy(b, cost, segment, policy)


def prediction_metrics(data: SeqData, pred: np.ndarray, ts: np.ndarray) -> dict:
    """Directional accuracy, MAE/RMSE (standardised units) and mean cross-sectional rank IC."""
    p, y, r = pred[ts], data.y[ts], data.ret[ts]
    raw_pred = p * data.sd + data.mu               # back to log-return units for the sign
    hit = (np.sign(raw_pred) == np.sign(r)).mean()
    ic = [stats.spearmanr(p[i], y[i])[0] for i in range(len(ts)) if p[i].std() > 1e-9]
    return {"dir_acc": float(hit), "mae": float(np.abs(p - y).mean()), "rmse": float(np.sqrt(((p - y) ** 2).mean())),
            "ic": float(np.nanmean(ic)) if ic else 0.0}


def sharpe_and_mdd(log_ret: np.ndarray, ppy: int, rf: float) -> tuple[float, float]:
    r = np.expm1(log_ret) - rf
    sharpe = float(r.mean() / (r.std(ddof=1) + 1e-12) * np.sqrt(ppy))
    curve = np.exp(np.cumsum(log_ret))
    mdd = float((1 - curve / np.maximum.accumulate(np.concatenate([[1.0], curve]))[1:]).max())
    return sharpe, mdd


def window_reward(log_ret: np.ndarray, hits: np.ndarray, ppy: int, rf: float, weights: dict,
                  ref: tuple[float, float] | None = None, ref_acc: float | None = None) -> tuple[float, dict]:
    """Composite bandit reward in [0, 1] for one evaluation window, from the plan's three metrics:
    Sharpe (risk-adjusted return), directional accuracy and maximum drawdown.

    With `ref` = (Sharpe, max drawdown) of a reference strategy on the *same* window, Sharpe and
    drawdown are scored relative to it. Every arm then faces the same market conditions (common
    random numbers), which strips most of the window-to-window noise out of the reward.

    With `ref_acc` = hit rate of always predicting "up" on the same window, accuracy is scored
    relative to it: in a rising market, always saying "up" scores well without any skill."""
    sharpe, mdd = sharpe_and_mdd(log_ret, ppy, rf)
    acc = float(hits.mean())
    if ref is None:
        s_part, d_part = (sharpe + 2.0) / 4.0, 1 - mdd / 0.30         # Sharpe -2..+2, drawdown 0..30%
    else:
        s_part = 0.5 + (sharpe - ref[0]) / 2.0                      # Sharpe gap -1..+1 -> 0..1
        d_part = 0.5 + (ref[1] - mdd) / 0.20                        # drawdown gap +-10 pts -> 0..1
    a_part = (acc - 0.40) / 0.20 if ref_acc is None else 0.5 + (acc - ref_acc) / 0.20   # +-10 pts -> 0..1
    parts = {"sharpe": np.clip(s_part, 0, 1), "accuracy": np.clip(a_part, 0, 1),
             "drawdown": np.clip(d_part, 0, 1)}
    tot = sum(weights.values())
    reward = float(sum(weights[k] * parts[k] for k in weights) / tot)
    return reward, {"sharpe": sharpe, "dir_acc": acc, "max_drawdown": mdd}
