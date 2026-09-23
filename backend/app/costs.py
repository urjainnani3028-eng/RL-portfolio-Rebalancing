"""Transaction cost models. Costs are expressed as a fraction of portfolio value and are
charged on the traded notional of risky assets (moving in/out of cash is the other leg)."""
from __future__ import annotations

import numpy as np

from .config import CostSettings


class CostModel:
    def __init__(self, cfg: CostSettings, equity_flags: list[bool]):
        """equity_flags: one bool per *risky* asset (cash excluded)."""
        self.cfg = cfg
        n = len(equity_flags)
        slip = cfg.slippage_bps / 1e4
        if cfg.model == "flat":
            r = cfg.flat_bps / 1e4 + slip
            self.buy = np.full(n, r)
            self.sell = np.full(n, r)
        else:
            p = 1 / 100.0
            fees = (cfg.brokerage_pct + cfg.exchange_txn_pct + cfg.sebi_fee_pct) * p
            gst = fees * cfg.gst_pct / 100.0
            base = fees + gst + slip
            self.buy = np.full(n, base + cfg.stamp_buy_pct * p)
            stt = np.array([cfg.stt_sell_etf_pct * p if e else 0.0 for e in equity_flags])
            self.sell = base + stt

    def cost(self, w_old: np.ndarray, w_new: np.ndarray) -> float:
        """Fraction of portfolio value paid to move from w_old to w_new (last entry = cash)."""
        d = w_new[:-1] - w_old[:-1]
        return float(np.dot(self.buy, np.clip(d, 0, None)) + np.dot(self.sell, np.clip(-d, 0, None)))

    def breakdown(self) -> dict:
        """Round-trip cost in bps per asset, for the UI."""
        return {
            "buy_bps": (self.buy * 1e4).round(3).tolist(),
            "sell_bps": (self.sell * 1e4).round(3).tolist(),
            "round_trip_bps": ((self.buy + self.sell) * 1e4).round(3).tolist(),
        }


def flat_cost_settings(bps: float) -> CostSettings:
    return CostSettings(model="flat", flat_bps=bps, slippage_bps=0.0)
