"""Central configuration: asset universe catalogue, default experiment settings, paths."""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
STORAGE = ROOT / "storage"
CACHE_DIR = STORAGE / "cache"
DATASETS_DIR = STORAGE / "datasets"
RUNS_DIR = STORAGE / "runs"
for _d in (CACHE_DIR, DATASETS_DIR, RUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Indian ETF universe (NSE tickers on Yahoo Finance use the ".NS" suffix).
# `listed` is approximate and only used for UI hints; the loader intersects on
# actual data availability. `stt_sell` flags equity-oriented ETFs which attract
# STT on sale of units; gold / debt ETFs do not.
# ---------------------------------------------------------------------------
CATALOGUE = [
    {"ticker": "NIFTYBEES.NS", "name": "Nippon India Nifty 50 BeES", "cls": "Equity - Large cap", "listed": "2002", "equity": True},
    {"ticker": "JUNIORBEES.NS", "name": "Nippon India Nifty Next 50 BeES", "cls": "Equity - Large/Mid", "listed": "2003", "equity": True},
    {"ticker": "BANKBEES.NS", "name": "Nippon India Nifty Bank BeES", "cls": "Equity - Banking", "listed": "2004", "equity": True},
    {"ticker": "PSUBNKBEES.NS", "name": "Nippon India Nifty PSU Bank BeES", "cls": "Equity - PSU banks", "listed": "2007", "equity": True},
    {"ticker": "INFRABEES.NS", "name": "Nippon India Nifty Infra BeES", "cls": "Equity - Infrastructure", "listed": "2010", "equity": True},
    {"ticker": "GOLDBEES.NS", "name": "Nippon India Gold BeES", "cls": "Gold", "listed": "2007", "equity": False},
    {"ticker": "MON100.NS", "name": "Motilal Oswal Nasdaq 100 ETF", "cls": "International equity", "listed": "2011", "equity": True},
    {"ticker": "CPSEETF.NS", "name": "CPSE ETF", "cls": "Equity - PSU", "listed": "2014", "equity": True},
    {"ticker": "LICNETFGSC.NS", "name": "LIC MF G-Sec Long Term ETF", "cls": "Bonds - Govt", "listed": "2014", "equity": False},
    {"ticker": "SETF10GILT.NS", "name": "SBI ETF 10 Year Gilt", "cls": "Bonds - Govt", "listed": "2016", "equity": False},
    {"ticker": "SILVERBEES.NS", "name": "Nippon India Silver ETF", "cls": "Silver", "listed": "2022", "equity": False},
    {"ticker": "ITBEES.NS", "name": "Nippon India Nifty IT ETF", "cls": "Equity - IT", "listed": "2020", "equity": True},
]
CATALOGUE_BY_TICKER = {c["ticker"]: c for c in CATALOGUE}

DEFAULT_TICKERS = [
    "NIFTYBEES.NS", "JUNIORBEES.NS", "BANKBEES.NS", "PSUBNKBEES.NS",
    "INFRABEES.NS", "GOLDBEES.NS", "MON100.NS",
]
MARKET_TICKER = "NIFTYBEES.NS"   # "market index" baseline & factor for alpha
CASH = "CASH"


class DataSettings(BaseModel):
    tickers: list[str] = Field(default_factory=lambda: list(DEFAULT_TICKERS))
    start: str = "2011-04-01"
    end: Optional[str] = None
    freq: Literal["W", "D"] = "W"
    source: Literal["auto", "yahoo", "synthetic"] = "auto"
    train_end: str = "2019-12-31"
    val_end: str = "2021-12-31"
    embargo_bars: int = 4


class CostSettings(BaseModel):
    model: Literal["flat", "india"] = "india"
    flat_bps: float = 10.0            # per unit of notional traded (flat model)
    slippage_bps: float = 5.0         # half-spread / impact, added in both models
    # India-specific (equity delivery via a discount broker). Values are % of
    # traded notional; verify against your broker's current schedule.
    brokerage_pct: float = 0.0
    stt_sell_etf_pct: float = 0.001   # STT on sale of equity-oriented ETF units
    exchange_txn_pct: float = 0.00307  # NSE transaction charge
    sebi_fee_pct: float = 0.0001      # Rs 10 / crore
    stamp_buy_pct: float = 0.015      # stamp duty on buy side (delivery)
    gst_pct: float = 18.0             # GST on brokerage + exchange + SEBI fees


class PortfolioSettings(BaseModel):
    rf_annual: float = 0.065          # cash / risk-free yield (liquid fund proxy)
    max_weight: float = 0.40          # cap per risky asset
    lookback: int = 20                # price window length (bars) in the state
    risk_window: int = 26             # rolling window for vol / correlation
    opt_window: int = 52              # trailing window for classical optimisers
    exec_delay: int = 1               # 1 = obs uses data up to t-1 (execution at next bar)


class Settings(BaseModel):
    data: DataSettings = Field(default_factory=DataSettings)
    costs: CostSettings = Field(default_factory=CostSettings)
    portfolio: PortfolioSettings = Field(default_factory=PortfolioSettings)


class RewardSettings(BaseModel):
    risk: Literal["none", "variance", "downside", "drawdown"] = "downside"
    risk_lambda: float = 5.0
    concentration_kappa: float = 0.0   # penalty on Herfindahl above 1/N
    turnover_penalty: float = 0.0      # extra penalty per unit turnover (beyond cost)
    scale: float = 100.0               # reward scaling for optimiser stability


class AgentSettings(BaseModel):
    name: str = ""
    mode: Literal["hybrid", "direct"] = "hybrid"
    target: Literal["min_variance", "risk_parity", "hrp", "mean_variance", "equal_weight"] = "risk_parity"
    algo: Literal["PPO", "A2C", "SAC", "TD3", "DQN"] = "PPO"
    arch: Literal["mlp", "eiie", "attention"] = "mlp"
    reward: RewardSettings = Field(default_factory=RewardSettings)
    timesteps: int = 30000
    seeds: int = 3
    seed_offset: int = 0
    learning_rate: float = 3e-4
    episode_len: int = 104             # bars per training episode (random start)
    eval_every: int = 5000             # evaluate on validation every N steps, keep best
    no_trade_band: float = 0.0         # skip trades whose turnover is below this
    walk_forward: bool = False
