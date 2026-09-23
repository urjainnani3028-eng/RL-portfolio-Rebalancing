"""Data pipeline: Yahoo Finance download -> split repair -> alignment -> resample ->
frozen Parquet snapshot. Falls back to a calibrated synthetic market if Yahoo is
unreachable, so the whole app still runs offline (clearly labelled as synthetic)."""
from __future__ import annotations

import hashlib
import io
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd

from .config import CACHE_DIR, CASH, CATALOGUE_BY_TICKER, DATASETS_DIR, DataSettings

log = logging.getLogger("data")


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------
@dataclass
class Dataset:
    id: str
    prices: pd.DataFrame            # bars x (assets + CASH), CASH last
    meta: dict = field(default_factory=dict)

    @property
    def assets(self) -> list[str]:
        return [c for c in self.prices.columns if c != CASH]

    @property
    def returns(self) -> pd.DataFrame:
        """Simple returns per bar. Row t = return from close t-1 to close t (row 0 = 0)."""
        r = self.prices.pct_change().fillna(0.0)
        return r

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.prices.index

    def periods_per_year(self) -> int:
        return 52 if self.meta.get("freq", "W") == "W" else 252


# ---------------------------------------------------------------------------
# Yahoo download with per-ticker cache
# ---------------------------------------------------------------------------
def _fetch_one_yahoo(ticker: str, start: str, end: Optional[str], refresh: bool) -> pd.Series:
    path = CACHE_DIR / f"raw_{ticker.replace('^', '_')}.parquet"
    if path.exists() and not refresh:
        s = pd.read_parquet(path)["close"]
        s.name = ticker
        return s
    import yfinance as yf  # imported lazily: slow import

    hist = yf.Ticker(ticker).history(start="2000-01-01", auto_adjust=True, actions=False)
    if hist is None or hist.empty or "Close" not in hist:
        raise RuntimeError(f"No data returned for {ticker}")
    s = hist["Close"].astype(float)
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    s = s[~s.index.duplicated(keep="last")].dropna()
    s = s[s > 0]
    out = pd.DataFrame({"close": s})
    out.to_parquet(path)
    s.name = ticker
    return s


def repair_splits(s: pd.Series, name: str) -> tuple[pd.Series, list[dict]]:
    """Yahoo's NSE data sometimes misses unit splits (e.g. 1:10, 1:100), which show up
    as a one-day price jump by an (almost) integer factor. Detect and back-adjust."""
    fixes = []
    s = s.copy()
    ratios = s / s.shift(1)
    for dt, ratio in ratios.dropna().items():
        if not (ratio < 0.55 or ratio > 1.8):
            continue
        for k in (2, 3, 4, 5, 10, 20, 25, 50, 100):
            for fac in (1.0 / k, float(k)):
                if abs(ratio / fac - 1) < 0.06:
                    s.loc[s.index < dt] = s.loc[s.index < dt] * fac
                    fixes.append({"ticker": name, "date": str(dt.date()), "factor": round(fac, 4)})
                    break
            else:
                continue
            break
    return s, fixes


# ---------------------------------------------------------------------------
# Synthetic, regime-switching market calibrated to rough Indian asset stats
# ---------------------------------------------------------------------------
_SYN_PARAMS = {
    # ticker: (annual drift, annual vol, factor loadings [india_eq, global_eq, gold, rates])
    "NIFTYBEES.NS": (0.12, 0.17, [1.00, 0.20, 0.00, 0.00]),
    "JUNIORBEES.NS": (0.14, 0.21, [1.10, 0.15, 0.00, 0.00]),
    "BANKBEES.NS": (0.13, 0.25, [1.25, 0.10, 0.00, -0.10]),
    "PSUBNKBEES.NS": (0.07, 0.34, [1.35, 0.05, 0.00, -0.15]),
    "INFRABEES.NS": (0.09, 0.23, [1.15, 0.10, 0.00, -0.05]),
    "GOLDBEES.NS": (0.10, 0.15, [-0.05, 0.00, 1.00, 0.10]),
    "MON100.NS": (0.19, 0.22, [0.25, 1.00, 0.00, 0.00]),
    "CPSEETF.NS": (0.11, 0.26, [1.10, 0.05, 0.00, -0.05]),
    "LICNETFGSC.NS": (0.072, 0.055, [-0.05, 0.00, 0.05, 1.00]),
    "SETF10GILT.NS": (0.070, 0.050, [-0.05, 0.00, 0.05, 1.00]),
    "SILVERBEES.NS": (0.09, 0.26, [0.10, 0.05, 1.20, 0.00]),
    "ITBEES.NS": (0.15, 0.24, [0.70, 0.60, 0.00, 0.00]),
}


def synthetic_prices(tickers: list[str], start: str, end: Optional[str], seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(pd.Timestamp(start) - pd.Timedelta(days=400), pd.Timestamp(end or date.today()))
    n = len(idx)
    dt = 1 / 252
    # Two-state Markov regime: calm / stressed
    p_stay = {0: 0.995, 1: 0.975}
    regime = np.zeros(n, dtype=int)
    for t in range(1, n):
        regime[t] = regime[t - 1] if rng.random() < p_stay[regime[t - 1]] else 1 - regime[t - 1]
    f_vol = np.array([0.15, 0.17, 0.14, 0.045])
    f_drift_stress = np.array([-0.45, -0.35, 0.10, 0.02])
    factors = rng.standard_normal((n, 4)) * f_vol * np.sqrt(dt)
    factors *= np.where(regime[:, None] == 1, 1.9, 1.0)
    # stressed regime drifts down; calm regime compensates so unconditional factor drift ~ 0
    p_st = (1 - p_stay[0]) / ((1 - p_stay[0]) + (1 - p_stay[1]))
    f_drift_calm = -f_drift_stress * p_st / (1 - p_st)
    factors += np.where(regime[:, None] == 1, f_drift_stress * dt, f_drift_calm * dt)
    # fat tails via occasional (mean-compensated) crash jumps in the Indian equity factor
    jumps = (rng.random(n) < 0.004) * rng.normal(-0.04, 0.03, n)
    factors[:, 0] += jumps + 0.004 * 0.04
    out = {}
    for tk in tickers:
        mu, vol, load = _SYN_PARAMS.get(tk, (0.10, 0.20, [0.9, 0.1, 0.0, 0.0]))
        load = np.array(load)
        sys = factors @ load
        sys_var = float(np.sum((load * f_vol) ** 2))
        idio_vol = np.sqrt(max(vol ** 2 - sys_var, 0.03 ** 2))
        idio = rng.standard_normal(n) * idio_vol * np.sqrt(dt)
        r = mu * dt - 0.5 * vol ** 2 * dt + sys + idio
        out[tk] = 100 * np.exp(np.cumsum(r))
    df = pd.DataFrame(out, index=idx)
    return df[df.index >= pd.Timestamp(start)]


# ---------------------------------------------------------------------------
# Assemble a dataset
# ---------------------------------------------------------------------------
def _resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if freq == "W":
        out = df.resample("W-FRI").last()
        out.index = df.index.to_series().resample("W-FRI").last().values   # label = last real trading day
        return out.dropna(how="all")
    return df


def _add_cash(prices: pd.DataFrame, rf: float) -> pd.DataFrame:
    days = (prices.index - prices.index[0]).days.values.astype(float)
    prices = prices.copy()
    prices[CASH] = 100.0 * (1 + rf) ** (days / 365.0)
    return prices


def _dataset_id(cfg: DataSettings, source: str, rf: float) -> str:
    key = json.dumps({"t": sorted(cfg.tickers), "s": cfg.start, "e": cfg.end, "f": cfg.freq, "src": source,
                      "rf": round(rf, 6)}, sort_keys=True)
    return hashlib.sha1(key.encode()).hexdigest()[:10]


def _finalise(raw: pd.DataFrame, cfg: DataSettings, rf: float, source: str, fixes: list, first_dates: dict,
              warnings: list) -> Dataset:
    raw = raw.sort_index()
    raw = raw[raw.index >= pd.Timestamp(cfg.start)]
    if cfg.end:
        raw = raw[raw.index <= pd.Timestamp(cfg.end)]
    # common start: all assets must have data
    common_start = raw.apply(lambda s: s.first_valid_index()).max()
    raw = raw[raw.index >= common_start].ffill(limit=5).dropna()
    prices = _resample(raw, cfg.freq)
    prices = _add_cash(prices, rf)
    ds_id = _dataset_id(cfg, source, rf)
    content_hash = hashlib.sha256(pd.util.hash_pandas_object(prices, index=True).values.tobytes()).hexdigest()[:12]
    if prices.index[0] > pd.Timestamp(cfg.train_end) - pd.DateOffset(years=4):
        warnings.append("Very short training history: common start date is " + str(prices.index[0].date()) +
                        ". Consider removing late-listed ETFs.")
    meta = {
        "id": ds_id, "source": source, "freq": cfg.freq, "tickers": list(cfg.tickers),
        "created": datetime.now().isoformat(timespec="seconds"), "sha": content_hash,
        "start": str(prices.index[0].date()), "end": str(prices.index[-1].date()), "bars": len(prices),
        "split_fixes": fixes, "first_dates": first_dates, "warnings": warnings,
        "train_end": cfg.train_end, "val_end": cfg.val_end, "embargo_bars": cfg.embargo_bars,
    }
    return Dataset(id=ds_id, prices=prices, meta=meta)


def save_dataset(ds: Dataset) -> None:
    ds.prices.to_parquet(DATASETS_DIR / f"{ds.id}.parquet")
    (DATASETS_DIR / f"{ds.id}.json").write_text(json.dumps(ds.meta, indent=2))


def load_dataset(ds_id: str) -> Optional[Dataset]:
    p, m = DATASETS_DIR / f"{ds_id}.parquet", DATASETS_DIR / f"{ds_id}.json"
    if not (p.exists() and m.exists()):
        return None
    return Dataset(id=ds_id, prices=pd.read_parquet(p), meta=json.loads(m.read_text()))


def prepare_dataset(cfg: DataSettings, rf: float, refresh: bool = False) -> Dataset:
    """Load the frozen snapshot for these settings, or build (and freeze) a new one."""
    sources = ["yahoo", "synthetic"] if cfg.source == "auto" else [cfg.source]
    errors = []
    for src in sources:
        if not refresh:
            ds = load_dataset(_dataset_id(cfg, src, rf))
            if ds is not None:
                ds.meta.update({"train_end": cfg.train_end, "val_end": cfg.val_end, "embargo_bars": cfg.embargo_bars})
                return ds
        try:
            if src == "yahoo":
                series, fixes, first_dates = [], [], {}
                for tk in cfg.tickers:
                    s = _fetch_one_yahoo(tk, cfg.start, cfg.end, refresh)
                    s, fx = repair_splits(s, tk)
                    fixes += fx
                    first_dates[tk] = str(s.index[0].date())
                    series.append(s.rename(tk))
                raw = pd.concat(series, axis=1)
                warnings = [f"Repaired {len(fixes)} unadjusted split(s) in Yahoo data."] if fixes else []
                ds = _finalise(raw, cfg, rf, "yahoo", fixes, first_dates, warnings)
            else:
                raw = synthetic_prices(cfg.tickers, cfg.start, cfg.end)
                warn = ["SYNTHETIC DATA: Yahoo Finance was unreachable or not selected. Results are for pipeline "
                        "testing only - connect to the internet and click 'Refresh from Yahoo' for real data."]
                if errors:
                    warn.append("Yahoo error: " + errors[-1][:200])
                ds = _finalise(raw, cfg, rf, "synthetic", [], {t: cfg.start for t in cfg.tickers}, warn)
            save_dataset(ds)
            return ds
        except Exception as e:  # noqa: BLE001
            log.warning("source %s failed: %s", src, e)
            errors.append(f"{src}: {e}")
    raise RuntimeError("Could not build dataset. " + " | ".join(errors))


def dataset_from_csv(content: bytes, cfg: DataSettings, rf: float) -> Dataset:
    df = pd.read_csv(io.BytesIO(content))
    date_col = next((c for c in df.columns if c.lower() in ("date", "datetime", "time")), df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index().apply(pd.to_numeric, errors="coerce")
    df = df.drop(columns=[c for c in df.columns if c.upper() == CASH], errors="ignore")
    cfg = cfg.model_copy(update={"tickers": list(df.columns)})
    fixes = []
    cols = []
    for c in df.columns:
        s, fx = repair_splits(df[c].dropna(), c)
        fixes += fx
        cols.append(s.rename(c))
    raw = pd.concat(cols, axis=1)
    ds = _finalise(raw, cfg, rf, "csv", fixes, {c: str(raw[c].first_valid_index().date()) for c in raw}, [])
    ds.id = "csv_" + ds.meta["sha"][:8]
    ds.meta["id"] = ds.id
    save_dataset(ds)
    return ds


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def split_indices(ds: Dataset) -> dict[str, tuple[int, int]]:
    """Return decision-index ranges [start, stop) per segment. A decision at index t
    earns the return of bar t+1, so a segment [a, b) uses returns a+1..b.
    An embargo of `embargo_bars` separates consecutive segments."""
    dates = ds.dates
    emb = int(ds.meta.get("embargo_bars", 4))
    T = len(dates)
    tr_end = int(np.searchsorted(dates, pd.Timestamp(ds.meta["train_end"]), side="right")) - 1
    va_end = int(np.searchsorted(dates, pd.Timestamp(ds.meta["val_end"]), side="right")) - 1
    tr_end = max(min(tr_end, T - 3), 1)
    va_end = max(min(va_end, T - 2), tr_end + 1)
    return {
        "train": (0, tr_end),
        "val": (min(tr_end + emb, va_end - 1), va_end),
        "test": (min(va_end + emb, T - 2), T - 1),
    }


def asset_info(ds: Dataset) -> list[dict]:
    out = []
    for a in ds.prices.columns:
        c = CATALOGUE_BY_TICKER.get(a, {})
        out.append({"ticker": a, "name": c.get("name", "Cash (liquid fund proxy)" if a == CASH else a),
                    "cls": c.get("cls", "Cash" if a == CASH else "Custom"),
                    "equity": bool(c.get("equity", a != CASH))})
    return out
