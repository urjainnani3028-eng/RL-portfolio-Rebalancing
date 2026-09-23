"""Process-wide state: persisted settings, active dataset, and caches for derived bundles
and baseline results (so the UI feels instant after the first computation)."""
from __future__ import annotations

import json
import threading
from typing import Optional

from .baselines import run_baselines
from .config import STORAGE, CostSettings, PortfolioSettings, Settings
from .data import Dataset, load_dataset, prepare_dataset
from .sim import Bundle, make_bundle

SETTINGS_PATH = STORAGE / "settings.json"
_lock = threading.RLock()
_bundles: dict = {}
_baselines: dict = {}
_active: dict = {"ds": None}


def load_settings() -> Settings:
    if SETTINGS_PATH.exists():
        try:
            return Settings(**json.loads(SETTINGS_PATH.read_text()).get("settings", {}))
        except Exception:  # noqa: BLE001
            pass
    return Settings()


def save_settings(s: Settings, active_id: Optional[str] = None) -> None:
    prev = {}
    if SETTINGS_PATH.exists():
        try:
            prev = json.loads(SETTINGS_PATH.read_text())
        except Exception:  # noqa: BLE001
            prev = {}
    SETTINGS_PATH.write_text(json.dumps({"settings": s.model_dump(),
                                         "active_dataset": active_id or prev.get("active_dataset")}, indent=2))


def active_dataset_id() -> Optional[str]:
    if SETTINGS_PATH.exists():
        try:
            return json.loads(SETTINGS_PATH.read_text()).get("active_dataset")
        except Exception:  # noqa: BLE001
            return None
    return None


def set_active(ds: Dataset, s: Settings) -> None:
    with _lock:
        _active["ds"] = ds
        save_settings(s, ds.id)


def get_active() -> Optional[Dataset]:
    with _lock:
        if _active["ds"] is not None:
            return _active["ds"]
        ds_id = active_dataset_id()
        if ds_id:
            ds = load_dataset(ds_id)
            if ds is not None:
                s = load_settings()
                ds.meta.update({"train_end": s.data.train_end, "val_end": s.data.val_end,
                                "embargo_bars": s.data.embargo_bars})
                _active["ds"] = ds
                return ds
    return None


def ensure_dataset(s: Settings, refresh: bool = False) -> Dataset:
    ds = prepare_dataset(s.data, s.portfolio.rf_annual, refresh=refresh)
    set_active(ds, s)
    return ds


def dataset_for(ds_id: str, s: Settings) -> Optional[Dataset]:
    ds = load_dataset(ds_id)
    if ds is not None:
        ds.meta.update({"train_end": s.data.train_end, "val_end": s.data.val_end, "embargo_bars": s.data.embargo_bars})
    return ds


def _bkey(ds: Dataset, p: PortfolioSettings) -> str:
    m = ds.meta
    return json.dumps([ds.id, m.get("sha"), m.get("train_end"), m.get("val_end"), m.get("embargo_bars"),
                       p.model_dump()], sort_keys=True)


def get_bundle(ds: Dataset, p: PortfolioSettings) -> Bundle:
    k = _bkey(ds, p)
    with _lock:
        if k not in _bundles:
            if len(_bundles) > 8:
                _bundles.clear()
            _bundles[k] = make_bundle(ds, p)
        return _bundles[k]


def get_baselines(ds: Dataset, s: Settings, segment: str, cost: Optional[CostSettings] = None) -> dict:
    cost = cost or s.costs
    b = get_bundle(ds, s.portfolio)
    k = _bkey(ds, s.portfolio) + json.dumps(cost.model_dump(), sort_keys=True) + str(segment)
    with _lock:
        if k in _baselines:
            return _baselines[k]
    res = run_baselines(b, cost, segment)
    with _lock:
        if len(_baselines) > 64:
            _baselines.clear()
        _baselines[k] = res
    return res
