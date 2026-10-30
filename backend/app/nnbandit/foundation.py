"""TimesFM (Google Research, open source: github.com/google-research/timesfm) as a zero-shot arm.

TimesFM is a pretrained decoder-only time-series foundation model (Das et al., ICML 2024). It is
used as released, with no training on our data, so it is deterministic: one inference pass serves
every bandit "seed", and its cost is inference time rather than training time.

How it forecasts here (same information set as the trained models):
    for decision t and ETF i, the context is ETF i's standardised log returns up to bar t - delay
    (at most `context` bars, train-fit scaler), and the forecast for bar t + 1 is step delay + 1
    of the forecast horizon. Each ETF is forecast as its own univariate series.

Weights are downloaded from Hugging Face on first use (or loaded from a local directory):
    2.5  google/timesfm-2.5-200m-pytorch   200M params, Apache-2.0 weights (default)
    3.0  google/timesfm-3.0-pytorch        weights for non-commercial use only
"""
from __future__ import annotations

import time

import numpy as np

from .features import SeqData

REPOS = {"2.5": "google/timesfm-2.5-200m-pytorch", "3.0": "google/timesfm-3.0-pytorch"}
PARAMS = {"2.5": 200_000_000, "3.0": 330_000_000}


class TimesFMUnavailable(RuntimeError):
    pass


def _load(version: str, checkpoint: str | None, context: int, horizon: int, batch: int):
    """Returns forecast(list of 1-D arrays) -> (n, horizon) point forecasts."""
    src = checkpoint or REPOS[version]
    try:
        if version == "2.5":
            import timesfm
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(src, torch_compile=False)
            model.compile(timesfm.ForecastConfig(
                max_context=context, max_horizon=horizon, per_core_batch_size=batch, normalize_inputs=True,
                infer_is_positive=False,          # returns can be negative: never clamp at zero
                force_flip_invariance=True, fix_quantile_crossing=True))
            return lambda xs: model.forecast(horizon=horizon, inputs=list(xs))[0]
        from timesfm3 import TimesFM3Forecaster
        model = TimesFM3Forecaster.from_pretrained(src, device="cpu")

        def run(xs):
            outs = model.predict_batch(list(xs), horizon=horizon, make_positive=False, return_quantiles=False)
            return np.stack([o.forecast for o in outs])
        return run
    except ImportError as e:
        raise TimesFMUnavailable("TimesFM is not installed: pip install 'timesfm[torch]'") from e
    except Exception as e:  # noqa: BLE001 - download / checkpoint errors come in many types
        raise TimesFMUnavailable(
            f"Could not load TimesFM {version} from '{src}' ({type(e).__name__}: {str(e)[:200]}). It downloads from "
            "huggingface.co on first use, so check your internet access, or pass a local checkpoint directory. "
            "To run without it, drop 'timesfm' from the models.") from e


def check_available(version: str = "2.5", checkpoint: str | None = None) -> None:
    _load(version, checkpoint, 32, 32, 1)


def forecast_timesfm(z: np.ndarray, data: SeqData, version: str = "2.5", checkpoint: str | None = None,
                     context: int = 256, batch: int = 32, forecast_fn=None) -> dict:
    """Point forecasts for every valid decision (T x N, NaN elsewhere), in the same standardised
    units as the trained models' targets. `forecast_fn` lets tests swap in a stub model."""
    d = data.delay
    horizon = d + 1
    t0 = time.perf_counter()
    fn = forecast_fn or _load(version, checkpoint, context, max(horizon, 1), batch)
    ts = np.flatnonzero(data.valid)
    T, N = data.y.shape
    jobs = [(t, i) for t in ts for i in range(N)]
    pred = np.full((T, N), np.nan, dtype=np.float32)
    for k in range(0, len(jobs), 512):
        chunk = jobs[k:k + 512]
        xs = [z[max(0, t - d - context + 1):t - d + 1, i].astype(np.float32) for t, i in chunk]
        out = np.asarray(fn(xs))
        for (t, i), row in zip(chunk, out):
            pred[t, i] = row[horizon - 1]
    return {"pred": pred, "seconds": time.perf_counter() - t0, "n_params": PARAMS.get(version, 0)}
