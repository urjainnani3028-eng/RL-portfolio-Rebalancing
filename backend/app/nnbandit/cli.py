"""Command-line runner for the model-selection study.

    cd backend
    python -m app.nnbandit.cli                       # full study on validation (test stays locked)
    python -m app.nnbandit.cli --test                # ... and the one-shot test evaluation
    python -m app.nnbandit.cli --quick               # small smoke run (few seeds / epochs)
    python -m app.nnbandit.cli --source synthetic    # offline prices
    python -m app.nnbandit.cli --models rnn lstm gru tcn patchtst   # without TimesFM (no download)

Writes results JSON to storage/nnbandit/results/ and a Markdown report + PNG charts to --out.
"""
from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

from ..config import Settings
from ..data import prepare_dataset
from ..sim import make_bundle
from .bandit import ALGORITHMS
from .experiment import BanditSettings, run_study
from .models import DEFAULT_MODELS, MODEL_IDS
from .report import write_report


def cfg_min_train(s: Settings) -> int:
    """Training weeks needed: the 52-week inner validation slice + 4-week embargo + at least ~2 years to fit."""
    return 52 + 4 + 104


def main(argv=None):
    ap = argparse.ArgumentParser(description="RNN / LSTM / GRU / TCN / TimesFM selection with a multi-armed bandit")
    ap.add_argument("--algorithm", default="successive_halving", choices=list(ALGORITHMS))
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=MODEL_IDS)
    ap.add_argument("--timesfm-version", default="2.5", choices=["2.5", "3.0"])
    ap.add_argument("--timesfm-checkpoint", default=None, help="local TimesFM checkpoint (default: Hugging Face)")
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--max-seeds", type=int, default=8)
    ap.add_argument("--window", type=int, default=26)
    ap.add_argument("--replay-runs", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", default="auto", choices=["auto", "yahoo", "synthetic"])
    ap.add_argument("--refresh", action="store_true", help="re-download prices from Yahoo")
    ap.add_argument("--tickers", nargs="+", default=None,
                    help="ETF universe (default: the 7 in app/config.py), e.g. to drop a late-listed ETF")
    ap.add_argument("--test", action="store_true", help="unlock the test block (one-shot)")
    ap.add_argument("--quick", action="store_true", help="smoke run: 3 seeds, 15 epochs, budget 15")
    ap.add_argument("--out", default="storage/nnbandit/report")
    a = ap.parse_args(argv)

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.WARNING)
    s = Settings()
    s.data.source = a.source
    if a.tickers:
        s.data.tickers = [t if t.endswith(".NS") else t + ".NS" for t in a.tickers]
    ds = prepare_dataset(s.data, s.portfolio.rf_annual, refresh=a.refresh)
    print(f"Dataset {ds.id}: {ds.meta['source']}, {ds.meta['start']} .. {ds.meta['end']}, {ds.meta['bars']} bars")
    for w in ds.meta.get("warnings", []):
        print("  !", w)
    first = ds.meta.get("first_dates", {})
    if ds.meta["source"] != "synthetic":
        print("  First price per ETF: " + ", ".join(f"{k.replace('.NS', '')} {v}" for k, v in first.items()))
    b = make_bundle(ds, s.portfolio)
    n_train = b.segments["train"][1] - b.warmup
    if n_train < cfg_min_train(s):
        late = max(first, key=first.get) if first else None
        raise SystemExit(
            f"\nOnly {max(n_train, 0)} training weeks: the dataset starts {ds.meta['start']}, because every ETF must have "
            f"prices on every date"
            + (f" and {late.replace('.NS', '')} only starts {first[late]}" if late else "") + ".\n"
            "Drop the late ETF with --tickers, e.g.:\n  ../.venv/bin/python -m app.nnbandit.cli --quick --tickers "
            + " ".join(t.replace(".NS", "") for t in s.data.tickers if t != late) + "\n")
    cfg = BanditSettings(models=a.models, algorithm=a.algorithm, budget=a.budget, max_seeds=a.max_seeds,
                         window=a.window, replay_runs=a.replay_runs, max_epochs=a.epochs, seed=a.seed,
                         timesfm_version=a.timesfm_version, timesfm_checkpoint=a.timesfm_checkpoint,
                         evaluate_test=a.test)
    if a.quick:
        cfg = cfg.model_copy(update={"max_seeds": 3, "max_epochs": 15, "budget": 15, "replay_runs": 100})

    last = [-1.0]

    def progress(p, msg):
        if p - last[0] >= 0.02 or p >= 1:
            last[0] = p
            print(f"[{p * 100:5.1f}%] {msg}", flush=True)

    res = run_study(b, s.costs, cfg, ds.meta, progress=progress)
    out = write_report(res, Path(a.out))
    print(f"\nBandit ({cfg.algorithm}) winner: {res['live']['winner']}   "
          f"true best on validation: {res['replay']['true_best']}")
    print(f"Report: {out}")


if __name__ == "__main__":
    main()
