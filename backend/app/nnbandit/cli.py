"""Command-line runner for the model-selection study.

    cd backend
    python -m app.nnbandit.cli                       # full study on validation (test stays locked)
    python -m app.nnbandit.cli --test                # ... and the one-shot test evaluation
    python -m app.nnbandit.cli --quick               # small smoke run (few seeds / epochs)
    python -m app.nnbandit.cli --source synthetic    # offline

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
from .models import MODEL_IDS
from .report import write_report


def main(argv=None):
    ap = argparse.ArgumentParser(description="RNN / LSTM / GRU / TCN / PatchTST selection with a multi-armed bandit")
    ap.add_argument("--algorithm", default="thompson", choices=list(ALGORITHMS))
    ap.add_argument("--models", nargs="+", default=MODEL_IDS, choices=MODEL_IDS)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--max-seeds", type=int, default=8)
    ap.add_argument("--window", type=int, default=26)
    ap.add_argument("--replay-runs", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", default="auto", choices=["auto", "yahoo", "synthetic"])
    ap.add_argument("--refresh", action="store_true", help="re-download prices from Yahoo")
    ap.add_argument("--test", action="store_true", help="unlock the test block (one-shot)")
    ap.add_argument("--quick", action="store_true", help="smoke run: 3 seeds, 15 epochs, budget 15")
    ap.add_argument("--out", default="storage/nnbandit/report")
    a = ap.parse_args(argv)

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.WARNING)
    s = Settings()
    s.data.source = a.source
    ds = prepare_dataset(s.data, s.portfolio.rf_annual, refresh=a.refresh)
    print(f"Dataset {ds.id}: {ds.meta['source']}, {ds.meta['start']} .. {ds.meta['end']}, {ds.meta['bars']} bars")
    for w in ds.meta.get("warnings", []):
        print("  !", w)
    b = make_bundle(ds, s.portfolio)
    cfg = BanditSettings(models=a.models, algorithm=a.algorithm, budget=a.budget, max_seeds=a.max_seeds,
                         window=a.window, replay_runs=a.replay_runs, max_epochs=a.epochs, seed=a.seed,
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
