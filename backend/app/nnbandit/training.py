"""Train one (model, seed): Adam + L2 weight decay, Huber loss, gradient clipping, early stopping.

Early stopping uses an *inner* validation slice carved from the end of the training block
(separated by an embargo), so the outer validation block stays untouched for the bandit and the
test block stays locked.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
from torch import nn

from .features import SeqData
from .models import build_model, n_params

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def inner_split(data: SeqData, train_span: tuple[int, int], inner_bars: int, embargo: int):
    ts = data.decisions(*train_span)
    if len(ts) < inner_bars + embargo + 30:
        raise ValueError(f"Training block too short: {len(ts)} decisions")
    return ts[:-(inner_bars + embargo)], ts[-inner_bars:]


def _batches(n: int, size: int, rng: np.random.Generator):
    idx = rng.permutation(n)
    for i in range(0, n, size):
        yield idx[i:i + size]


@torch.no_grad()
def predict(model: nn.Module, data: SeqData, batch: int = 4096) -> np.ndarray:
    """Predictions for every valid decision (T x N, NaN elsewhere)."""
    model.eval()
    T, N = data.y.shape
    out = np.full((T, N), np.nan, dtype=np.float32)
    ts = np.flatnonzero(data.valid)
    X, _ = data.flat(ts)
    preds = [model(torch.from_numpy(X[i:i + batch])).numpy() for i in range(0, len(X), batch)]
    out[ts] = np.concatenate(preds).reshape(len(ts), N)
    return out


def lag_saliency(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Mean |d prediction / d input| per lag (oldest -> newest), normalised to sum to 1.
    Shows how far back each architecture effectively looks (vanishing gradients show up here)."""
    model.eval()
    x = torch.from_numpy(X[:1024]).requires_grad_(True)
    model(x).sum().backward()
    g = x.grad.abs().mean(dim=(0, 2)).numpy()
    return g / (g.sum() + 1e-12)


def train_one(model_id: str, data: SeqData, train_span: tuple[int, int], seed: int, *, hidden: int = 64,
              layers: int = 2, dropout: float = 0.1, lr: float = 1e-3, weight_decay: float = 1e-4,
              batch_size: int = 128, max_epochs: int = 60, patience: int = 8, inner_bars: int = 52,
              embargo: int = 4, stop=lambda: False) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    fit_t, val_t = inner_split(data, train_span, inner_bars, embargo)
    Xf, yf = (torch.from_numpy(a) for a in data.flat(fit_t))
    Xv, yv = (torch.from_numpy(a) for a in data.flat(val_t))

    model = build_model(model_id, data.n_features, data.lookback, hidden, layers, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.HuberLoss(delta=1.0)
    best, best_state, best_epoch, bad = np.inf, None, 0, 0
    hist_train, hist_val = [], []
    t0 = time.perf_counter()
    for epoch in range(max_epochs):
        model.train()
        tot = 0.0
        for idx in _batches(len(yf), batch_size, rng):
            opt.zero_grad()
            loss = loss_fn(model(Xf[idx]), yf[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            v = float(loss_fn(model(Xv), yv))
        hist_train.append(tot / len(yf))
        hist_val.append(v)
        if v < best - 1e-5:
            best, best_epoch, bad = v, epoch, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if stop():
            break
    model.load_state_dict(best_state)
    seconds = time.perf_counter() - t0
    return {
        "model": model_id, "seed": seed, "pred": predict(model, data),
        "saliency": lag_saliency(model, Xv.numpy()), "train_loss": hist_train, "val_loss": hist_val,
        "best_epoch": best_epoch, "epochs": len(hist_train), "best_inner_val_loss": best,
        "seconds": seconds, "n_params": n_params(model),
    }
