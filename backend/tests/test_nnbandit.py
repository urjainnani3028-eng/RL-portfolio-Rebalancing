"""Tests for the neural-forecaster + multi-armed bandit module (offline, synthetic market)."""
import numpy as np
import pytest
import torch

from app.config import Settings
from app.data import prepare_dataset
from app.nnbandit.bandit import ALGORITHMS, make_bandit
from app.nnbandit.features import build_sequences
from app.nnbandit.foundation import forecast_timesfm
from app.nnbandit.models import TRAINABLE, build_model
from app.nnbandit.portfolio import sharpe_and_mdd, target_weights, window_reward
from app.nnbandit.training import train_one
from app.sim import make_bundle


@pytest.fixture(scope="module")
def bundle():
    s = Settings()
    s.data.source = "synthetic"
    ds = prepare_dataset(s.data, s.portfolio.rf_annual)
    return make_bundle(ds, s.portfolio)


def test_target_is_next_bar(bundle):
    d = build_sequences(bundle)
    ts = np.flatnonzero(d.valid)
    assert np.allclose(d.y[ts], bundle.z[ts + 1])
    assert np.allclose(d.ret[ts], bundle.R[ts + 1, :bundle.n])


def test_no_lookahead_in_windows(bundle):
    """Corrupting every bar after t - exec_delay must not change the window for decision t."""
    d0 = build_sequences(bundle)
    t = int(np.flatnonzero(d0.valid)[100])
    cut = t - bundle.pcfg.exec_delay + 1
    z, v, c = bundle.z.copy(), bundle.vol_z.copy(), bundle.corr_z.copy()
    try:
        bundle.z[cut:] = 99.0
        bundle.vol_z[cut:] = 99.0
        bundle.corr_z[cut:] = 99.0
        d1 = build_sequences(bundle)
    finally:
        bundle.z, bundle.vol_z, bundle.corr_z = z, v, c
    assert np.array_equal(d0.X[t], d1.X[t])
    assert not np.array_equal(d0.X[t + 2], d1.X[t + 2])      # negative control


@pytest.mark.parametrize("model_id", TRAINABLE)
def test_models_forward_and_backward(model_id):
    m = build_model(model_id, n_features=11, lookback=20)
    x = torch.randn(8, 20, 11, requires_grad=True)
    y = m(x)
    assert y.shape == (8,)
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("model_id", ["tcn", "patchtst"])
def test_open_source_models_are_causal_in_time(model_id):
    """The prediction depends on the window only (no leakage across samples in a batch)."""
    m = build_model(model_id, n_features=11, lookback=20).eval()
    x = torch.randn(4, 20, 11)
    with torch.no_grad():
        a = m(x)
        x2 = x.clone()
        x2[1:] += 5.0
        b = m(x2)
    assert torch.allclose(a[0], b[0], atol=1e-5)


def test_training_predicts_every_valid_decision(bundle):
    d = build_sequences(bundle)
    out = train_one("gru", d, bundle.segments["train"], seed=0, max_epochs=3, patience=3)
    assert out["pred"].shape == d.y.shape
    assert np.isfinite(out["pred"][d.valid]).all() and np.isnan(out["pred"][~d.valid]).all()
    assert abs(out["saliency"].sum() - 1) < 1e-4


def test_flat_forecast_is_equal_weight():
    w = target_weights(np.zeros(7), tilt=5.0, cap=0.4)
    assert np.allclose(w[:-1], 1 / 7) and w[-1] == 0
    w = target_weights(np.array([3.0, 0, 0, 0, 0, 0, 0]), tilt=5.0, cap=0.4)
    assert abs(w.sum() - 1) < 1e-9 and w[0] <= 0.4 + 1e-9


def test_reward_relative_to_reference_is_centered():
    lr = np.random.default_rng(0).normal(0.002, 0.02, 26)
    hits = np.full(26, 0.5)
    r, _ = window_reward(lr, hits, 52, 0.001, {"sharpe": 1.0, "drawdown": 1.0, "accuracy": 1.0},
                         ref=sharpe_and_mdd(lr, 52, 0.001))
    assert abs(r - 0.5) < 1e-9     # same returns as the reference + 50% hit rate -> exactly neutral


@pytest.mark.parametrize("name", [a for a in ALGORITHMS if a != "uniform"])
def test_adaptive_bandits_find_clear_winner(name):
    """5 Gaussian arms, clear gap: every adaptive algorithm should identify the best arm and pull it
    more often than uniform allocation would."""
    means = np.array([0.45, 0.50, 0.55, 0.70, 0.48])
    correct, on_best = 0, 0
    for run in range(60):
        rng = np.random.default_rng(run)
        bd = make_bandit(name, 5, 40, run)
        for _ in range(40):
            k = bd.select()
            bd.update(k, float(np.clip(rng.normal(means[k], 0.1), 0, 1)))
        correct += bd.recommend() == 3
        on_best += bd.n[3]
    assert correct / 60 >= 0.85
    assert on_best / 60 > 40 / 5


def test_successive_halving_respects_budget():
    bd = make_bandit("successive_halving", 5, 40, 0)
    for _ in range(40):
        bd.update(bd.select(), 0.5)
    assert bd.n.sum() == 40


def test_timesfm_wrapper_uses_only_past_data(bundle):
    """With a stub in place of the pretrained model: each context must end at t - exec_delay, and the
    forecast for bar t + 1 is horizon step exec_delay + 1."""
    d = build_sequences(bundle)
    seen = []

    def stub(xs):
        seen.extend(len(x) for x in xs)
        # echo the last context value at every horizon step, so the mapping is checkable
        return np.stack([np.array([x[-1], x[-1] + 1.0]) for x in xs])

    out = forecast_timesfm(bundle.z, d, context=64, forecast_fn=stub)
    ts = np.flatnonzero(d.valid)
    delay = bundle.pcfg.exec_delay
    assert max(seen) <= 64
    t = ts[50]
    expected = bundle.z[t - delay, :] + (1.0 if delay == 1 else 0.0)
    assert np.allclose(out["pred"][t], expected)
    assert np.isnan(out["pred"][~d.valid]).all()
