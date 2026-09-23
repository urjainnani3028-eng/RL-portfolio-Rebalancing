"""Run with:  cd backend && python -m pytest -q
Uses the synthetic market so it works offline."""
import numpy as np
import pytest

from app.config import AgentSettings, Settings
from app.costs import CostModel
from app.data import prepare_dataset
from app.optimizers import apply_cap
from app.sanity import FAST
from app.sim import PortfolioEnv, make_bundle


@pytest.fixture(scope="module")
def setup():
    s = Settings()
    s.data.source = "synthetic"
    ds = prepare_dataset(s.data, s.portfolio.rf_annual)
    return ds, s


@pytest.mark.parametrize("test", FAST, ids=lambda f: f.__name__)
def test_fast_sanity(setup, test):
    ds, s = setup
    r = test(ds, s)
    assert r["passed"], r["detail"]


def test_cap_projection():
    rng = np.random.default_rng(0)
    for _ in range(200):
        w = apply_cap(rng.dirichlet(np.ones(8) * 0.3), 0.3)
        assert abs(w.sum() - 1) < 1e-9 and (w >= 0).all() and (w[:-1] <= 0.3 + 1e-9).all()


def test_india_costs_asymmetric():
    s = Settings()
    cm = CostModel(s.costs, [True, False])
    assert cm.buy[0] > cm.sell[0] - 1e-12          # stamp duty on buys
    assert cm.sell[0] > cm.sell[1]                # STT only on equity ETF sells
    w0, w1 = np.array([0, 0, 1.0]), np.array([0.5, 0.5, 0.0])
    assert cm.cost(w0, w1) > 0 and cm.cost(w1, w1) == 0


@pytest.mark.parametrize("mode", ["hybrid", "direct"])
def test_no_lookahead_in_observation(setup, mode):
    """Corrupting every return from the decision bar onward must not change the observation."""
    from app.sim import _compute_features

    ds, s = setup
    agent = AgentSettings(mode=mode)
    b = make_bundle(ds, s.portfolio)
    env = PortfolioEnv(b, agent, s.costs, segment="val", training=False)
    obs0, _ = env.reset()
    t = env.sim.t
    b2 = make_bundle(ds, s.portfolio)
    b2.key, b2._targets = "", {}                        # no cache: recompute targets from corrupted data
    b2.R = b.R.copy()
    b2.R[t - s.portfolio.exec_delay + 1:, :-1] *= -3.0  # corrupt the "future"
    _compute_features(b2)
    env2 = PortfolioEnv(b2, agent, s.costs, segment="val", training=False)
    obs1, _ = env2.reset()
    assert np.allclose(obs0, obs1, atol=1e-6)

    # negative control: corrupting the last *observable* bar must change the observation
    b3 = make_bundle(ds, s.portfolio)
    b3.key, b3._targets = "", {}
    b3.R = b.R.copy()
    b3.R[t - s.portfolio.exec_delay:, :-1] *= -3.0
    _compute_features(b3)
    obs2, _ = PortfolioEnv(b3, agent, s.costs, segment="val", training=False).reset()
    assert not np.allclose(obs0, obs2, atol=1e-6)
