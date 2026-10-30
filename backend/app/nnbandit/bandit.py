"""Multi-armed bandit algorithms. One arm = one neural architecture; one pull = train/evaluate
that architecture once more (new seed and/or a new validation window); reward in [0, 1].

All algorithms share the interface  select() -> arm,  update(arm, reward),  recommend() -> arm.

    uniform             round-robin: the "test everything equally" baseline from the plan
    epsilon_greedy      exploit the best mean, explore uniformly with probability eps
    ucb1                Auer, Cesa-Bianchi & Fischer (2002): mean + c * sqrt(2 ln t / n)
    ucb1_tuned          UCB1 with a variance-aware bonus (same paper) - suits low-variance rewards
    thompson            Gaussian Thompson sampling with a Normal-Gamma posterior (unknown variance)
    successive_halving  Karnin, Koren & Somekh (2013): fixed-budget best-arm identification,
                        drop the worse half of the arms each round
"""
from __future__ import annotations

import math

import numpy as np


class Bandit:
    name = "base"

    def __init__(self, n_arms: int, budget: int, rng: np.random.Generator, warmup: int = 2):
        self.k, self.budget, self.rng = n_arms, budget, rng
        self.warmup = warmup              # pulls per arm before adapting (one noisy reward is not evidence)
        self.n = np.zeros(n_arms, int)
        self.s = np.zeros(n_arms)
        self.s2 = np.zeros(n_arms)
        self.t = 0

    @property
    def means(self) -> np.ndarray:
        return np.where(self.n > 0, self.s / np.maximum(self.n, 1), -np.inf)

    def _argmax(self, x: np.ndarray) -> int:
        best = np.flatnonzero(x == np.max(x))
        return int(self.rng.choice(best))

    def _unpulled(self):
        z = np.flatnonzero(self.n < min(self.warmup, max(1, self.budget // self.k)))
        return int(self.rng.choice(z[self.n[z] == self.n[z].min()])) if len(z) else None

    def select(self) -> int:
        raise NotImplementedError

    def update(self, arm: int, reward: float) -> None:
        self.n[arm] += 1
        self.s[arm] += reward
        self.s2[arm] += reward * reward
        self.t += 1

    def recommend(self) -> int:
        """Most-played arm, ties broken by the empirical mean (the robust "MPA" rule of
        Bubeck, Munos & Stoltz 2009 for recommending after an adaptive allocation)."""
        return self._argmax(self.n + 1e-6 * np.nan_to_num(self.means, neginf=-1e6))


class Uniform(Bandit):
    name = "uniform"

    def select(self):
        return self._argmax(-self.n.astype(float))

    def recommend(self):
        return self._argmax(self.means)


class EpsilonGreedy(Bandit):
    name = "epsilon_greedy"

    def __init__(self, *a, eps: float = 0.1, **kw):
        super().__init__(*a, **kw)
        self.eps = eps

    def select(self):
        u = self._unpulled()
        if u is not None:
            return u
        if self.rng.random() < self.eps:
            return int(self.rng.integers(self.k))
        return self._argmax(self.means)


class UCB1(Bandit):
    name = "ucb1"

    def __init__(self, *a, c: float = 1.0, **kw):
        super().__init__(*a, **kw)
        self.c = c

    def bonus(self) -> np.ndarray:
        return self.c * np.sqrt(2 * math.log(max(self.t, 1)) / self.n)

    def select(self):
        u = self._unpulled()
        if u is not None:
            return u
        return self._argmax(self.means + self.bonus())


class UCB1Tuned(UCB1):
    name = "ucb1_tuned"

    def bonus(self):
        lt = math.log(max(self.t, 1))
        var = self.s2 / self.n - (self.s / self.n) ** 2 + np.sqrt(2 * lt / self.n)
        return self.c * np.sqrt(lt / self.n * np.minimum(0.25, var))


class Thompson(Bandit):
    """Normal-Gamma posterior per arm; prior mean 0.5 and a prior reward sd of about 0.15
    (alpha0 / beta0 = expected precision), so a lucky first window cannot lock the arm in."""
    name = "thompson"

    def __init__(self, *a, mu0: float = 0.5, kappa0: float = 1.0, alpha0: float = 2.0, beta0: float = 0.05, **kw):
        super().__init__(*a, **kw)
        self.mu0, self.kappa0, self.alpha0, self.beta0 = mu0, kappa0, alpha0, beta0

    def select(self):
        u = self._unpulled()
        if u is not None:
            return u
        n = self.n
        xbar = self.s / n
        ss = np.maximum(self.s2 - n * xbar ** 2, 0.0)
        kn = self.kappa0 + n
        mun = (self.kappa0 * self.mu0 + n * xbar) / kn
        an = self.alpha0 + n / 2
        bn = self.beta0 + 0.5 * ss + self.kappa0 * n * (xbar - self.mu0) ** 2 / (2 * kn)
        tau = self.rng.gamma(an, 1.0 / bn)
        return self._argmax(self.rng.normal(mun, 1.0 / np.sqrt(kn * tau)))


class SuccessiveHalving(Bandit):
    name = "successive_halving"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.warmup = 0
        self.alive = list(range(self.k))
        self.rounds = max(1, math.ceil(math.log2(self.k)))
        self._plan()

    def _plan(self):
        per = max(1, self.budget // (len(self.alive) * self.rounds))
        self.queue = [a for _ in range(per) for a in self.alive]
        self.round_n = np.zeros(self.k, int)
        self.round_s = np.zeros(self.k)

    def select(self):
        if not self.queue:
            if len(self.alive) > 1:
                m = self.round_s[self.alive] / np.maximum(self.round_n[self.alive], 1)
                keep = max(1, math.ceil(len(self.alive) / 2))
                self.alive = [self.alive[i] for i in np.argsort(-m, kind="stable")[:keep]]
            self._plan()
        return self.queue.pop(0)

    def update(self, arm, reward):
        super().update(arm, reward)
        self.round_n[arm] += 1
        self.round_s[arm] += reward

    def recommend(self):
        if len(self.alive) == 1:
            return self.alive[0]
        if self.round_n.sum() == 0:
            return self._argmax(np.where(np.isin(np.arange(self.k), self.alive), self.means, -np.inf))
        m = np.where(self.round_n > 0, self.round_s / np.maximum(self.round_n, 1), -np.inf)
        m[[i for i in range(self.k) if i not in self.alive]] = -np.inf
        return self._argmax(m)


ALGORITHMS = {c.name: c for c in (Uniform, EpsilonGreedy, UCB1, UCB1Tuned, Thompson, SuccessiveHalving)}


def make_bandit(name: str, n_arms: int, budget: int, seed: int) -> Bandit:
    if name not in ALGORITHMS:
        raise ValueError(f"Unknown bandit '{name}'. Choose from {list(ALGORITHMS)}")
    return ALGORITHMS[name](n_arms, budget=budget, rng=np.random.default_rng(seed))
