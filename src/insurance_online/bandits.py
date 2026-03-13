"""
Multi-armed bandit implementations for insurance price optimisation.

Each bandit maintains state across quotes and selects the price arm that balances
exploration (trying less-tested prices) with exploitation (favouring known-good prices).

The reward signal is expected revenue: price * P(convert | price, customer).
This is different from pure click-through rate optimisation — we care about
revenue, not just conversion rate. A 10% reduction in price with 30% higher
conversion is a net positive; a 5% reduction with 2% higher conversion is not.

Bandit arms correspond to price levels (e.g. [0.90, 0.95, 1.00, 1.05, 1.10] as
multiples of the base technical premium). Each quote selects an arm, observes
convert/no-convert, and updates the arm statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
import numpy as np


@dataclass
class ArmStats:
    """Statistics for one price arm."""
    price_level: float   # price as multiple of base premium
    n_trials: int = 0
    n_successes: int = 0
    total_revenue: float = 0.0
    # Thompson: Beta distribution parameters
    alpha: float = 1.0   # successes + prior
    beta: float = 1.0    # failures + prior

    @property
    def conversion_rate(self) -> float:
        if self.n_trials == 0:
            return 0.0
        return self.n_successes / self.n_trials

    @property
    def mean_revenue_per_quote(self) -> float:
        if self.n_trials == 0:
            return 0.0
        return self.total_revenue / self.n_trials

    def update(self, price: float, converted: bool, base_premium: float = 1.0):
        self.n_trials += 1
        revenue = price * base_premium if converted else 0.0
        self.total_revenue += revenue
        if converted:
            self.n_successes += 1
            self.alpha += 1
        else:
            self.beta += 1


class UCB1Bandit:
    """Upper Confidence Bound 1 (UCB1) bandit for price optimisation.

    Selects the arm with the highest UCB:
        UCB_i(t) = mu_i + sqrt(2 * ln(t) / n_i)
    where mu_i is the empirical mean reward for arm i, t is total rounds,
    and n_i is the number of times arm i has been pulled.

    Parameters
    ----------
    price_levels : sequence of float
        Price levels as multiples of the base premium.
    exploration_coef : float
        Coefficient on the exploration term. Default 2.0 (standard UCB1).
    reward : 'revenue' or 'conversion'
        What to optimise. 'revenue' = price * convert; 'conversion' = binary convert.

    References
    ----------
    Auer, P., et al. (2002). Finite-time analysis of the multiarmed bandit problem.
    *Machine Learning*, 47(2-3), 235-256.
    """

    def __init__(
        self,
        price_levels: Sequence[float],
        exploration_coef: float = 2.0,
        reward: str = "revenue",
    ):
        self.price_levels   = list(price_levels)
        self.exploration_coef = exploration_coef
        self.reward         = reward
        self.arms           = [ArmStats(p) for p in price_levels]
        self.t              = 0   # total rounds
        self._history: list[dict] = []

    def select_price(self, base_premium: float = 1.0) -> float:
        """Select the price level for this quote.

        Returns the price level (as float, multiply by base_premium for actual price).
        """
        # Initialise: try each arm at least once
        for i, arm in enumerate(self.arms):
            if arm.n_trials == 0:
                return self.price_levels[i]

        # UCB1 selection
        t = max(self.t, 1)
        ucbs = []
        for arm in self.arms:
            if self.reward == "revenue":
                mu = arm.mean_revenue_per_quote / max(base_premium, 1e-8)
            else:
                mu = arm.conversion_rate
            ucb = mu + np.sqrt(self.exploration_coef * np.log(t) / arm.n_trials)
            ucbs.append(ucb)

        return self.price_levels[int(np.argmax(ucbs))]

    def update(self, price_level: float, converted: bool, base_premium: float = 1.0):
        """Update arm statistics after observing outcome.

        Parameters
        ----------
        price_level : float
            The price level that was used (must match one of self.price_levels).
        converted : bool
            Whether the customer converted (bought the policy).
        base_premium : float
            Base technical premium for revenue calculation.
        """
        idx = self.price_levels.index(price_level)
        self.arms[idx].update(price_level, converted, base_premium)
        self.t += 1
        self._history.append({
            "t": self.t,
            "price_level": price_level,
            "converted": converted,
            "revenue": price_level * base_premium if converted else 0.0,
        })

    def best_price(self) -> float:
        """Return the current best estimated price level (highest conversion rate)."""
        if all(a.n_trials == 0 for a in self.arms):
            return self.price_levels[len(self.price_levels) // 2]
        rates = [a.conversion_rate for a in self.arms]
        return self.price_levels[int(np.argmax(rates))]

    def summary(self) -> str:
        lines = ["UCB1 Bandit Summary", "-" * 40]
        for arm in self.arms:
            lines.append(
                f"  Price {arm.price_level:.2f}: "
                f"n={arm.n_trials:>5}, "
                f"conv={arm.conversion_rate:.3f}, "
                f"rev/quote={arm.mean_revenue_per_quote:.3f}"
            )
        lines.append(f"  Total rounds: {self.t}")
        lines.append(f"  Best price:   {self.best_price():.2f}")
        return "\n".join(lines)


class ThompsonBandit:
    """Thompson Sampling bandit for price optimisation.

    Maintains a Beta(alpha_i, beta_i) posterior over conversion rate for each arm.
    At each round, samples theta_i ~ Beta(alpha_i, beta_i) for each arm and selects
    the arm with the highest expected revenue: price_i * theta_i.

    Thompson Sampling typically outperforms UCB1 in terms of cumulative regret
    on binary reward problems. The Bayesian prior (alpha=1, beta=1) is uniform —
    equivalent to "I have no prior information about conversion rates".

    References
    ----------
    Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds another.
    *Biometrika*, 25(3/4), 285-294.

    Russo, D., et al. (2018). A tutorial on Thompson sampling. *Found. Trends Mach. Learn.*
    """

    def __init__(
        self,
        price_levels: Sequence[float],
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        reward: str = "revenue",
        random_state: int | None = None,
    ):
        self.price_levels  = list(price_levels)
        self.prior_alpha   = prior_alpha
        self.prior_beta    = prior_beta
        self.reward        = reward
        self.arms          = [ArmStats(p, alpha=prior_alpha, beta=prior_beta)
                              for p in price_levels]
        self.t             = 0
        self._rng          = np.random.default_rng(random_state)
        self._history: list[dict] = []

    def select_price(self, base_premium: float = 1.0) -> float:
        """Sample from posterior and select best arm."""
        samples = []
        for arm in self.arms:
            theta = self._rng.beta(arm.alpha, arm.beta)
            if self.reward == "revenue":
                expected_revenue = arm.price_level * base_premium * theta
            else:
                expected_revenue = theta
            samples.append(expected_revenue)
        return self.price_levels[int(np.argmax(samples))]

    def update(self, price_level: float, converted: bool, base_premium: float = 1.0):
        """Update Beta posterior for the selected arm."""
        idx = self.price_levels.index(price_level)
        self.arms[idx].update(price_level, converted, base_premium)
        self.t += 1
        self._history.append({
            "t": self.t,
            "price_level": price_level,
            "converted": converted,
            "revenue": price_level * base_premium if converted else 0.0,
        })

    def best_price(self) -> float:
        """Return the current MAP estimate of the best price."""
        if all(a.n_trials == 0 for a in self.arms):
            return self.price_levels[len(self.price_levels) // 2]
        # Best arm by posterior mean conversion rate
        mean_rates = [(a.alpha - 1) / max(a.alpha + a.beta - 2, 1e-8) for a in self.arms]
        return self.price_levels[int(np.argmax(mean_rates))]

    def summary(self) -> str:
        lines = ["Thompson Sampling Bandit Summary", "-" * 40]
        for arm in self.arms:
            post_mean = arm.alpha / (arm.alpha + arm.beta)
            lines.append(
                f"  Price {arm.price_level:.2f}: "
                f"n={arm.n_trials:>5}, "
                f"conv={arm.conversion_rate:.3f} "
                f"[post_mean={post_mean:.3f}], "
                f"rev/quote={arm.mean_revenue_per_quote:.3f}"
            )
        lines.append(f"  Total rounds: {self.t}")
        lines.append(f"  Best price:   {self.best_price():.2f}")
        return "\n".join(lines)


class EpsilonGreedyBandit:
    """Epsilon-greedy bandit: with probability epsilon explore uniformly, else exploit.

    Simplest bandit algorithm. Included for completeness.

    Parameters
    ----------
    price_levels : sequence of float
    epsilon : float
        Exploration probability. 0 = pure greedy; 1 = pure random.
    decay : float
        Multiplicative decay of epsilon per round. 1.0 = no decay.
    """

    def __init__(
        self,
        price_levels: Sequence[float],
        epsilon: float = 0.1,
        decay: float = 1.0,
        random_state: int | None = None,
    ):
        self.price_levels = list(price_levels)
        self.epsilon      = epsilon
        self.decay        = decay
        self.arms         = [ArmStats(p) for p in price_levels]
        self.t            = 0
        self._rng         = np.random.default_rng(random_state)
        self._current_eps = epsilon
        self._history: list[dict] = []

    def select_price(self, base_premium: float = 1.0) -> float:
        """Epsilon-greedy selection."""
        if self._rng.random() < self._current_eps:
            return self._rng.choice(self.price_levels)
        if all(a.n_trials == 0 for a in self.arms):
            return self._rng.choice(self.price_levels)
        rates = [a.mean_revenue_per_quote / max(base_premium, 1e-8) for a in self.arms]
        return self.price_levels[int(np.argmax(rates))]

    def update(self, price_level: float, converted: bool, base_premium: float = 1.0):
        idx = self.price_levels.index(price_level)
        self.arms[idx].update(price_level, converted, base_premium)
        self.t += 1
        self._current_eps *= self.decay
        self._history.append({
            "t": self.t,
            "price_level": price_level,
            "converted": converted,
            "revenue": price_level * base_premium if converted else 0.0,
        })

    def best_price(self) -> float:
        if all(a.n_trials == 0 for a in self.arms):
            return self.price_levels[len(self.price_levels) // 2]
        rates = [a.conversion_rate for a in self.arms]
        return self.price_levels[int(np.argmax(rates))]
