"""
A/B test baseline for price optimisation.

Flat equal allocation: split quotes evenly across all price arms for a fixed
exploration period, then commit to the winner.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
from insurance_online.bandits import ArmStats


class ABTestBaseline:
    """Flat A/B test with equal allocation then winner commit.

    Parameters
    ----------
    price_levels : sequence of float
    exploration_rounds : int
        Number of rounds to explore (equal allocation) before committing to winner.
        If None, explore forever (pure A/B test, never commit).
    random_state : int or None
    """

    def __init__(
        self,
        price_levels: Sequence[float],
        exploration_rounds: int | None = None,
        random_state: int | None = None,
    ):
        self.price_levels       = list(price_levels)
        self.exploration_rounds = exploration_rounds
        self.arms               = [ArmStats(p) for p in price_levels]
        self.t                  = 0
        self._committed_arm: int | None = None
        self._rng               = np.random.default_rng(random_state)
        self._history: list[dict] = []

    def select_price(self, base_premium: float = 1.0) -> float:
        """Select price arm. Equal allocation during exploration, best arm after."""
        if self._committed_arm is not None:
            return self.price_levels[self._committed_arm]

        # Round-robin equal allocation
        idx = self.t % len(self.price_levels)
        return self.price_levels[idx]

    def update(self, price_level: float, converted: bool, base_premium: float = 1.0):
        """Update arm stats and commit to winner if exploration is done."""
        idx = self.price_levels.index(price_level)
        self.arms[idx].update(price_level, converted, base_premium)
        self.t += 1
        self._history.append({
            "t": self.t,
            "price_level": price_level,
            "converted": converted,
            "revenue": price_level * base_premium if converted else 0.0,
        })

        # Check if we should commit
        if (self.exploration_rounds is not None
                and self.t >= self.exploration_rounds
                and self._committed_arm is None):
            rates = [a.conversion_rate for a in self.arms]
            self._committed_arm = int(np.argmax(rates))

    def best_price(self) -> float:
        if self._committed_arm is not None:
            return self.price_levels[self._committed_arm]
        if all(a.n_trials == 0 for a in self.arms):
            return self.price_levels[len(self.price_levels) // 2]
        rates = [a.conversion_rate for a in self.arms]
        return self.price_levels[int(np.argmax(rates))]
