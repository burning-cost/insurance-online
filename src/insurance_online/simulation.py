"""
Pricing simulation environment for evaluating bandit vs A/B test performance.

The environment generates a stream of quotes with known true conversion probabilities.
Each algorithm selects a price, observes convert/no-convert, and we measure:
- Cumulative regret: revenue lost vs oracle (always choosing optimal price)
- Convergence speed: rounds until best arm is identified
- Total revenue: sum of price * convert over all rounds
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class SimulationResult:
    """Results from a pricing simulation.

    Attributes
    ----------
    method_name : str
    n_rounds : int
    price_levels : list[float]
    cumulative_regret : np.ndarray
        Cumulative regret at each round t. Regret = oracle revenue - actual revenue.
    cumulative_revenue : np.ndarray
        Cumulative revenue at each round t.
    arm_selections : np.ndarray
        Which arm was selected at each round.
    conversions : np.ndarray
        Whether the customer converted at each round.
    final_best_price : float
        The algorithm's best price estimate after all rounds.
    true_best_price : float
        The oracle best price.
    total_regret : float
        Total regret over all rounds.
    total_revenue : float
    """

    method_name: str
    n_rounds: int
    price_levels: list[float]
    cumulative_regret: np.ndarray = field(repr=False)
    cumulative_revenue: np.ndarray = field(repr=False)
    arm_selections: np.ndarray = field(repr=False)
    conversions: np.ndarray = field(repr=False)
    final_best_price: float = 0.0
    true_best_price: float = 0.0
    total_regret: float = 0.0
    total_revenue: float = 0.0

    def summary(self) -> str:
        lines = [
            f"{self.method_name} Simulation Results",
            "-" * 45,
            f"  N rounds:         {self.n_rounds:,}",
            f"  Total revenue:    {self.total_revenue:,.2f}",
            f"  Total regret:     {self.total_regret:,.2f}",
            f"  Mean regret/rnd:  {self.total_regret/max(self.n_rounds,1):.4f}",
            f"  True best price:  {self.true_best_price:.3f}",
            f"  Final best price: {self.final_best_price:.3f}",
            f"  Convergence:      {'Correct' if abs(self.final_best_price - self.true_best_price) < 0.01 else 'Wrong'}",
        ]
        return "\n".join(lines)


class PricingSimulation:
    """Simulate a stream of insurance quotes and evaluate bandit vs A/B test.

    The environment has a true conversion function:
        P(convert | price_level, base_rate) = base_rate * exp(-elasticity * (price_level - 1))

    This is a log-linear demand model — standard in insurance price elasticity research.

    Parameters
    ----------
    price_levels : sequence of float
    base_conversion_rate : float
        Conversion rate at price_level = 1.0. Typical UK motor: 0.15-0.30.
    price_elasticity : float
        Price sensitivity. Higher = more sensitive. Typical: 2-5.
    n_rounds : int
        Total simulation rounds.
    base_premium : float
        Mean base premium (for revenue calculation). Default 1000.
    noise_std : float
        Noise on base conversion rate (simulates customer heterogeneity).
    random_state : int

    Examples
    --------
    >>> from insurance_online import ThompsonBandit, PricingSimulation
    >>> bandit = ThompsonBandit(price_levels=[0.90, 0.95, 1.0, 1.05, 1.10])
    >>> sim = PricingSimulation(price_levels=[0.90, 0.95, 1.0, 1.05, 1.10])
    >>> result = sim.run(bandit, "Thompson")
    >>> print(result.summary())
    """

    def __init__(
        self,
        price_levels: Sequence[float],
        base_conversion_rate: float = 0.20,
        price_elasticity: float = 3.0,
        n_rounds: int = 5_000,
        base_premium: float = 1_000.0,
        noise_std: float = 0.05,
        random_state: int = 42,
    ):
        self.price_levels          = list(price_levels)
        self.base_conversion_rate  = base_conversion_rate
        self.price_elasticity      = price_elasticity
        self.n_rounds              = n_rounds
        self.base_premium          = base_premium
        self.noise_std             = noise_std
        self.random_state          = random_state

        # True conversion probability for each price level
        self.true_conv_probs = np.array([
            base_conversion_rate * np.exp(-price_elasticity * (p - 1.0))
            for p in price_levels
        ])
        self.true_conv_probs = np.clip(self.true_conv_probs, 0.0, 1.0)

        # Oracle: arm with highest expected revenue = price * conv_prob * base_premium
        self.true_expected_revenues = (
            np.array(price_levels) * self.true_conv_probs * base_premium
        )
        self.oracle_idx     = int(np.argmax(self.true_expected_revenues))
        self.oracle_revenue = float(self.true_expected_revenues[self.oracle_idx])
        self.true_best_price = price_levels[self.oracle_idx]

    def run(self, algorithm, method_name: str = "Algorithm") -> SimulationResult:
        """Run the simulation.

        Parameters
        ----------
        algorithm : bandit or A/B test object with select_price() and update()
        method_name : str

        Returns
        -------
        SimulationResult
        """
        rng = np.random.default_rng(self.random_state)

        cum_regret  = np.zeros(self.n_rounds)
        cum_revenue = np.zeros(self.n_rounds)
        arm_sel     = np.zeros(self.n_rounds, dtype=float)
        conversions = np.zeros(self.n_rounds, dtype=bool)

        total_regret  = 0.0
        total_revenue = 0.0

        for t in range(self.n_rounds):
            # Each quote has a slightly different base rate (customer heterogeneity)
            base_rate = self.base_conversion_rate + rng.normal(0, self.noise_std)
            base_prem = self.base_premium * (1 + rng.normal(0, 0.1))

            # Algorithm selects price
            price_level = algorithm.select_price(base_premium=base_prem)
            arm_idx     = self.price_levels.index(price_level)

            # True conversion probability for this customer at this price
            true_p = float(np.clip(
                base_rate * np.exp(-self.price_elasticity * (price_level - 1.0)),
                0.0, 1.0,
            ))
            converted = bool(rng.random() < true_p)

            # Revenue and regret
            revenue       = price_level * base_prem if converted else 0.0
            oracle_rev    = self.oracle_revenue  # approximate oracle revenue
            regret        = max(oracle_rev - revenue, 0.0)

            total_revenue += revenue
            total_regret  += regret

            cum_regret[t]  = total_regret
            cum_revenue[t] = total_revenue
            arm_sel[t]     = price_level
            conversions[t] = converted

            # Update algorithm
            algorithm.update(price_level, converted, base_premium=base_prem)

        return SimulationResult(
            method_name=method_name,
            n_rounds=self.n_rounds,
            price_levels=self.price_levels,
            cumulative_regret=cum_regret,
            cumulative_revenue=cum_revenue,
            arm_selections=arm_sel,
            conversions=conversions,
            final_best_price=algorithm.best_price(),
            true_best_price=self.true_best_price,
            total_regret=total_regret,
            total_revenue=total_revenue,
        )

    def oracle_revenue_rate(self) -> float:
        """Oracle expected revenue per round."""
        return self.oracle_revenue

    def summary(self) -> str:
        lines = [
            "Pricing Simulation Environment",
            "-" * 45,
            f"  Price levels:        {self.price_levels}",
            f"  Base conv. rate:     {self.base_conversion_rate:.3f}",
            f"  Price elasticity:    {self.price_elasticity:.1f}",
        ]
        lines.append("  True conv. probs:")
        for p, cp, er in zip(self.price_levels, self.true_conv_probs, self.true_expected_revenues):
            marker = " <-- ORACLE" if abs(er - self.oracle_revenue) < 0.01 else ""
            lines.append(f"    {p:.2f}: conv={cp:.4f}  exp_rev={er:.2f}{marker}")
        return "\n".join(lines)
