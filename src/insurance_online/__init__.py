"""
insurance-online: Bandit pricing algorithms for insurance conversion optimisation.

Insurance pricing is a sequential decision problem: at each new quote, you choose
a price, observe whether the customer converted, and update your pricing model.
Standard A/B testing requires a fixed allocation before you start and wastes
quotes on suboptimal prices. Multi-armed bandits learn and exploit simultaneously.

Key classes:
    UCB1Bandit          — Upper Confidence Bound 1 bandit (frequentist)
    ThompsonBandit      — Thompson Sampling bandit (Bayesian Beta-Binomial)
    EpsilonGreedyBandit — Epsilon-greedy bandit
    ABTestBaseline      — Flat A/B test with fixed equal allocation
    BanditResult        — Result container with regret and conversion metrics

Typical usage::

    from insurance_online import ThompsonBandit

    bandit = ThompsonBandit(price_levels=[0.9, 0.95, 1.0, 1.05, 1.10])
    for quote in quotes:
        price = bandit.select_price(quote.features)
        # customer accepts or declines
        bandit.update(price, converted=customer_decision)

    print(bandit.summary())
"""

from insurance_online.bandits import UCB1Bandit, ThompsonBandit, EpsilonGreedyBandit
from insurance_online.ab_test import ABTestBaseline
from insurance_online.simulation import PricingSimulation, SimulationResult

__version__ = "0.1.0"

__all__ = [
    "UCB1Bandit",
    "ThompsonBandit",
    "EpsilonGreedyBandit",
    "ABTestBaseline",
    "PricingSimulation",
    "SimulationResult",
    "__version__",
]
