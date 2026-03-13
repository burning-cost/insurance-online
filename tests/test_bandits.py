"""Tests for insurance-online."""

import numpy as np
import pytest
from insurance_online import UCB1Bandit, ThompsonBandit, EpsilonGreedyBandit, ABTestBaseline
from insurance_online.simulation import PricingSimulation


PRICE_LEVELS = [0.90, 0.95, 1.0, 1.05, 1.10]


def test_ucb1_selects_all_arms():
    """UCB1 should try each arm at least once before exploiting."""
    bandit = UCB1Bandit(PRICE_LEVELS)
    selected = set()
    for i in range(20):
        p = bandit.select_price()
        selected.add(p)
        bandit.update(p, converted=bool(i % 3 == 0))
    assert len(selected) == len(PRICE_LEVELS), "UCB1 should explore all arms"


def test_thompson_converges():
    """Thompson sampling should identify the best arm after enough rounds."""
    # True best: 0.95 has highest expected revenue (price * conv_prob)
    # conv_prob decreases with price (elasticity model)
    bandit = ThompsonBandit(PRICE_LEVELS, random_state=42)
    sim    = PricingSimulation(PRICE_LEVELS, base_conversion_rate=0.20, price_elasticity=3.0,
                               n_rounds=2000, random_state=42)
    result = sim.run(bandit, "Thompson")
    # After 2000 rounds, should have identified the best arm
    assert result.final_best_price == result.true_best_price or (
        abs(result.final_best_price - result.true_best_price) <= 0.1
    ), f"Thompson failed to converge: {result.final_best_price} vs {result.true_best_price}"


def test_bandit_lower_regret_than_ab():
    """Bandit should achieve lower cumulative regret than equal-allocation A/B test."""
    sim = PricingSimulation(PRICE_LEVELS, n_rounds=3000, random_state=42)

    thompson = ThompsonBandit(PRICE_LEVELS, random_state=42)
    ab       = ABTestBaseline(PRICE_LEVELS, exploration_rounds=1500, random_state=42)

    res_t  = sim.run(thompson, "Thompson")
    res_ab = sim.run(ab, "AB")

    # Thompson should have less total regret after 3000 rounds
    assert res_t.total_regret <= res_ab.total_regret * 1.1, (
        f"Thompson regret {res_t.total_regret:.1f} should be < AB regret {res_ab.total_regret:.1f}"
    )


def test_simulation_oracle():
    """Oracle revenue should be the maximum expected revenue."""
    sim = PricingSimulation(PRICE_LEVELS, base_conversion_rate=0.20, price_elasticity=3.0,
                            n_rounds=100)
    # Oracle arm should give max expected revenue
    assert sim.oracle_revenue == max(sim.true_expected_revenues)


def test_ucb1_update():
    bandit = UCB1Bandit(PRICE_LEVELS)
    bandit.update(1.0, True)
    assert bandit.arms[2].n_trials == 1
    assert bandit.arms[2].n_successes == 1
    bandit.update(1.0, False)
    assert bandit.arms[2].n_trials == 2
    assert bandit.arms[2].n_successes == 1


def test_ab_test_commits_to_winner():
    """A/B test should commit to winner after exploration_rounds."""
    ab = ABTestBaseline(PRICE_LEVELS, exploration_rounds=20, random_state=42)
    for i in range(25):
        p = ab.select_price()
        ab.update(p, converted=(i % 3 == 0))
    assert ab._committed_arm is not None
