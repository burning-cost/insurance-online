# insurance-online

Multi-armed bandit algorithms for insurance conversion optimisation.

## The problem

You want to find the optimal price for a motor policy. You have 7 candidate price
levels (85% to 115% of the technical rate). Standard approach: run an A/B test,
allocate equal volume to all 7 prices for 3 months, then commit to the winner.

The problem: 3 months of equal allocation means allocating ~14% of volume to each
price — including the prices that turn out to be 20% below optimal. On a book writing
2,000 quotes/day, that's 60,000 quotes priced suboptimally. At £750 expected premium,
the regret is material.

Bandits explore and exploit simultaneously. Thompson Sampling starts steering volume
toward promising prices after the first few hundred quotes. By the time you'd commit
an A/B test, the bandit has already been exploiting the near-optimal price for weeks.

## The solution

`ThompsonBandit` maintains a Beta(alpha, beta) posterior over conversion probability
for each price arm. At each quote, it samples from each posterior and selects the
price with the highest expected revenue (price * sampled_conversion_rate). The
Bayesian posterior update is a single line: alpha += convert, beta += (1 - convert).

Also included: `UCB1Bandit` (Upper Confidence Bound), `EpsilonGreedyBandit`,
and `ABTestBaseline` for comparison.

## Installation

```bash
pip install git+https://github.com/burning-cost/insurance-online.git
```

## Usage

```python
from insurance_online import ThompsonBandit, PricingSimulation

bandit = ThompsonBandit(
    price_levels=[0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15],
    reward="revenue",   # optimise price * P(convert), not just conversion rate
)

# In your quote flow:
for quote in incoming_quotes:
    price_level = bandit.select_price(base_premium=quote.technical_premium)
    actual_price = price_level * quote.technical_premium
    # ... present price to customer ...
    bandit.update(price_level, converted=customer_bought, base_premium=quote.technical_premium)

print(bandit.summary())
print(f"Best price: {bandit.best_price():.2f}x technical rate")
```

## When to use it

Use bandit pricing when:
- Quote volume is limited (specialist, commercial, fleet)
- You can't afford to allocate volume to bad prices for a long exploration period
- The optimal price may drift over time (inflation, competition)
- You want continuous learning rather than a fixed experiment

Use flat A/B testing when:
- You need a formal statistical test with clear significance thresholds
- Equal allocation is required for regulatory compliance
- Volume is high enough that exploration waste is negligible

## Performance

Benchmarked against flat A/B test (50% exploration, 50% exploitation) on a simulated
stream of UK motor quotes (10,000 rounds, elasticity=3.0, 7 price arms).
See `notebooks/benchmark_online.py` for full methodology.

- **Thompson Sampling reduces cumulative regret by 20-40%** vs flat A/B test over
  10,000 quotes. UCB1 reduces regret by 10-25%. Epsilon-greedy is marginal.
- **Bandit advantage is largest at high price elasticity** (competitive PCW markets
  where conversion drops sharply with price) and with more price arms.
- **Convergence speed**: Thompson identifies the best arm (best price) roughly 2-3x
  faster than the A/B test commits, measured as rounds until stable arm selection.
- **The revenue advantage** compounds: bandits also generate more total revenue than
  A/B tests because they exploit the near-optimal price earlier.
- **Limitation**: the simulation uses a stationary demand model. Real demand is
  non-stationary (seasonality, competition). Use a sliding window update for drift.

## References

- Thompson, W. R. (1933). On the likelihood that one unknown probability exceeds another. *Biometrika*, 25(3/4).
- Auer, P., et al. (2002). Finite-time analysis of the multiarmed bandit problem. *Machine Learning*, 47(2-3).
- Russo, D., et al. (2018). A tutorial on Thompson Sampling. *Found. Trends Mach. Learn.*
