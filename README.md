# fina-scripts

A collection of small Python scripts exploring option pricing, stochastic processes, and Monte Carlo methods in quantitative finance, using `numpy`, `scipy`, and `matplotlib`.

## Main scripts

- **`0_logNormalRandomWalk.py`** — Simulates multiple discrete-time stock price paths using a log-normal (geometric Brownian motion-like) random walk and plots them.

- **`1_bs.py`** — Computes the Black-Scholes price, Delta, Gamma, and Theta for a European call option, verifies that the individual terms satisfy the Black-Scholes PDE, and plots the option price and PDE contributions across a range of stock prices.

- **`2_deltaHedge.py`** — Simulates a single geometric Brownian motion stock path and builds a Delta-hedged option portfolio (`Pi = V - Delta * S`), rebalancing Delta at every step. Compares the P&L volatility of the hedged vs. unhedged position to show how Delta hedging removes first-order risk.

- **`3_american_vs_european.py`** — Compares European call values against the immediate-exercise payoff `max(S-K, 0)`, with and without dividends, to illustrate when the European pricing formula would be invalid as an American option price (i.e. where early exercise could be optimal).

- **`4_pdfs.py`** — Explores the forward and backward Kolmogorov equations for geometric Brownian motion: Monte Carlo vs. analytic transition densities (forward), and Monte Carlo vs. analytic solutions for `P(S_T > K | S_t = S)` evolving backward toward its terminal condition.

## `miscelanea/`

Exploratory / scratch scripts, less polished than the main sequence above:

- **`randomWalk.py`** — Simple multiplicative coin-toss random walk (no market data), saves the plot to `test.pdf`.
- **`niceRandomWalk.py`** — Variant of the log-normal random walk with several different drift (`mu`) values plotted together.
- **`replication_exp.py`** — Black-Scholes helper functions plus an experiment in replicating an option payoff.
- **`plot_perez_c.py`** — Loads historical price data for Pérez Companc from `perez_companc_extracted.txt` (not included in this repo) and plots period returns.
- **`openbb/1D_Tale.py`** — Fetches AAPL daily price data via the OpenBB terminal SDK and plots the closing price.

## Requirements

```
numpy
scipy
matplotlib
```

`miscelanea/openbb/1D_Tale.py` additionally requires the `openbb_terminal` package.

Install with:

```bash
pip install numpy scipy matplotlib
```
