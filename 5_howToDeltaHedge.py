import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# ============================================================
# Black-Scholes functions
# ============================================================

def call_price(S, K, tau, r, sigma):

    if tau <= 0:  return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)


def call_delta(S, K, tau, r, sigma):

    if tau <= 0:  return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))

    return norm.cdf(d1)


# ============================================================
# Parameters
# ============================================================

S0 = 100.0
K = 100.0
T = 1.0

r = 0.03

sigma_actual = 0.20
sigma_implied = 0.25

N = 252
dt = T / N

np.random.seed(4)


# ============================================================
# Simulate stock using ACTUAL volatility
# ============================================================

times = np.linspace(0, T, N + 1)

S = np.zeros(N + 1)
S[0] = S0

for i in range(N):
    Z = np.random.normal()
    S[i + 1] = S[i] * np.exp( (r - 0.5 * sigma_actual**2) * dt + sigma_actual * np.sqrt(dt) * Z )

# ============================================================
# Initial option price
#
# We SELL one call.
# The market price is determined using implied volatility.
# ============================================================

V0 = call_price(S0, K, T, r, sigma_implied)

print(f"Initial call price = {V0:.4f}")

# ============================================================
# Arrays
# ============================================================

delta          = np.zeros(N + 1)
cash           = np.zeros(N + 1)
stock_position = np.zeros(N + 1)
portfolio      = np.zeros(N + 1)
option_value   = np.zeros(N + 1)


# ============================================================
# Initial hedge
# ============================================================

delta[0] = call_delta(S0, K, T, r, sigma_implied)

# We are SHORT one call.
# To delta hedge a short call, buy Delta shares.

stock_position[0] = delta[0]

# Initial cash:
#
# receive V0 for selling call
# spend Delta*S buying shares

cash[0] = V0 - delta[0] * S0

option_value[0] = V0

portfolio[0] = (    stock_position[0] * S0    + cash[0]    - option_value[0])

# ============================================================
# Dynamic delta hedging
# ============================================================

for i in range(1, N + 1):

    tau = T - times[i]

    # --------------------------------------------------------
    # Cash grows at risk-free rate
    # --------------------------------------------------------

    cash[i] = cash[i - 1] * np.exp(r * dt)

    # --------------------------------------------------------
    # Previous stock position
    # --------------------------------------------------------

    stock_position[i] = stock_position[i - 1]

    # --------------------------------------------------------
    # Current option value
    # --------------------------------------------------------

    option_value[i] = call_price( S[i], K, tau, r, sigma_implied    )

    # --------------------------------------------------------
    # Compute new delta
    # --------------------------------------------------------

    new_delta = call_delta( S[i], K, tau, r, sigma_implied    )

    # --------------------------------------------------------
    # Rebalance stock hedge
    # --------------------------------------------------------

    shares_to_buy = new_delta - stock_position[i]

    cash[i] -= shares_to_buy * S[i]

    stock_position[i] = new_delta
    delta[i] = new_delta

    # --------------------------------------------------------
    # Value of total hedged position
    #
    # long Delta shares
    # + cash
    # - short call
    # --------------------------------------------------------

    portfolio[i] = (stock_position[i]*S[i] + cash[i] - option_value[i])


# ============================================================
# Settle option at expiration
# ============================================================

payoff = max(S[-1] - K, 0.0)

final_hedge_value = stock_position[-1] * S[-1] + cash[-1]

final_pnl = final_hedge_value - payoff


print()
print("Final results")
print("-----------------------------")
print(f"Final stock price       = {S[-1]:.4f}")
print(f"Call payoff             = {payoff:.4f}")
print(f"Hedge portfolio value   = {final_hedge_value:.4f}")
print(f"Final hedging P&L       = {final_pnl:.4f}")


# ============================================================
# Plot 1: Stock path
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(times, S, label="Stock price")
plt.axhline(K, linestyle="--", label="Strike")
plt.xlabel("Time")
plt.ylabel("Stock price")
plt.title(fr"Stock path: actual volatility = {sigma_actual:.0%}")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# Plot 2: Delta
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(times, delta)
plt.xlabel("Time")
plt.ylabel(r"$\Delta$")
plt.title(fr"Delta hedge using implied volatility = {sigma_implied:.0%}")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# Plot 3: Stock position
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(times, stock_position)
plt.xlabel("Time")
plt.ylabel("Number of shares")
plt.title("Stock position required for delta hedge")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# Plot 4: Hedging P&L
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(times, portfolio)
plt.axhline(0, linestyle="--")
plt.xlabel("Time")
plt.ylabel("Hedged portfolio value")
plt.title("Value of short-call + delta-hedge portfolio")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
