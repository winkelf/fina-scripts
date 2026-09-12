import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ============================================================
# Parameters
# ============================================================

S0 = 100.0
K = 100.0
T = 1.0

mu = 0.10        # actual stock drift
r = 0.05         # risk-free rate
sigma = 0.20

N_steps = 252
dt = T / N_steps

rng = np.random.default_rng(42)

# ============================================================
# Black-Scholes functions
# ============================================================

def d1(S, tau):
    return ( np.log(S / K) + (r + 0.5 * sigma**2) * tau ) / (sigma * np.sqrt(tau))


def d2(S, tau):
    return d1(S, tau) - sigma * np.sqrt(tau)


def black_scholes_call_price(S, tau):
    if tau <= 0:
        return max(S - K, 0.0)

    D1 = d1(S, tau)
    D2 = d2(S, tau)

    return ( S * norm.cdf(D1)  - K * np.exp(-r * tau) * norm.cdf(D2)  )


def delta(S, tau):
    if tau <= 0:
        return 1.0 if S > K else 0.0

    return norm.cdf(d1(S, tau))


# ============================================================
# Simulate one stock path
# ============================================================

times = np.linspace(0, T, N_steps + 1)

S = np.zeros(N_steps + 1)
S[0] = S0

for i in range(N_steps):

    Z = rng.normal()

    # Exact solution of geometric Brownian motion over one step
    S[i+1] = S[i] * np.exp( (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z )


# ============================================================
# Construct delta-hedged portfolio
#
# Portfolio:
#
#     Pi = V - Delta S
#
# We rebalance Delta at every time step.
# ============================================================

V     = np.zeros(N_steps + 1)
Delta = np.zeros(N_steps + 1)
Pi    = np.zeros(N_steps + 1)

for i in range(N_steps):

    tau      = T - times[i]
    V[i]     = black_scholes_call_price(S[i], tau)
    Delta[i] = delta(S[i], tau)
    Pi[i]    = V[i] - Delta[i] * S[i]


# final option payoff
V[-1]     = max(S[-1] - K, 0.0)
Delta[-1] = 1.0 if S[-1] > K else 0.0
Pi[-1]    = V[-1] - Delta[-1] * S[-1]


# ============================================================
# Look directly at changes
# ============================================================

dS = np.diff(S)
dV = np.diff(V)

# Delta from beginning of each interval
dPi_hedged = dV - Delta[:-1] * dS

# Compare against an unhedged option change
dPi_unhedged = dV


# ============================================================
# Plot stock path
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(times, S)
plt.axhline(K, linestyle="--")

plt.xlabel("Time [years]")
plt.ylabel("Stock price")
plt.title("Simulated stock path")

plt.grid()
plt.show()

# ============================================================
# Plot option value and delta
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(times, V, label="Option value V")
plt.plot(times, Delta, label="Delta")

plt.xlabel("Time [years]")
plt.title("Option value and Delta")

plt.legend()
plt.grid()
plt.show()


# ============================================================
# Compare random fluctuations
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(times[:-1], dPi_unhedged, label=r"$dV$")

plt.plot(times[:-1], dPi_hedged, label=r"$dV-\Delta dS$")

plt.axhline(0)

plt.xlabel("Time [years]")
plt.ylabel("Change over one step")
plt.title("Effect of Delta Hedging")

plt.legend()
plt.grid()
plt.show()


# ============================================================
# Histograms: unhedged vs hedged
# ============================================================

plt.figure(figsize=(9, 5))

plt.hist(dPi_unhedged, bins=40, alpha=0.6, label="Unhedged option change")

plt.hist(dPi_hedged, bins=40, alpha=0.6, label="Delta-hedged change")

plt.xlabel("One-step P&L")
plt.ylabel("Counts")
plt.title("Delta hedging removes most first-order randomness")

plt.legend()
plt.grid()
plt.show()


# ============================================================
# Numerical comparison
# ============================================================

print("Standard deviation of dV:")
print(np.std(dPi_unhedged))

print()

print("Standard deviation of dV - Delta*dS:")
print(np.std(dPi_hedged))

print()

print("Ratio:")
print(np.std(dPi_hedged) / np.std(dPi_unhedged))
