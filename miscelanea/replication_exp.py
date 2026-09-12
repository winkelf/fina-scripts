import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ============================================================
# Parameters
# ============================================================

S0 = 100.0
K = 100.0
T = 1.0

mu = 0.10       # actual stock drift
r = 0.05        # risk-free rate
sigma = 0.20

N_steps = 252
dt = T / N_steps

rng = np.random.default_rng(42)

# ============================================================
# Black-Scholes functions
# ============================================================

def d1(S, tau):
    return (
        np.log(S / K)
        + (r + 0.5 * sigma**2) * tau
    ) / (sigma * np.sqrt(tau))


def d2(S, tau):
    return d1(S, tau) - sigma * np.sqrt(tau)


def call_price(S, tau):

    if tau <= 0:
        return max(S - K, 0.0)

    D1 = d1(S, tau)
    D2 = d2(S, tau)

    return (
        S * norm.cdf(D1)
        - K * np.exp(-r * tau) * norm.cdf(D2)
    )


def call_delta(S, tau):

    if tau <= 0:
        return 1.0 if S > K else 0.0

    return norm.cdf(d1(S, tau))


# ============================================================
# Time grid
# ============================================================

times = np.linspace(0, T, N_steps + 1)

# ============================================================
# Simulate stock path
# ============================================================

S = np.zeros(N_steps + 1)
S[0] = S0

for i in range(N_steps):

    Z = rng.normal()

    S[i+1] = S[i] * np.exp(
        (mu - 0.5 * sigma**2) * dt
        + sigma * np.sqrt(dt) * Z
    )


# ============================================================
# Arrays for option and hedge
# ============================================================

V = np.zeros(N_steps + 1)

Delta = np.zeros(N_steps + 1)

cash = np.zeros(N_steps + 1)

portfolio = np.zeros(N_steps + 1)


# ============================================================
# Initial condition
# ============================================================

tau = T

V[0] = call_price(S[0], tau)

Delta[0] = call_delta(S[0], tau)

# Replicating portfolio:
#
# V = Delta*S + cash
#
# therefore:
#
# cash = V - Delta*S

cash[0] = V[0] - Delta[0] * S[0]

portfolio[0] = Delta[0] * S[0] + cash[0]


print("INITIAL VALUES")
print("---------------------------")
print("Stock price:     ", S[0])
print("Option price:    ", V[0])
print("Delta:           ", Delta[0])
print("Stock position:  ", Delta[0] * S[0])
print("Cash position:   ", cash[0])
print("Portfolio value: ", portfolio[0])


# ============================================================
# Dynamic delta hedge
# ============================================================

for i in range(N_steps):

    # --------------------------------------------------------
    # 1. Move forward in time
    # --------------------------------------------------------

    tau_next = T - times[i+1]


    # --------------------------------------------------------
    # 2. Cash earns the risk-free rate
    # --------------------------------------------------------

    cash_before_rebalance = cash[i] * np.exp(r * dt)


    # --------------------------------------------------------
    # 3. Calculate option price and new delta
    # --------------------------------------------------------

    if i + 1 < N_steps:

        V[i+1] = call_price(
            S[i+1],
            tau_next
        )

        Delta[i+1] = call_delta(
            S[i+1],
            tau_next
        )

    else:

        # At maturity
        V[i+1] = max(S[i+1] - K, 0.0)

        Delta[i+1] = (
            1.0 if S[i+1] > K else 0.0
        )


    # --------------------------------------------------------
    # 4. Rebalance stock position
    # --------------------------------------------------------

    delta_change = Delta[i+1] - Delta[i]

    # If delta increases, we buy stock.
    # Money leaves the cash account.
    #
    # If delta decreases, we sell stock.
    # Money enters the cash account.

    cash[i+1] = (
        cash_before_rebalance
        - delta_change * S[i+1]
    )


    # --------------------------------------------------------
    # 5. Total hedge portfolio
    # --------------------------------------------------------

    portfolio[i+1] = (
        Delta[i+1] * S[i+1]
        + cash[i+1]
    )


# ============================================================
# Final result
# ============================================================

payoff = max(S[-1] - K, 0.0)

hedging_error = portfolio[-1] - payoff


print()
print("FINAL VALUES")
print("---------------------------")
print("Final stock price: ", S[-1])
print("Option payoff:     ", payoff)
print("Portfolio value:   ", portfolio[-1])
print("Hedging error:     ", hedging_error)


# ============================================================
# Plot 1: stock path
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(times, S)

plt.axhline(
    K,
    linestyle="--",
    label="Strike"
)

plt.xlabel("Time [years]")
plt.ylabel("Stock price")
plt.title("Simulated stock path")

plt.legend()
plt.grid()

plt.show()


# ============================================================
# Plot 2: Delta
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(times, Delta)

plt.xlabel("Time [years]")
plt.ylabel("Delta")
plt.title("Dynamic hedge: Delta")

plt.grid()

plt.show()


# ============================================================
# Plot 3:
# Black-Scholes price vs replicating portfolio
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    times,
    V,
    label="Black-Scholes option value"
)

plt.plot(
    times,
    portfolio,
    linestyle="--",
    label="Replicating portfolio"
)

plt.xlabel("Time [years]")
plt.ylabel("Value")
plt.title("Option vs replicating portfolio")

plt.legend()
plt.grid()

plt.show()


# ============================================================
# Plot 4: replication error through time
# ============================================================

error = portfolio - V

plt.figure(figsize=(9, 5))

plt.plot(times, error)

plt.axhline(0)

plt.xlabel("Time [years]")
plt.ylabel("Portfolio - option")
plt.title("Replication error")

plt.grid()

plt.show()
