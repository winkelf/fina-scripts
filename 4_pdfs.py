import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm, norm

# ============================================================
# Parameters
# ============================================================

S0 = 100.0
mu = 0.10
sigma = 0.20
T = 1.0
Nmc = 200_000
rng = np.random.default_rng(42)


# ============================================================
# Exact GBM evolution (geometric brownian motion)
#
# S(t+dt) = S(t) exp[
#
#     (mu - sigma^2/2) dt
#     + sigma sqrt(dt) Z
#
# ]
# ============================================================

def simulate_gbm(S_initial, dt, N):

    Z = rng.normal(size=N)

    S_final = S_initial * np.exp( (mu - 0.5 * sigma**2)*dt + sigma*np.sqrt(dt)*Z )

    return S_final


# ============================================================
# Analytic transition PDF for GBM
# ============================================================

def gbm_pdf(S, S_initial, t):

    # log(S_t/S_0) is Gaussian:
    #
    # log(S_t/S_0)
    # ~ N((mu - sigma^2/2)t, sigma^2 t)

    shape = sigma * np.sqrt(t)

    scale = S_initial * np.exp( (mu - 0.5 * sigma**2) * t )

    return lognorm.pdf( S, s=shape, scale=scale )


# ============================================================
#
# PART 1
#
# FORWARD EQUATION
#
# ============================================================

times = [0.05, 0.25, 0.50, 1.00]

S_plot = np.linspace(40, 220, 1000)


for t in times:

    # Monte Carlo evolution from the SAME initial point
    samples = simulate_gbm( S0,  t,  Nmc )

    plt.figure(figsize=(8, 5))

    # MC estimate of probability density
    plt.hist( samples,  bins=150, density=True, alpha=0.5, label="Monte Carlo" )

    # Analytic solution
    pdf = gbm_pdf( S_plot,  S0, t )

    plt.plot( S_plot, pdf, linewidth=2, label="Analytic PDF" )
    plt.axvline( S0, linestyle="--", label=r"$S_0$" )
    plt.xlabel("S")
    plt.ylabel("Probability density")
    plt.title(f"Forward evolution: t = {t:.2f}")
    plt.legend()
    plt.grid()
    plt.show()


# ============================================================
#
# PART 2
#
# BACKWARD EQUATION
#
# Question:
#
#     What is P(S_T > K | S_t = S)?
#
# ============================================================

K = 110.0

t = 0.0

tau = T - t

S_grid = np.linspace(
    50,
    170,
    100
)

prob_MC = np.zeros_like(S_grid)


for i, S_initial in enumerate(S_grid):

    ST = simulate_gbm(
        S_initial,
        tau,
        30_000
    )

    # Digital payoff:
    #
    # g(ST) = 1 if ST > K
    #         0 otherwise

    payoff = (ST > K)

    # Monte Carlo expectation
    prob_MC[i] = np.mean(payoff)


# ============================================================
# Analytic answer
# ============================================================

#
# ST > K
#
# log(ST/S)
#
# is normally distributed
#

d = (
    np.log(S_grid / K)
    + (mu - 0.5 * sigma**2) * tau
) / (
    sigma * np.sqrt(tau)
)

prob_exact = norm.cdf(d)


# ============================================================
# Plot backward solution
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    S_grid,
    prob_MC,
    ".",
    label="Monte Carlo"
)

plt.plot(
    S_grid,
    prob_exact,
    linewidth=2,
    label="Analytic"
)

plt.axvline(
    K,
    linestyle="--",
    label="K"
)

plt.xlabel("Current stock price S")

plt.ylabel(
    r"$P(S_T>K \mid S_t=S)$"
)

plt.title(
    "Backward problem"
)

plt.legend()
plt.grid()

plt.show()


# ============================================================
#
# PART 3
#
# WATCH THE BACKWARD SOLUTION EVOLVE
#
# ============================================================

times = [
    0.0,
    0.5,
    0.8,
    0.95,
    0.99
]

plt.figure(figsize=(9, 6))


for t in times:

    tau = T - t

    d = (
        np.log(S_grid / K)
        + (mu - 0.5 * sigma**2) * tau
    ) / (
        sigma * np.sqrt(tau)
    )

    probability = norm.cdf(d)

    plt.plot(
        S_grid,
        probability,
        label=f"t = {t:.2f}"
    )


# Terminal condition
terminal = (S_grid > K).astype(float)

plt.plot(
    S_grid,
    terminal,
    linestyle="--",
    linewidth=2,
    label=r"$g(S)=1_{S>K}$"
)

plt.xlabel("S")

plt.ylabel(
    r"$u(S,t)$"
)

plt.title(
    "Backward evolution toward terminal condition"
)

plt.legend()
plt.grid()

plt.show()
