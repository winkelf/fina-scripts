import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

# ============================================================
# Parameters
# ============================================================

S0 = 100.0       # stock price today
K = 100.0        # strike
T = 1.0          # maturity [years]
r = 0.05         # risk-free rate
sigma = 0.20     # volatility

# ============================================================
# Black-Scholes functions
# ============================================================

def d1(S, tau):
    return ( np.log(S / K) + (r + 0.5*sigma**2)*tau ) / (sigma*np.sqrt(tau))


def d2(S, tau):
    return d1(S, tau) - sigma*np.sqrt(tau)


# V(S,t), solution of BS for a call 
def call_price(S, tau):

    D1 = d1(S, tau)
    D2 = d2(S, tau)

    return ( S*norm.cdf(D1) - K*np.exp(-r*tau)*norm.cdf(D2))


def delta(S, tau):
    return norm.cdf(d1(S, tau))


def gamma(S, tau):
    return ( norm.pdf(d1(S, tau)) / (S * sigma * np.sqrt(tau)) )


def theta(S, tau):

    D1 = d1(S, tau)
    D2 = d2(S, tau)

    return ( -S * norm.pdf(D1) * sigma / (2 * np.sqrt(tau)) - r * K * np.exp(-r * tau) * norm.cdf(D2) )


# ============================================================
# Look at the Black-Scholes PDE
# at a fixed time-to-maturity
# ============================================================

tau = 0.5

S = np.linspace(50, 150, 500)

V = call_price(S, tau)

Delta = delta(S, tau)
Gamma = gamma(S, tau)
Theta = theta(S, tau)


# ------------------------------------------------------------
# Individual terms in the Black-Scholes PDE
#
# Theta
# + 1/2 sigma^2 S^2 Gamma
# + r S Delta
# - r V
# = 0
# ------------------------------------------------------------

theta_term = Theta

gamma_term = 0.5 * sigma**2 * S**2 * Gamma

delta_term = r * S * Delta

discount_term = -r * V

total = (
    theta_term
    + gamma_term
    + delta_term
    + discount_term
)


# ============================================================
# Plot option price
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(S, V)
plt.axvline(K, linestyle="--")

plt.xlabel("Stock price S")
plt.ylabel("Call price V")
plt.title("Black-Scholes call price")

plt.grid()

plt.show()


# ============================================================
# Plot Delta and Gamma
## ============================================================
#
#plt.figure(figsize=(8, 5))
#
#plt.plot(S, Delta, label="Delta")
#plt.plot(S, Gamma, label="Gamma")
#
#plt.axvline(K, linestyle="--")
#
#plt.xlabel("Stock price S")
#plt.title("Delta and Gamma")
#
#plt.legend()
#plt.grid()
#
#plt.show()


# ============================================================
# Plot all terms of the PDE
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(S, theta_term,
         label=r"$\Theta$")

plt.plot(S, gamma_term,
         label=r"$\frac{1}{2}\sigma^2S^2\Gamma$")

plt.plot(S, delta_term,
         label=r"$rS\Delta$")

plt.plot(S, discount_term,
         label=r"$-rV$")

plt.plot(S, total,
         linestyle="--",
         linewidth=2,
         label="Sum")

plt.axhline(0, linewidth=1)
plt.axvline(K, linestyle=":")

plt.xlabel("Stock price S")
plt.ylabel("Contribution")
plt.title("Terms in the Black-Scholes PDE")

plt.legend()
plt.grid()

plt.show()


# ============================================================
# Numerical check at S = 100
# ============================================================

S_test = 100.0

V_test = call_price(S_test, tau)
Delta_test = delta(S_test, tau)
Gamma_test = gamma(S_test, tau)
Theta_test = theta(S_test, tau)

terms = [
    Theta_test,
    0.5 * sigma**2 * S_test**2 * Gamma_test,
    r * S_test * Delta_test,
    -r * V_test
]

print("At S =", S_test)
print()
print("Option value:", V_test)
print("Delta:", Delta_test)
print("Gamma:", Gamma_test)

print()
print("Black-Scholes PDE terms:")
print("Theta                 =", terms[0])
print("Gamma contribution     =", terms[1])
print("r S Delta              =", terms[2])
print("-r V                   =", terms[3])

print()
print("Sum =", sum(terms))
