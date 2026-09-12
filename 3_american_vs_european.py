import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# ============================================================
# Black-Scholes European call with continuous dividend yield
# ============================================================

def european_call(S, K, T, r, sigma, q=0.0):

    S = np.asarray(S)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# ============================================================
# Parameters
# ============================================================

K = 100.0
T = 1.0
r = 0.03
sigma = 0.25

S = np.linspace(1, 1000, 3000)

exercise_value = np.maximum(S - K, 0.0)


# ============================================================
# Plot 1: no dividends
# ============================================================

q_no_div = 0.0

call_no_div = european_call(S, K, T, r, sigma, q_no_div)
difference_no_div = call_no_div - exercise_value

S_point = 120.0
call_point = european_call(S_point, K, T, r, sigma, q_no_div)
exercise_point = max(S_point - K, 0.0)

plt.figure(figsize=(8, 5))
plt.plot(S, call_no_div, label="European call")
plt.plot(S, exercise_value, "--", label=r"Immediate exercise: $\max(S-K,0)$")

plt.scatter(S_point, call_point, s=60, zorder=5)
plt.scatter(S_point, exercise_point, s=60, zorder=5)
plt.plot([S_point, S_point], [exercise_point, call_point], ":")

plt.annotate(f"European call = {call_point:.2f}", xy=(S_point, call_point), xytext=(S_point + 40, call_point + 15), arrowprops=dict(arrowstyle="->"))
plt.annotate(f"Exercise now = {exercise_point:.2f}", xy=(S_point, exercise_point), xytext=(S_point + 40, exercise_point - 10), arrowprops=dict(arrowstyle="->"))

plt.xlabel("Stock price S")
plt.ylabel("Option value")
plt.title("Call option with no dividends")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# Plot 2: with dividends
# ============================================================

q_div = 0.05

call_div = european_call(S, K, T, r, sigma, q_div)
difference_div = call_div - exercise_value

# Find where the European call first violates
# the American constraint:
#
# C_E < max(S-K, 0)

violation_indices = np.where(difference_div < 0)[0]

if len(violation_indices) > 0:
    S_critical = S[violation_indices[0]]
else:
    S_critical = None


plt.figure(figsize=(8, 5))
plt.plot(S, call_div, label="European call")
plt.plot(S, exercise_value, "--", label=r"Immediate exercise: $\max(S-K,0)$")

if S_critical is not None:
    plt.axvline(S_critical, linestyle=":", linewidth=2, label=fr"Constraint violation starts at $S \approx {S_critical:.1f}$")
    plt.text(S_critical + 20, 0.55 * np.max(call_div), "European solution\nnot valid as\nAmerican solution")

plt.xlabel("Stock price S")
plt.ylabel("Option value")
plt.title(f"Call option with dividend yield q = {q_div:.0%}")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# Plot 3: test the American constraint directly
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(S, difference_no_div, label="No dividends")
plt.plot(S, difference_div, label=f"Dividend yield q = {q_div:.0%}")
plt.axhline(0, linestyle="--")

if S_critical is not None:
    plt.axvline(S_critical, linestyle=":", linewidth=2, label=fr"Constraint violation starts at $S \approx {S_critical:.1f}$")

plt.xlabel("Stock price S")
plt.ylabel(r"$C_{\rm European} - \max(S-K,0)$")
plt.title("Testing the American early-exercise constraint")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# Print useful values
# ============================================================

print("No-dividend example:")
print(f"S = {S_point:.2f}")
print(f"European call value    = {call_point:.2f}")
print(f"Immediate exercise     = {exercise_point:.2f}")
print(f"Continuation advantage = {call_point - exercise_point:.2f}")
print()

if S_critical is not None:
    print("With dividends:")
    print(f"The European solution first violates")
    print(f"the American constraint at approximately S = {S_critical:.2f}")
else:
    print("No violation found in the chosen S range.")
