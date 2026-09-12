import numpy as np
import matplotlib.pyplot as plt

# Parameter
S0 = 100    # initial stock price
mu = 0.08   # price movement 20% per year 
sigma = 0.2 # market volatility 
T = 1.0     # total time simulation 1 year
N = 252     # number of trading day in a year
M = 50     # num of trials / walker path

dt = T / N

# Time grid
t = np.linspace(0, T, N)

# Storage
S = np.zeros((M, N))
S[:, 0] = S0

# Simulasi Path 
for i in range(M):
    for j in range(1, N):
        dW = np.random.normal(0, np.sqrt(dt)) # similar to takeStep and decide new direction
        S[i, j] = S[i, j-1] * (1 + mu*dt + sigma*dW) # apply the step taken (moveWalker)

# Plot
plt.figure(figsize=(10, 5))
for i in range(M):
    plt.plot(t, S[i])
plt.xlabel("Time (year)")
plt.ylabel("Stock Price ")
plt.show()
