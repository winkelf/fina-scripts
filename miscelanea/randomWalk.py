import random
import matplotlib.pyplot as plt

# Parameters
n_tosses      = 10000
initial_value = 1000.0

# Store trajectory
values = [initial_value]

current_value = initial_value

for _ in range(n_tosses):
    toss = random.choice(["H", "T"])

    if toss == "H":
        current_value *= 1.001
    else:
        current_value *= 0.999

    values.append(current_value)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(values, lw=1.5)
plt.xlabel("Number of tosses")
plt.ylabel("Value")
plt.title("Multiplicative Coin Toss Process")
plt.grid(True)
plt.savefig("test.pdf")
