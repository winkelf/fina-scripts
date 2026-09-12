import numpy as np
import matplotlib.pyplot as plt

# Load the two columns
data = np.loadtxt("perez_companc_extracted.txt", skiprows=1)

x = data[:, 0]
y = data[:, 1]

R = [0]

for i in range(len(x)-1):
    #print( (y[i+1]-y[i])/y[i] )
    R.append((y[i+1]-y[i])/y[i])

print(sum(R)/len(R))

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(x, R, color="black")
#plt.plot(x, y, color="black", linewidth=1.5)

plt.xlabel("Index")
plt.ylabel("Perez Companc")
plt.grid(alpha=0.3)

#plt.hist(R, bins=20)

plt.tight_layout()
plt.show()
