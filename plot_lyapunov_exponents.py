import matplotlib.pyplot as plt
import numpy as np

# Load data
LE = np.load("l96_le.npy")
print(LE.shape)

# Plot
plt.figure()
plt.plot(LE[::100], lw=0.5)
plt.xlabel("time")
plt.ylabel("Lyapunov exponent")
plt.show()

# Plot
fig = plt.figure()
plt.plot(LE[-1], "o-")
plt.xlabel("Index")
plt.ylabel("Lyapunov exponent")
plt.show()
fig.savefig("lyapunov_exponents.pdf", transparent=True)
