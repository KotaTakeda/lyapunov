import matplotlib.pyplot as plt
import numpy as np
from set_params_l96 import J, F, dt, T

# Load data
LE = np.load("l96_le.npy")
print(LE.shape)

# Plot time series
t = np.arange(0.0, T, dt)
plot_per = 100
fig1 = plt.figure()
plt.plot(t[::plot_per], LE[::plot_per], lw=0.5)
plt.plot(t[::plot_per], np.zeros_like(t[::plot_per]), lw=0.5, color="grey", linestyle="--")
plt.xlabel("time")
plt.ylabel("Lyapunov exponent")
plt.title(f"F={F}, J={J}, dt={dt}, T={T}")
fig1.savefig("lyapunov_exponents_t.png", transparent=True)
plt.show()

# Plot final Lyapunov exponents
idx_zero = np.argmin(np.abs(LE[-1]))
fig2 = plt.figure()
plt.plot(LE[-1], "o-", lw=0.5, color="black")
plt.plot(
    idx_zero,
    LE[-1, idx_zero],
    color="red",
    marker="o",
    markersize=5,
)
plt.xlabel("Index")
plt.ylabel("Lyapunov exponent")
plt.title(f"F={F}, J={J}, dt={dt}, T={T}")
fig2.savefig("lyapunov_exponents.png", transparent=True)
plt.show()
