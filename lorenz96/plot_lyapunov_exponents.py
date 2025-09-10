import matplotlib.pyplot as plt
import numpy as np
from set_params_l96 import J, F, dt, T

# Set save format
save_fmt = "pdf"  # ["png", "pdf"]

# Load data
LE = np.load("l96_le.npy")
print(LE.shape)

# Plot time series
t = np.arange(0.0, T, dt)
n_iter = int(T / dt)  # variable to display
assert np.isclose(T, n_iter * dt)
plot_per = 100
fig1 = plt.figure()
plt.plot(t[::plot_per], LE[::plot_per], lw=0.5)
plt.plot(t[::plot_per], np.zeros_like(t[::plot_per]), lw=0.5, color="grey", ls="--")
plt.xlabel("time")
plt.ylabel("Lyapunov exponent")
plt.title(f"$F={F}, J={J}, dt={dt}, n={n_iter}$")
fig1.savefig(f"lyapunov_exponents_t.{save_fmt}", transparent=True)
plt.show()

# Plot final Lyapunov exponents
i_neutral = np.argmin(np.abs(LE[-1]))
fig2 = plt.figure()
plt.plot(LE[-1], "o-", lw=0.5, color="black")
plt.plot(i_neutral, LE[-1, i_neutral], color="red", marker="o", markersize=5)
plt.xlabel("Index")
plt.ylabel("Lyapunov exponent")
plt.title(f"$F={F}, J={J}, dt={dt}, n={n_iter}$")
fig2.savefig(f"lyapunov_exponents.{save_fmt}", transparent=True)
plt.show()
