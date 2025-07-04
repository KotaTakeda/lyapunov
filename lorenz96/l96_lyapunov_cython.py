import numpy as np
from lorenz96_cython import rk4_cython, lorenz96_cython, lorenz96_jacobian_cython
from tqdm import tqdm
from set_params_l96 import J, F, dt, T, T_spinup

# Print parameters
print("(J, F)", (J, F))

# Set parameters
p = np.full(J, F)

# Time array
t = np.arange(0.0, T, dt)

# Compute spin-up trajectory
# Initial conditions
x0 = np.ones(J) * F
x0[J // 2] *= 1.01

# Spin-up
t_spinup = np.arange(0.0, T_spinup, dt)
x = x0
for tt in tqdm(t_spinup):
    x = rk4_cython(lorenz96_cython, tt, x, p, dt)


# Function to compute Lyapunov exponents
def compute_le_cython(f, Jf, x0, t, p):
    Nx = len(x0)
    Nt = len(t)

    def dVdt(V, t, x):
        V = V.reshape((Nx, Nx))
        dV = Jf(t, x, p).dot(V)
        return dV.flatten()

    def dSdt(t, xv, p):
        x = xv[:Nx]
        V = xv[Nx:]
        dx = f(t, x, p)
        dV = dVdt(V, t, x)
        return np.concatenate([dx, dV])

    V0 = np.eye(Nx).flatten()
    LE = np.zeros((Nt - 1, Nx))
    xv = np.zeros((Nt, Nx * (Nx + 1)))
    xv[0] = np.concatenate([x0, V0])
    for i, (t1, t2) in tqdm(enumerate(zip(t[:-1], t[1:]))):
        xv_t = rk4_cython(dSdt, t1, xv[i], p, t2 - t1)
        V = xv_t[Nx:].reshape((Nx, Nx))
        Q, R = np.linalg.qr(V)
        xv[i + 1] = np.concatenate([xv_t[:Nx], Q.flatten()])
        LE[i] = np.log(np.abs(np.diag(R)))

    LE = np.cumsum(LE, axis=0) / np.tile(t[1:], (Nx, 1)).T
    return LE


# Compute Lyapunov exponents
LE = compute_le_cython(lorenz96_cython, lorenz96_jacobian_cython, x, t, p)
i_neutral = np.argmin(np.abs(LE[-1])) # The neutral exponent is the closest one to zero.
print("Final Lyapunov exponents:", LE[-1])
print("Number of positive-neutral exponents:", i_neutral+1)

# Save Lyapunov exponents
np.save("l96_le.npy", LE)
