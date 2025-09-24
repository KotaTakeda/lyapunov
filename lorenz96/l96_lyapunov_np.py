import numpy as np
from tqdm import tqdm
from set_params import J, F, dt, T # load local set_params.py

# Print parameters
print("(J, F)", (J, F))

# Set arameters
p = np.full(J, F)

# Time array
t = np.arange(0.0, T, dt)


# 4th order Runge Kutta
def rk4(f, t, x, p, dt):
    """
    4th order Runge Kutta integrator.
    """
    k1 = f(t, x, p)
    k2 = f(t + dt / 2, x + k1 * dt / 2, p)
    k3 = f(t + dt / 2, x + k2 * dt / 2, p)
    k4 = f(t + dt, x + k3 * dt, p)
    xt = x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
    return xt


# Lorenz 96 model
def lorenz96(t, x, p):
    J = x.shape[0]
    dxdt = np.zeros_like(x)
    for i in range(J):
        dxdt[i] = (x[(i + 1) % J] - x[i - 2]) * x[i - 1] - x[i] + p[i]
    return dxdt


# Jacobian of Lorenz 96
def lorenz96_jacobian(t, x, p):
    J = x.shape[0]
    jac = np.zeros((J, J))
    for i in range(J):
        jac[i, (i - 2) % J] = -x[i - 1]
        jac[i, (i - 1) % J] = x[(i + 1) % J] - x[i - 2]
        jac[i, i] = -1
        jac[i, (i + 1) % J] = x[i - 1]
    return jac


# Compute Lyapunov exponents
def compute_le(f, Jf, x0, t, p, return_sol=False):
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
        # return np.concatenate([dx, dV])
        return np.append(dx, dV)

    V0 = np.eye(Nx).flatten()
    LE = np.zeros((Nt - 1, Nx))
    xv = np.zeros((Nt, Nx * (Nx + 1)))
    xv[0] = np.concatenate([x0, V0])

    for i, (t1, t2) in tqdm(enumerate(zip(t[:-1], t[1:]))):
        xv_t = rk4(dSdt, t1, xv[i], p, t2 - t1)
        V = xv_t[Nx:].reshape((Nx, Nx))
        Q, R = np.linalg.qr(V)
        xv[i + 1] = np.concatenate([xv_t[:Nx], Q.flatten()])
        LE[i] = np.log(np.abs(np.diag(R)))

    LE = np.cumsum(LE, axis=0) / np.tile(t[1:], (Nx, 1)).T

    if return_sol:
        return LE, xv[:, :Nx]
    return LE


# Compute spin-up trajectory
# Initial conditions
x0 = np.ones(J) * F
x0[J // 2] *= 1.01

# spin-up
t_spinup = np.arange(0.0, 1000.0, dt)
x = x0
for t in tqdm(t_spinup):
    x = rk4(lorenz96, t, x, p, dt)

# Compute Lyapunov exponents
LE = compute_le(lorenz96, lorenz96_jacobian, x, t, p)
i_neutral = np.argmin(np.abs(LE[-1])) # The neutral exponent is the closest one to zero.
print("Final Lyapunov exponents:", LE[-1])
print("Number of positive-neutral exponents:", i_neutral+1)

# Save Lyapunov exponents
np.save("l96_le.npy", LE)
