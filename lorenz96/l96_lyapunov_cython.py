import numpy as np
import scipy as sp
from lorenz96_cython import rk4_cython, lorenz96_cython, lorenz96_jacobian_cython, dSdt_cython, rk4_cython_dSdt_cython
from tqdm import tqdm
from lorenz96.set_params import J, F, dt, T, T_spinup

# Print parameters
print("(J, F)", (J, F))

# Set parameters
p = np.full(J, F)


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
    xv[0, :Nx]  = x0
    xv[0, Nx:]  = V0
    for i, (t1, t2) in tqdm(enumerate(zip(t[:-1], t[1:]))):
        # main
        # xv_t = rk4_cython(dSdt, t1, xv[i], p, t2 - t1)  # ver1: state-variation evolution
        # xv_t = rk4_cython(dSdt_cython, t1, xv[i], p, t2 - t1)  # ver2: state-variation evolution
        xv_t = rk4_cython_dSdt_cython(t1, xv[i], p, t2 - t1) # ver3: state-variation evolution
        V = xv_t[Nx:].reshape((Nx, Nx))  # extract variation
        Q, R = sp.linalg.qr(V)
        xv[i+1, :Nx] = xv_t[:Nx]
        xv[i+1, Nx:] = Q.flatten()
        LE[i] = np.log(np.abs(np.diag(R)))  # local LE t1 -> t2

    LE = np.cumsum(LE, axis=0) / np.tile(t[1:], (Nx, 1)).T
    return LE


def compute_le_cython_eps(f, Jf, x0, p, t0, dt, maxiter, eps):
    Nx = len(x0)

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
    xv = np.concatenate([x0, V0])
    i = 0
    t_list = []
    LE_list = []
    while i < maxiter:
        i += 1
        t = i*dt+t0
        t_list.append(t)

        # main
        # xv_t = rk4_cython(dSdt, t, xv, p, dt) # ver1
        xv_t = rk4_cython_dSdt_cython(t, xv, p, dt)
        V = xv_t[Nx:].reshape((Nx, Nx))
        Q, R = sp.linalg.qr(V)
        xv[:Nx] = xv_t[:Nx]
        xv[Nx:] = Q.flatten()
        LE_list.append(np.log(np.abs(np.diag(R))))  # local LE t -> t + dt

        if i % 100 == 0:
            LE = np.cumsum(LE_list, axis=0) / np.tile(t_list, (Nx, 1)).T
            if np.all(np.abs(LE[-2] - LE[-1]) < np.abs(LE[-2]) * eps):
                break
            else:
                print(i, "residual: ", max(np.abs(LE[-2] - LE[-1])))

    LE = np.cumsum(LE_list, axis=0) / np.tile(t_list, (Nx, 1)).T
    return LE


# Compute Lyapunov exponents
# TODO: remove criteria 1
# Criteria 1: Fixed time array
t = np.arange(0.0, T, dt)
# LE = compute_le_cython(lorenz96_cython, lorenz96_jacobian_mv, x, t, p)
LE = compute_le_cython(lorenz96_cython, lorenz96_jacobian_cython, x, t, p)

# Criteria 2: Maxiter and threshold (eps)
maxiter = int(T/dt)
eps = 1e-6
LE = compute_le_cython_eps(lorenz96_cython, lorenz96_jacobian_cython, x, p, 0.0, dt, maxiter, eps)

# Print results
i_neutral = np.argmin(np.abs(LE[-1])) # The neutral exponent is the closest one to zero.
print("Final Lyapunov exponents:", LE[-1])
print("Number of positive-neutral exponents:", i_neutral+1)

# Save the time series of Lyapunov exponents
np.save("l96_le.npy", LE)

# if __name__ == "__main__":
#     import cProfile
#     cProfile.run("LE = compute_le_cython(lorenz96_cython, lorenz96_jacobian_cython, x, t, p)")