import jax
import jax.numpy as jnp
from jax import jit, jacfwd, vmap, lax
from tqdm import tqdm
from functools import partial
from lorenz96.set_params_l96 import J, F, dt, T

# Enable Metal
jax.config.update("jax_platform_name", "cpu")  # "cpu" or "METAL"(macOS)
jax.config.update("jax_enable_x64", False)
print("JAX Devices:", jax.devices())

# Parameters
F = jnp.float32(F)
p = jnp.full(J, F, dtype=jnp.float32)


@partial(jit, static_argnames=("f",))
def rk4(f, t, x, p, dt):
    k1 = f(t, x, p)
    k2 = f(t + dt / 2, x + k1 * dt / 2, p)
    k3 = f(t + dt / 2, x + k2 * dt / 2, p)
    k4 = f(t + dt, x + k3 * dt, p)
    xt = x + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
    return xt


@jit
def lorenz96(t, x, p):
    def compute_dx(i):
        return (x[(i + 1) % J] - x[i - 2]) * x[i - 1] - x[i] + p[i]

    return vmap(compute_dx)(jnp.arange(J))


lorenz96_jacobian = jit(jacfwd(lorenz96, argnums=1))


@partial(jit, static_argnames=("f", "Jf"))
def compute_le_jnp(f, Jf, x0, t, p, return_sol=False):
    Nx = len(x0)
    Nt = len(t)

    def dVdt(t, V, x):
        V = jnp.reshape(V, (Nx, Nx))
        dV = jnp.dot(Jf(t, x, p), V)
        return dV.flatten()

    def dSdt(t, xv, p):
        x = xv[:Nx]
        V = xv[Nx:]
        dx = f(t, x, p)
        dV = dVdt(t, V, x)
        return jnp.append(dx, dV)

    V0 = jnp.eye(Nx).flatten()
    LE = jnp.zeros((Nt - 1, Nx))
    xv = jnp.zeros((Nt, Nx * (Nx + 1)))
    xv = xv.at[0].set(jnp.append(x0, V0))

    for i, (t1, t2) in tqdm(enumerate(zip(t[:-1], t[1:]))):
        xv_t = rk4(dSdt, t1, xv[i], p, t2 - t1)
        V = jnp.reshape(xv_t[Nx:], (Nx, Nx))
        # Q, R = jnp.linalg.qr(V, mode="reduced")
        Q, R = lax.linalg.qr(V, full_matrices=False)
        xv = xv.at[i + 1].set(jnp.append(xv_t[:Nx], Q.flatten()))
        LE = LE.at[i].set(jnp.log(jnp.abs(jnp.diag(R))))

    LE = jnp.cumsum(LE, axis=0) / jnp.tile(t[1:], (Nx, 1)).T

    if return_sol:
        return LE, xv[:, :Nx]
    return LE


x0 = jnp.ones(J, dtype=jnp.float32) * F
x0 = x0.at[J // 2].multiply(1.01)

t = jnp.arange(0.0, T, dt)  # Larger range for better Metal utilization

t_spinup = jnp.arange(0.0, 1000.0, dt)
x = x0
for tt in tqdm(t_spinup):
    x = rk4(lorenz96, tt, x, p, dt)


LE = compute_le_jnp(lorenz96, lorenz96_jacobian, x, t, p)
print("Final Lyapunov exponents:", LE[-1])
print("Number of positive exponents:", sum(LE[-1] > 0))


# Save Lyapunov exponents
jnp.save("l96_le.npy", LE)
