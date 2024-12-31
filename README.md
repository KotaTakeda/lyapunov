# Computing Lyapunov exponents of Lorenz 96
## Requirements
- Numpy
- Matplotlib
- Cython (if use)
<!-- - JAX (if use) -->

## Compute
### by Numpy
Compute LEs
```sh
python3 l96_lyapunov_np.py
```

### by Cython
Build
```sh
python3 setup.py build_ext --inplace
```

Compute LEs
```sh
python3 l96_lyapunov_cython.py
```
<!-- 
### by JAX
Compute LEs
```sh
python3 l96_lyapunov_jnp.py
``` -->

## Plot
Plot
```sh
python3 plot_lyapunov_exponents.py
```

## References
- Sandri, M. (1996). Numerical calculation of Lyapunov exponents. Mathematica Journal, 6(3), 78-84.
- Hubertus F. von Bremen, F. E. Udwadia, W. Proskurowski, An efficient QR based method for the computation of Lyapunov exponents, Physica D: Nonlinear Phenomena, Volume 101(1–2), 1997, 1-16.
- Cython, https://github.com/cython/cython
- JAX, https://jax.readthedocs.io/en/latest/