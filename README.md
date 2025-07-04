# Computing Lyapunov exponents

## Requirements
- Numpy
- Matplotlib
- tqdm
- Cython (if use)
<!-- - JAX (if use) -->

## Lorenz 96 equation
The Lorenz 96 equation is a toy model for spatio-temporal chaos.
For a number of variables $J \in \mathbb{N}$, external force $F \in \mathbb{R}$ and state variable $\mathbb{u} \in \mathbb{R}^J$ on 1D periodic domain, the Lorenz 96 equation (Lorenz 1996) is given by

$$ \frac {du^{i}}{dt} =(u^{i+1}-u^{i-2})u^{i-1}-u^{i}+F, $$

with $u^{-1} = u^{J-1}$, $u^0 = u^J$, and $u^{J+1} = u^1$.

Typical parameter: (J, F) = (40, 8.0)

### Compute
#### Move directory
```sh
cd lorenz96
```

#### Set parameters
modify `set_params_l96.py`

#### by Numpy
Compute LEs
```sh
python3 l96_lyapunov_np.py
```

#### by Cython
Build
```sh
python3 setup.py build_ext --inplace
```

Compute LEs
```sh
python3 l96_lyapunov_cython.py
```
<!-- 
#### by JAX
Compute LEs
```sh
python3 l96_lyapunov_jnp.py
``` -->

### Plot
Plot
```sh
python3 plot_lyapunov_exponents.py
```

![LE](https://github.com/KotaTakeda/lyapunov/blob/main/lorenz96/lyapunov_exponents.png)
![LE_t](https://github.com/KotaTakeda/lyapunov/blob/main/lorenz96/lyapunov_exponents_t.png)


## References
- Sandri, M. (1996). Numerical calculation of Lyapunov exponents. Mathematica Journal, 6(3), 78-84.
- Hubertus F. von Bremen, F. E. Udwadia, W. Proskurowski, An efficient QR based method for the computation of Lyapunov exponents, Physica D: Nonlinear Phenomena, Volume 101(1–2), 1997, 1-16.
- Cython, https://github.com/cython/cython
- JAX, https://jax.readthedocs.io/en/latest/
