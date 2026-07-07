"""Poisson solver: charge density -> electrostatic potential.

Mirrors ``chg2pot.cpp``. The original solves Poisson's equation in atomic
units on the grid using a second-order finite-difference stencil evaluated
in reciprocal space via FFT. FFTW is replaced here by ``scipy.fft`` (falls
back to ``numpy.fft`` if SciPy is unavailable), which gives identical
results without any external C library.

The reciprocal-space multiplier reproduces the C++ eigenvalue exactly:
for each axis the finite-difference Laplacian contributes
``(2 - 2*cos(2*pi*m/N)) / h**2``.
"""

from __future__ import annotations

import numpy as np

from cubetools.cube import Cube

try:  # prefer scipy's pocketfft (fast, multi-threaded)
    from scipy.fft import fftn as _fftn, ifftn as _ifftn

    def fftn(a):
        return _fftn(a, workers=-1)   # -1 => use all CPU cores

    def ifftn(a):
        return _ifftn(a, workers=-1)
except Exception:  # pragma: no cover - numpy fallback (single-threaded)
    from numpy.fft import fftn, ifftn


def chg2pot(charge: Cube, keep_dc: bool = True) -> Cube:
    """Solve Poisson's equation, returning the potential as a ``Cube``.

    Parameters
    ----------
    charge : Cube
        Charge density cube (atomic units).
    keep_dc : bool
        If True (default), the (0,0,0) Fourier component is left untouched,
        exactly reproducing ``chg2pot.cpp``. Set False to zero it out, which
        removes the arbitrary constant offset in the potential.
    """
    n0, n1, n2 = (int(g) for g in charge.grid)
    h = charge.spacing  # (hx, hy, hz)

    rho = 4.0 * np.pi * charge.data
    rho_q = fftn(rho)

    # finite-difference Laplacian eigenvalues per axis
    m0 = (2.0 - 2.0 * np.cos(2.0 * np.pi * np.arange(n0) / n0)) / h[0] ** 2
    m1 = (2.0 - 2.0 * np.cos(2.0 * np.pi * np.arange(n1) / n1)) / h[1] ** 2
    m2 = (2.0 - 2.0 * np.cos(2.0 * np.pi * np.arange(n2) / n2)) / h[2] ** 2
    denom = m0[:, None, None] + m1[None, :, None] + m2[None, None, :]

    dc = rho_q[0, 0, 0]
    denom[0, 0, 0] = 1.0  # avoid division by zero; DC handled explicitly
    pot_q = rho_q / denom
    pot_q[0, 0, 0] = dc if keep_dc else 0.0

    pot = ifftn(pot_q).real
    return charge.like(pot)
