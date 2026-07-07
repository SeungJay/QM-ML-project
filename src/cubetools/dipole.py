"""Dipole correction for a charge/potential cube.

Mirrors the (correct) ``dir==3`` branch of ``dipc2.cpp``, generalised to
all three axes. A sawtooth dipole moment is accumulated along the chosen
axis and two adjacent planes at ``position`` are set to +/- that moment,
normalised by the in-plane grid count.

Note: the ``dir==1`` and ``dir==2`` branches of the original C++ contained
loop-variable typos; this implementation follows the well-formed z-branch
consistently for every axis.
"""

from __future__ import annotations

import numpy as np

from .cube import Cube


def dipole_correction(cube: Cube, axis: int, position: int) -> Cube:
    """Apply a dipole correction along ``axis`` (0=x, 1=y, 2=z).

    Parameters
    ----------
    cube : Cube
        Input cube (modified copy returned).
    axis : int
        Axis index 0, 1 or 2. (CLI ``dir`` values 1/2/3 map to 0/1/2.)
    position : int
        Grid index of the correction plane.
    """
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")

    out = cube.copy()
    data = out.data
    n = [int(g) for g in cube.grid]
    N = n[axis]

    idx = np.arange(N)
    weight = np.where(idx < position, idx, idx - N).astype(float)

    # broadcast weight along the chosen axis and sum over the whole grid
    shape = [1, 1, 1]
    shape[axis] = N
    dipole = float(np.sum(data * weight.reshape(shape)))

    in_plane = (n[0] * n[1] * n[2]) // N
    val = dipole / in_plane

    slicer_a = [slice(None)] * 3
    slicer_b = [slice(None)] * 3
    slicer_a[axis] = position
    slicer_b[axis] = position + 1
    data[tuple(slicer_a)] = val
    data[tuple(slicer_b)] = -val

    return out
