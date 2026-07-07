"""Elementwise cube operations (add, subtract, multiply, sigmoid).

These mirror ``cube_add.c``, ``cube_sub.c``, ``cube_multi.c`` and
``cube_sigmoid.c``.
"""

from __future__ import annotations

import numpy as np

from .cube import Cube


def add(a: Cube, b: Cube) -> Cube:
    """Return ``a + b`` (elementwise). Mirrors ``cube_add``."""
    return a.like(a.data + b.data)


def subtract(a: Cube, b: Cube) -> Cube:
    """Return ``a - b`` (elementwise). Mirrors ``cube_sub``."""
    return a.like(a.data - b.data)


def multiply(a: Cube, factor: float) -> Cube:
    """Return ``a * factor`` (scalar). Mirrors ``cube_multi``."""
    return a.like(a.data * float(factor))


def sigmoid_profile(
    cube: Cube,
    final_value: float,
    z_ini: float,
    z_fin: float,
    steepness: float,
) -> Cube:
    """Replace the grid with a sigmoid ramp along z. Mirrors ``cube_sigmoid``.

    For each z-plane the value is

        final_value / (1 + exp(-steepness * s)),  s = (z - z_ini)/(z_fin - z_ini)*12 - 6

    where ``z = k * spacing_z``. The profile is constant over x and y.
    """
    n0, n1, n2 = (int(g) for g in cube.grid)
    dz = cube.spacing[2]
    k = np.arange(n2)
    z = k * dz
    scaled = (z - z_ini) / (z_fin - z_ini) * 12.0 - 6.0
    profile = final_value / (1.0 + np.exp(-steepness * scaled))  # shape (n2,)
    data = np.broadcast_to(profile, (n0, n1, n2)).copy()
    return cube.like(data)
