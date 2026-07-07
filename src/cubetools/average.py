"""Planar-averaged profile along one axis. Mirrors ``potAvg`` in chg2pot.cpp."""

from __future__ import annotations

import numpy as np

from .cube import Cube


def planar_average(cube: Cube, axis: int):
    """Return ``(coords, avg)`` averaging the data over the two other axes.

    Parameters
    ----------
    cube : Cube
    axis : int
        Axis index 0, 1 or 2. (CLI ``dir`` 1/2/3 map to 0/1/2.)

    Returns
    -------
    coords : ndarray, shape (N,)
        Position along ``axis`` in Bohr (``index * spacing``).
    avg : ndarray, shape (N,)
        In-plane average at each slice.
    """
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    other = tuple(a for a in (0, 1, 2) if a != axis)
    avg = cube.data.mean(axis=other)
    N = int(cube.grid[axis])
    coords = np.arange(N) * cube.spacing[axis]
    return coords, avg


def save_planar_average(cube: Cube, axis: int, fname: str) -> None:
    """Write a two-column ``coord  value`` file (matches ``pot.{x,y,z}.avg``)."""
    coords, avg = planar_average(cube, axis)
    with open(fname, "w") as fp:
        for c, v in zip(coords, avg):
            fp.write(f"{c:f} {v:f}\n")
