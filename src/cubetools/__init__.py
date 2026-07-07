"""cubetools — NumPy-backed toolkit for Gaussian cube files.

Cube container plus the cube-manipulation tools (arithmetic, sigmoid profile,
dipole correction, planar averaging). Heavy work is delegated to NumPy, so no
compilation is required.

The Poisson solver lives in the sibling package :mod:`poisson` (``chg2pot``),
and the QM/ML driver in :mod:`qmml_run`.
"""

from __future__ import annotations

from .cube import Cube
from .operations import add, subtract, multiply, sigmoid_profile
from .dipole import dipole_correction
from .average import planar_average, save_planar_average

__version__ = "0.1.0"

__all__ = [
    "Cube",
    "add",
    "subtract",
    "multiply",
    "sigmoid_profile",
    "dipole_correction",
    "planar_average",
    "save_planar_average",
]
