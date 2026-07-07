"""In-process replacements for the original cube command-line binaries.

Each function reads its input cube file(s), runs the operation, and writes the
SAME default output filename that the corresponding C/C++ binary produced
(``multiplied.cube``, ``add.cube``, ``subtracted.cube``, ``cube.{x,y,z}.avg``,
``z_sig.cube``, ``pot.cube``, ``dipc.cube``).

This lets a driver call them directly instead of spawning the compiled binaries
via subprocess — no binary dependency, no process-spawn overhead. Header comment
lines match the originals; they are not parsed by any downstream tool.

Depends on :mod:`cubetools` (cube ops) and :mod:`poisson` (chg2pot).

Example
-------
    from qmml_run import pipeline as cube
    cube.cube_multi("v_hartree.cube", 2)      # -> multiplied.cube
    cube.dipc("MDrho.cube", 3, 10)            # -> dipc.cube
    cube.chg2pot("dipc.cube", 3)              # -> pot.cube
    cube.cube_avg("MDpot.cube", 3)            # -> cube.z.avg
"""

from __future__ import annotations

from cubetools import Cube
from cubetools.operations import (
    add as _add, subtract as _sub, multiply as _mul, sigmoid_profile as _sig,
)
from cubetools.dipole import dipole_correction as _dipc
from cubetools.average import save_planar_average as _save_avg
from poisson import chg2pot as _chg2pot

_RESCALE = (" Cubefile rescaled", " lattice unit Bohr")
_AVG_NAME = {1: "cube.x.avg", 2: "cube.y.avg", 3: "cube.z.avg"}


def cube_multi(infile: str, factor, out: str = "multiplied.cube") -> str:
    """multiplied.cube = infile * factor. Mirrors ``cube_multi``."""
    _mul(Cube.from_file(infile), float(factor)).to_file(out, comment=_RESCALE)
    return out


def cube_add(a: str, b: str, out: str = "add.cube") -> str:
    """add.cube = a + b. Mirrors ``cube_add``."""
    _add(Cube.from_file(a), Cube.from_file(b)).to_file(out, comment=_RESCALE)
    return out


def cube_sub(a: str, b: str, out: str = "subtracted.cube") -> str:
    """subtracted.cube = a - b. Mirrors ``cube_sub``."""
    _sub(Cube.from_file(a), Cube.from_file(b)).to_file(out, comment=_RESCALE)
    return out


def cube_sigmoid(infile: str, target, z_ini, z_fin, steepness,
                 out: str = "z_sig.cube") -> str:
    """z_sig.cube = sigmoid ramp along z. Mirrors ``cube_sigmoid``."""
    _sig(Cube.from_file(infile), float(target), float(z_ini), float(z_fin),
         float(steepness)).to_file(out, comment=_RESCALE)
    return out


def cube_avg(infile: str, direction, out: str | None = None) -> str:
    """Write cube.{x,y,z}.avg (coord, in-plane average). Mirrors ``cube_avg``."""
    d = int(direction)
    if out is None:
        out = _AVG_NAME[d]
    _save_avg(Cube.from_file(infile), d - 1, out)
    return out


def chg2pot(infile: str, direction=3, out: str = "pot.cube") -> str:
    """pot.cube = Poisson solve of infile (Ha units). Mirrors ``chg2pot``.

    ``direction`` is accepted for call-signature parity with the binary (which
    also wrote a planar average); use :func:`cube_avg` for the average.
    """
    _chg2pot(Cube.from_file(infile)).to_file(
        out, comment=(" Cubefile created from chg2pot",
                      " ES potential, lattice unit Bohr, grid unit Ha"))
    return out


def dipc(infile: str, direction, position, out: str = "dipc.cube") -> str:
    """dipc.cube = dipole-corrected infile. Mirrors ``dipc``/``dipc2``."""
    _dipc(Cube.from_file(infile), axis=int(direction) - 1,
          position=int(position)).to_file(
        out, comment=(" Cubefile created from dipc2",
                      " ES potential, lattice unit Bohr, grid unit Ry"))
    return out
