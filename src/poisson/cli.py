"""Command-line entry point for the Poisson solver.

Console script (installed via pyproject):
    chg2pot chgden.cube avg_dir[x:1|y:2|z:3]

Writes ``pot.cube`` (potential, Ha units) and ``pot.{x,y,z}.avg`` (planar
average), matching the original C++ ``chg2pot`` binary.
"""

from __future__ import annotations

import sys

from cubetools.cube import Cube
from cubetools.average import save_planar_average
from .poisson import chg2pot


def chg2pot_main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("Usage: chg2pot chgden.cube avg_dir[x:1|y:2|z:3]")
        return 0
    chg = Cube.from_file(argv[0])
    print("### Cell parameters(Bohr): %f %f %f" % tuple(chg.cell))
    print("### Number of Atoms: %d" % chg.n_atoms)
    print("### Grid Dimension: %d %d %d" % tuple(int(g) for g in chg.grid))

    direction = int(argv[1])
    pot = chg2pot(chg)
    pot.to_file(
        "pot.cube",
        comment=(" Cubefile created from chg2pot",
                 " ES potential, lattice unit Bohr, grid unit Ha"),
    )
    print("### Potential was saved to pot.cube")

    axis = direction - 1
    name = {0: "pot.x.avg", 1: "pot.y.avg", 2: "pot.z.avg"}[axis]
    save_planar_average(pot, axis, name)
    print(f"### Averaged potential saved: {name}")
    return 0
