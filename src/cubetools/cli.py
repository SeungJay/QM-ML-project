"""Command-line entry points, preserving the original tools' usage.

Console scripts (installed via pyproject):
    cube_add   cube_sub   cube_multi   cube_sigmoid   cube_avg   dipc

(The ``chg2pot`` console script lives in the ``poisson`` package.)

They accept the same positional arguments as the C/C++ binaries and write
output files with the same default names.
"""

from __future__ import annotations

import sys

from .cube import Cube
from . import operations, dipole, average


def _info(cube: Cube) -> None:
    print("### Cell parameters(Bohr): %f %f %f" % tuple(cube.cell))
    print("### Number of Atoms: %d" % cube.n_atoms)
    print("### Grid Dimension: %d %d %d" % tuple(int(g) for g in cube.grid))


def cube_avg_main(argv=None) -> int:
    """Planar average along an axis. Mirrors the C ``cube_avg`` tool.

    Writes ``cube.{x,y,z}.avg`` (two columns: coordinate, in-plane average).
    """
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("Usage: cube_avg *.cube avg_dir[x=1|y=2|z=3]")
        return 0
    cube = Cube.from_file(argv[0])
    _info(cube)
    axis = int(argv[1]) - 1
    name = {0: "cube.x.avg", 1: "cube.y.avg", 2: "cube.z.avg"}[axis]
    average.save_planar_average(cube, axis, name)
    print(f"Averaged values were saved: {name}")
    return 0




def cube_add_main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("for cube1+cube2, Usage: cube_add cube1 cube2")
        return 0
    a, b = Cube.from_file(argv[0]), Cube.from_file(argv[1])
    operations.add(a, b).to_file("add.cube")
    print("### New cubefile has been saved as add.cube")
    return 0


def cube_sub_main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("for cube1-cube2, Usage: cube_sub cube1 cube2")
        return 0
    a, b = Cube.from_file(argv[0]), Cube.from_file(argv[1])
    operations.subtract(a, b).to_file("subtracted.cube")
    print("### New cubefile has been saved as subtracted.cube")
    return 0


def cube_multi_main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("Usage: cube_multi cubefile multiplier")
        return 0
    a = Cube.from_file(argv[0])
    operations.multiply(a, float(argv[1])).to_file("multiplied.cube")
    print("### New cubefile has been saved as multiplied.cube")
    return 0


def cube_sigmoid_main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 5:
        print("Usage: cube_sigmoid cubefile target_pot z_ini z_fin steepness")
        return 0
    a = Cube.from_file(argv[0])
    out = operations.sigmoid_profile(
        a,
        final_value=float(argv[1]),
        z_ini=float(argv[2]),
        z_fin=float(argv[3]),
        steepness=float(argv[4]),
    )
    out.to_file("z_sig.cube")
    print("### New cubefile has been saved as z_sig.cube")
    return 0


def dipc_main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3:
        print("Usage: dipc chgden.cube dir[x:1|y:2|z:3] position(grid#)")
        return 0
    a = Cube.from_file(argv[0])
    _info(a)
    axis = int(argv[1]) - 1
    position = int(argv[2])
    dipole.dipole_correction(a, axis, position).to_file(
        "dipc.cube",
        comment=(" Cubefile created from dipc",
                 " ES potential, lattice unit Bohr, grid unit Ry"),
    )
    print("### Potential was saved to dipc.cube")
    return 0
