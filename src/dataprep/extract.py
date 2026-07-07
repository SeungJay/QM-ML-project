"""Extract DFT single-point results into RuNNer ``.data`` structures.

Per job (one ``calculator-NNN`` directory):
  * coordinates + element names come from the single-point coordinate XYZ
    (``trajectory-input.xyz``), in Angstrom;
  * energy + forces come from the CP2K ``forces-output.xyz`` (energy in the
    comment line, forces in atomic units);
  * the cell is a fixed box (the AIMD ``&CELL``), supplied once as ``cell`` —
    either 3 lengths / a 3x3 matrix (Angstrom), or a path to a CP2K input file
    whose ``&CELL`` is parsed.

Everything is stored in atomic units (positions/cell in Bohr, energy in Hartree,
forces in Hartree/Bohr), matching the RuNNer ``.data`` convention.
"""

from __future__ import annotations

from typing import Optional, Union
import glob
import os

import numpy as np

from .structures import Structure, Structures, Property
from .frames import read_xyz_frames, parse_lattice  # noqa: F401

ANG2BOHR = 1.0 / 0.529177210903  # Bohr per Angstrom

_UNIT2BOHR = {
    "angstrom": ANG2BOHR, "ang": ANG2BOHR, "a": ANG2BOHR,
    "bohr": 1.0, "au": 1.0, "a.u.": 1.0,
    "nm": 10.0 * ANG2BOHR, "pm": 0.01 * ANG2BOHR,
}


# --------------------------------------------------------------------------
# CP2K forces output (energy + forces, atomic units)
# --------------------------------------------------------------------------
def read_cp2k_forces(fname: str):
    """Read a CP2K ``forces-output.xyz`` (single frame).

    Returns ``(energy, forces)`` — energy in Hartree, parsed robustly from the
    comment line (``... E = -2029.6838 ...`` with or without a trailing comma),
    and forces an ``(n, 3)`` array in Hartree/Bohr.
    """
    with open(fname, "r") as fp:
        n = int(fp.readline().split()[0])
        comment = fp.readline()
        energy = float(comment.split("E =")[1].strip().split(",")[0].split()[0])
        forces = []
        for _ in range(n):
            p = fp.readline().split()
            forces.append([float(p[1]), float(p[2]), float(p[3])])
    return energy, np.array(forces)


# --------------------------------------------------------------------------
# XYZ coordinates (single frame, Angstrom)
# --------------------------------------------------------------------------
def read_xyz(fname: str):
    """Read a single-frame XYZ. Returns ``(names, positions, lattice)`` — positions
    in Angstrom and ``lattice`` a ``(3, 3)`` Angstrom cell if the comment is an
    extended-XYZ ``Lattice="..."`` header, else None."""
    names, pos, comment = read_xyz_frames(fname)[0]
    return names, pos, parse_lattice(comment)


# --------------------------------------------------------------------------
# Cell resolution (-> Bohr 3x3)
# --------------------------------------------------------------------------
def _resolve_cell(cell, cell_unit: str = "angstrom") -> np.ndarray:
    if isinstance(cell, str) and os.path.isfile(cell):
        return read_cell_from_cp2k_input(cell)   # already Bohr
    factor = _UNIT2BOHR.get(cell_unit.lower(), ANG2BOHR)
    arr = np.asarray(cell, dtype=float)
    if arr.shape == (3,):
        m = np.diag(arr)
    elif arr.shape == (3, 3):
        m = arr
    else:
        raise ValueError("cell must be 3 lengths, a 3x3 matrix, or a CP2K input path")
    return m * factor


def read_cell_from_cp2k_input(inp_file: str) -> np.ndarray:
    """Parse the ``&CELL`` block of a CP2K input; return a 3x3 cell in Bohr.

    Handles ``ABC [unit] a b c`` and ``A``/``B``/``C`` vectors. Default unit is
    Angstrom (CP2K default)."""
    cell = np.zeros((3, 3))
    in_cell = False
    with open(inp_file, "r") as fp:
        for raw in fp:
            line = raw.split("#")[0].split("!")[0].strip()
            if not line:
                continue
            low = line.lower()
            if low.startswith("&cell") and not low.startswith("&cell_file"):
                in_cell = True
                continue
            if low.startswith("&end"):
                if in_cell:
                    break
                continue
            if not in_cell:
                continue
            toks = line.split()
            key = toks[0].lower()
            rest = toks[1:]
            factor = ANG2BOHR
            if rest and rest[0].startswith("[") and rest[0].endswith("]"):
                factor = _UNIT2BOHR.get(rest[0][1:-1].lower(), ANG2BOHR)
                rest = rest[1:]
            if key == "abc":
                a, b, c = (float(x) for x in rest[:3])
                cell = np.diag([a * factor, b * factor, c * factor])
            elif key in ("a", "b", "c"):
                cell["abc".index(key)] = np.array([float(x) for x in rest[:3]]) * factor
    return cell


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------
def extract_structure(cp2k_forces_file: str, coord_xyz_file: str,
                      cell=None, cell_unit: str = "angstrom") -> Structure:
    """Build one :class:`Structure` from a CP2K forces file and a coordinate XYZ.

    The cell is taken, in order of preference, from (1) the coordinate XYZ's
    extended-XYZ ``Lattice=`` header (per-frame, e.g. NPT), or (2) the ``cell``
    argument (fixed box: 3 lengths / 3x3 Angstrom, or a CP2K input path)."""
    energy, forces = read_cp2k_forces(cp2k_forces_file)
    names, pos_ang, lattice = read_xyz(coord_xyz_file)
    if forces.shape[0] != len(names):
        raise ValueError(
            f"atom-count mismatch: {len(names)} in {coord_xyz_file} vs "
            f"{forces.shape[0]} in {cp2k_forces_file}")
    positions = np.asarray(pos_ang) * ANG2BOHR
    if lattice is not None:
        cell_bohr = np.asarray(lattice) * ANG2BOHR   # ext-xyz Lattice is Angstrom
    elif cell is not None:
        cell_bohr = _resolve_cell(cell, cell_unit)
    else:
        raise ValueError(
            f"no cell for {coord_xyz_file}: the XYZ has no Lattice header and no "
            f"'cell' argument was given")
    struc = Structure(names, positions, cell=cell_bohr)
    struc.properties["reference"] = Property(energy=energy, forces=forces)
    return struc


def extract_dataset(job_dirs, cell=None, cell_unit: str = "angstrom",
                    coord_name: str = "trajectory-input.xyz",
                    forces_name: str = "forces-output.xyz",
                    out_data: Optional[str] = None) -> Structures:
    """Extract many jobs into one :class:`Structures` (like ``grep.sh``).

    ``job_dirs`` : list of directories, or a glob string (e.g. ``"calculator-*"``).
    ``cell`` : fixed cell (3 lengths / 3x3 Angstrom, or a CP2K input path). Not
               needed if the coordinate XYZs carry a per-frame ``Lattice`` header
               (variable-cell / NPT).
    """
    if isinstance(job_dirs, str):
        job_dirs = sorted(glob.glob(job_dirs))
    out = Structures()
    for d in job_dirs:
        out.append(extract_structure(
            os.path.join(d, forces_name),
            os.path.join(d, coord_name),
            cell, cell_unit=cell_unit,
        ))
    if out_data is not None:
        out.to_file(out_data, label_prop="reference")
    return out
