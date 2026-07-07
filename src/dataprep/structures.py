"""Atomic structures and n2p2/RuNNer ``.data`` I/O.

A dependency-free reimplementation of the small subset of the old ``aml``
package needed for the QM/ML data-prep workflow (DFT extraction and charge
decoupling).

Units: everything is stored in **atomic units** (positions in Bohr, energy in
Hartree, forces in Hartree/Bohr), matching the RuNNer ``.data`` convention. The
``.data`` reader/writer perform **no** unit conversion — values round-trip
exactly as stored. (Conversion to Angstrom happens only when writing CP2K
input; see :mod:`dataprep.cp2k`.)

``.data`` frame grammar::

    begin
    comment <optional free text>
    lattice  ax ay az            # 3 rows = cell vectors
    lattice  bx by bz
    lattice  cx cy cz
    atom  x y z  <element>  q  n  fx fy fz   # q (charge) and n (atomic E) ignored
    ...
    energy  <E>
    charge  <Q>                  # ignored on read
    end
"""

from __future__ import annotations

from typing import Optional
import numpy as np


# --------------------------------------------------------------------------
# Property: an (energy, forces) pair for one structure under one label
# --------------------------------------------------------------------------
class Property:
    """Energy (scalar) and forces (n, 3) for a structure.

    ``.energy`` / ``.forces`` are read accessors (``forces`` returns a
    read-only view). The mutable backing fields ``_energy`` / ``_forces`` are
    exposed so callers can do in-place arithmetic, e.g. ``p._forces -= q.forces``.
    """

    __slots__ = ("_energy", "_forces")

    def __init__(self, energy: Optional[float] = None, forces=None):
        self._energy = None if energy is None else float(energy)
        self._forces = None if forces is None else np.asarray(forces, dtype=float)

    @property
    def energy(self):
        return self._energy

    @property
    def forces(self):
        if self._forces is None:
            return None
        v = self._forces.view()
        v.flags.writeable = False
        return v

    def __repr__(self):
        e = "None" if self._energy is None else f"{self._energy:.6f}"
        nf = "None" if self._forces is None else f"{self._forces.shape}"
        return f"Property(energy={e}, forces={nf})"


# --------------------------------------------------------------------------
# Structure: one atomic configuration
# --------------------------------------------------------------------------
class Structure:
    __slots__ = ("names", "positions", "cell", "comment", "properties")

    def __init__(self, names, positions, cell=None, comment=None,
                 properties=None):
        self.names = tuple(names)
        self.positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        self.cell = None if cell is None else np.asarray(cell, dtype=float).reshape(3, 3)
        self.comment = comment
        self.properties = dict(properties) if properties else {}

    @property
    def n_atoms(self) -> int:
        return len(self.names)

    def __repr__(self):
        return f"Structure({self.n_atoms} atoms)"


# --------------------------------------------------------------------------
# Structures: an ordered collection
# --------------------------------------------------------------------------
class Structures(list):
    """A list of :class:`Structure` with ``.data`` file I/O."""

    # ---- reading -------------------------------------------------------
    @classmethod
    def from_file(cls, fname: str, label_prop: str = "reference") -> "Structures":
        """Read a RuNNer ``.data`` file. Energy/forces are stored under
        ``label_prop`` (default ``"reference"``)."""
        out = cls()
        with open(fname, "r") as fp:
            while True:
                struc = _read_frame(fp, label_prop)
                if struc is None:
                    break
                out.append(struc)
        return out

    # ---- writing -------------------------------------------------------
    def to_file(self, fname: str, label_prop: Optional[str] = None) -> None:
        """Write a RuNNer ``.data`` file. If ``label_prop`` is given, that
        property's energy/forces are written; otherwise zeros are written."""
        with open(fname, "w") as fp:
            for struc in self:
                _write_frame(fp, struc, label_prop)


# --------------------------------------------------------------------------
# Frame-level read / write (no unit conversion)
# --------------------------------------------------------------------------
_FMT_LATTICE = "lattice " + 3 * "{:16.6f}" + "\n"
_FMT_F = "{:13.6f}"
_FMT_ATOM = "atom " + 3 * _FMT_F + "{:^6s}" + 5 * _FMT_F + "\n"
_FMT_ENERGY = "energy " + _FMT_F + "\n"
_FMT_CHARGE = "charge " + _FMT_F + "\n"


def _read_frame(fp, label_prop: str):
    """Read one ``begin..end`` frame; return a Structure or None at EOF."""
    # find 'begin'
    while True:
        line = fp.readline()
        if not line:
            return None
        if line.strip() == "begin":
            break

    names, positions, forces, cell_rows = [], [], [], []
    comment = None
    energy = None
    have_forces = False

    while True:
        line = fp.readline()
        if not line:
            raise ValueError("Unexpected EOF inside a frame (no 'end').")
        items = line.split()
        if not items:
            continue
        tag = items[0]
        if tag == "comment":
            comment = " ".join(items[1:])
        elif tag == "lattice":
            cell_rows.append([float(x) for x in items[1:4]])
        elif tag == "atom":
            positions.append([float(items[1]), float(items[2]), float(items[3])])
            names.append(items[4])
            # items[5] = atomic charge q, items[6] = atomic energy n  (ignored)
            fx, fy, fz = float(items[7]), float(items[8]), float(items[9])
            forces.append([fx, fy, fz])
            if fx or fy or fz:
                have_forces = True
        elif tag == "energy":
            energy = float(items[1])
        elif tag == "charge":
            pass  # total charge ignored
        elif tag == "end":
            break
        else:
            raise ValueError(f"Unexpected data in .data file: {line!r}")

    if not names:
        raise ValueError("No atomic data in frame.")
    cell = np.array(cell_rows) if cell_rows else None

    props = {}
    if energy is not None or have_forces:
        props[label_prop] = Property(
            energy=energy,
            forces=np.array(forces) if have_forces else None,
        )

    return Structure(names, positions, cell=cell, comment=comment,
                     properties=props)


def _write_frame(fp, struc: Structure, label_prop: Optional[str]) -> None:
    energy = None
    forces = None
    if label_prop is not None and label_prop in struc.properties:
        prop = struc.properties[label_prop]
        energy = prop.energy
        forces = prop.forces

    fp.write("begin\n")
    if struc.comment:
        fp.write(f"comment {struc.comment}\n")
    if struc.cell is not None:
        for row in struc.cell:
            fp.write(_FMT_LATTICE.format(*row))
    for i, name in enumerate(struc.names):
        x, y, z = struc.positions[i]
        if forces is not None:
            fx, fy, fz = forces[i]
        else:
            fx = fy = fz = 0.0
        # columns after element: q=0, n=0, fx, fy, fz
        fp.write(_FMT_ATOM.format(x, y, z, name, 0.0, 0.0, fx, fy, fz))
    fp.write(_FMT_ENERGY.format(0.0 if energy is None else energy))
    fp.write(_FMT_CHARGE.format(0.0))
    fp.write("end\n")
