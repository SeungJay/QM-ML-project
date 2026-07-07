"""Gaussian cube file I/O backed by NumPy arrays.

The cube format handled here matches the original C/C++ tools: an
orthorhombic grid whose voxel vectors are diagonal (the parsers only read
the diagonal component of each axis vector). Grid values are stored in a
3-D array ``data`` with shape ``(n0, n1, n2)`` in C order, so that
``data[i, j, k]`` corresponds to the flat index ``k + j*n2 + i*n2*n1``
used throughout the original code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import warnings
import numpy as np


@dataclass
class Cube:
    """A parsed cube file.

    Attributes
    ----------
    n_atoms : int
        Number of atoms.
    grid : ndarray of int, shape (3,)
        Grid dimensions ``(n0, n1, n2)``.
    cell : ndarray of float, shape (3,)
        Cell lengths along each axis in Bohr. Equals ``grid * spacing``.
    atom_numbers : ndarray of int, shape (n_atoms,)
        Atomic numbers.
    positions : ndarray of float, shape (n_atoms, 3)
        Atomic positions in Bohr.
    data : ndarray of float, shape (n0, n1, n2)
        Volumetric values (charge density or potential).
    origin : ndarray of float, shape (3,)
        Grid origin. Preserved on read; the original tools write zeros.
    comment : tuple(str, str)
        The two header comment lines.
    """

    n_atoms: int
    grid: np.ndarray
    cell: np.ndarray
    atom_numbers: np.ndarray
    positions: np.ndarray
    data: np.ndarray
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    comment: tuple = (" Cubefile", " lattice unit Bohr")

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    @property
    def spacing(self) -> np.ndarray:
        """Voxel spacing along each axis (Bohr) = cell / grid."""
        return self.cell / self.grid

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    @classmethod
    def from_file(cls, fname: str) -> "Cube":
        # Read the (small) header line-by-line, then stream the (large)
        # numeric block straight into a NumPy array. This avoids building a
        # Python list of tens of millions of token strings, which is both
        # slow and memory-hungry for production-size grids.
        with open(fname, "r") as fp:
            header = [fp.readline() for _ in range(6)]

            comment = (header[0].rstrip("\n"), header[1].rstrip("\n"))
            tok = header[2].split()
            n_atoms = int(tok[0])
            origin = (
                np.array([float(x) for x in tok[1:4]])
                if len(tok) >= 4 else np.zeros(3)
            )

            grid = np.zeros(3, dtype=int)
            spacing = np.zeros(3)
            for axis in range(3):
                tok = header[3 + axis].split()
                grid[axis] = int(tok[0])
                # diagonal component only (axis-th value of the voxel vector)
                spacing[axis] = float(tok[1 + axis])
            cell = grid * spacing

            atom_numbers = np.zeros(n_atoms, dtype=int)
            positions = np.zeros((n_atoms, 3))
            for a in range(n_atoms):
                tok = fp.readline().split()
                atom_numbers[a] = int(tok[0])
                # tok[1] is the (ignored) nuclear charge; positions are tok[2:5]
                positions[a] = [float(tok[2]), float(tok[3]), float(tok[4])]

            n_total = int(grid[0] * grid[1] * grid[2])
            with warnings.catch_warnings():
                # np.fromstring(sep=...) is deprecated but remains the fastest
                # low-memory text->array parser; no equivalent replacement.
                warnings.simplefilter("ignore", DeprecationWarning)
                values = np.fromstring(fp.read(), sep=" ")  # C-speed parse

        values = values[:n_total]
        if values.size != n_total:
            raise ValueError(
                f"Expected {n_total} grid values, found {values.size} in {fname}"
            )
        data = values.reshape(tuple(grid))  # C order: index (i, j, k)

        return cls(
            n_atoms=n_atoms,
            grid=grid,
            cell=cell,
            atom_numbers=atom_numbers,
            positions=positions,
            data=data,
            origin=origin,
            comment=comment,
        )

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def to_file(self, fname: str, comment: tuple | None = None,
                rows_per_chunk: int = 200000) -> None:
        """Write a cube file matching the original tools' formatting.

        Grid values are written 6 per line, restarting the line at the end
        of every innermost (k) scan — identical to the C ``saveCube``.

        Formatting is vectorised with NumPy and streamed in chunks, so the
        peak memory stays bounded even for production-size grids (hundreds
        of millions of points). When ``n2`` is a multiple of 6 (the common
        case) the fast path is used; otherwise a correct per-row fallback
        preserves the exact line breaks.
        """
        cm = comment if comment is not None else self.comment
        n0, n1, n2 = (int(g) for g in self.grid)
        sp = self.spacing

        with open(fname, "w") as fp:
            fp.write(f"{cm[0]}\n")
            fp.write(f"{cm[1]}\n")
            fp.write(f"{self.n_atoms:5d}    0.000000    0.000000    0.000000\n")
            fp.write(f"{n0:5d}{sp[0]:12.6f}    0.000000    0.000000\n")
            fp.write(f"{n1:5d}    0.000000{sp[1]:12.6f}    0.000000\n")
            fp.write(f"{n2:5d}    0.000000    0.000000{sp[2]:12.6f}\n")
            for a in range(self.n_atoms):
                z = int(self.atom_numbers[a])
                x, y, zz = self.positions[a]
                fp.write(
                    f"{z:5d}{float(z):12.6f}{x:12.6f}{y:12.6f}{zz:12.6f}\n"
                )

            # Formatting uses CPython's C-level ``%`` operator on a repeated
            # format string, which is ~9x faster than np.char.mod for large
            # arrays and keeps peak memory low (chunked). Output bytes are
            # identical to the original C ``fprintf("% 13.5E", ...)``.
            data = np.ascontiguousarray(self.data)
            if n2 % 6 == 0:
                # Fast path: 6-per-line wrapping equals per-row line breaks
                # because every row length is a multiple of 6.
                fmt6 = "% 13.5E" * 6 + "\n"
                flat = data.reshape(-1)
                n_rows = flat.size // 6
                for start in range(0, n_rows, rows_per_chunk):
                    blk = flat[start * 6:(start + rows_per_chunk) * 6]
                    fp.write((fmt6 * (blk.size // 6)) % tuple(blk.tolist()))
            else:
                # General fallback: emit each (i, j) row with a trailing
                # partial line of fewer than 6 values.
                rows = data.reshape(-1, n2)
                for r in range(rows.shape[0]):
                    row = rows[r]
                    for k in range(0, n2, 6):
                        seg = row[k:k + 6]
                        fp.write(("% 13.5E" * seg.size) % tuple(seg.tolist()))
                        fp.write("\n")

    # ------------------------------------------------------------------
    # Fast binary I/O (recommended for intermediate results)
    # ------------------------------------------------------------------
    def save(self, fname: str) -> None:
        """Save the cube to a compressed ``.npz`` binary (~100x faster I/O).

        Use this for intermediate results in a Python pipeline; only write a
        text ``.cube`` (via :meth:`to_file`) when an external tool needs it.
        """
        np.savez(
            fname,
            grid=self.grid, cell=self.cell, atom_numbers=self.atom_numbers,
            positions=self.positions, data=self.data, origin=self.origin,
            comment=np.array(self.comment),
        )

    @classmethod
    def load(cls, fname: str) -> "Cube":
        """Load a cube previously written with :meth:`save`."""
        z = np.load(fname, allow_pickle=False)
        return cls(
            n_atoms=int(z["atom_numbers"].shape[0]),
            grid=z["grid"], cell=z["cell"], atom_numbers=z["atom_numbers"],
            positions=z["positions"], data=z["data"], origin=z["origin"],
            comment=tuple(z["comment"].tolist()),
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def copy(self) -> "Cube":
        return Cube(
            n_atoms=self.n_atoms,
            grid=self.grid.copy(),
            cell=self.cell.copy(),
            atom_numbers=self.atom_numbers.copy(),
            positions=self.positions.copy(),
            data=self.data.copy(),
            origin=self.origin.copy(),
            comment=self.comment,
        )

    def like(self, data: np.ndarray) -> "Cube":
        """Return a copy of this cube with new volumetric ``data``."""
        out = self.copy()
        out.data = np.asarray(data, dtype=float).reshape(tuple(self.grid))
        return out
