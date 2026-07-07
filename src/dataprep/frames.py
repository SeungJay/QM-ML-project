"""Frame selection from an AIMD trajectory (the ``trj2xyz`` step).

Reads a multi-frame XYZ trajectory (e.g. CP2K MD ``trajectory-output.xyz``),
selects frames (evenly spaced or random), and writes one single-frame XYZ per
selection into an output directory, plus a ``frame_index_map.txt`` recording the
mapping to the original frame indices. Dependency-free (no ASE).

Variable cell (NPT): if a CP2K ``.cell`` trajectory file (or a fixed cell) is
supplied, each written frame is an **extended-XYZ** carrying its own cell in the
comment as ``Lattice="ax ay az bx by bz cx cy cz"`` (Angstrom). Downstream
:func:`dataprep.extract.extract_structure` reads that per-frame cell
automatically. CP2K's XYZ coordinate reader ignores the comment, so these files
are still valid coordinate inputs.
"""

from __future__ import annotations

from typing import Optional
import os
import random
import re

import numpy as np

_LATTICE_RE = re.compile(r'Lattice="([^"]+)"', re.IGNORECASE)


# --------------------------------------------------------------------------
# XYZ (multi-frame) and CP2K .cell readers
# --------------------------------------------------------------------------
def read_xyz_frames(path: str):
    """Read a multi-frame XYZ. Returns a list of ``(names, positions, comment)``
    where ``positions`` is an ``(n, 3)`` array (Angstrom)."""
    frames = []
    with open(path, "r") as f:
        while True:
            header = f.readline()
            if not header:
                break
            if not header.strip():
                continue
            n = int(header.split()[0])
            comment = f.readline().rstrip("\n")
            names, pos = [], []
            for _ in range(n):
                p = f.readline().split()
                names.append(p[0])
                pos.append([float(p[1]), float(p[2]), float(p[3])])
            frames.append((names, np.array(pos), comment))
    return frames


def read_cell_trajectory(cell_file: str):
    """Read a CP2K ``*.cell`` file. Returns a list of ``(3, 3)`` cell matrices
    (Angstrom), one per printed MD step.

    Column layout: ``step  time  Ax Ay Az  Bx By Bz  Cx Cy Cz  volume``.
    """
    cells = []
    with open(cell_file, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            vals = [float(x) for x in s.split()]
            cells.append(np.array(vals[2:11]).reshape(3, 3))
    return cells


def parse_lattice(comment: str):
    """Return a ``(3, 3)`` lattice (Angstrom) from an ext-xyz comment, or None."""
    m = _LATTICE_RE.search(comment or "")
    if not m:
        return None
    vals = [float(x) for x in m.group(1).split()]
    return np.array(vals).reshape(3, 3)


def _cell_to_3x3(cell):
    arr = np.asarray(cell, dtype=float)
    if arr.shape == (3,):
        return np.diag(arr)
    if arr.shape == (3, 3):
        return arr
    raise ValueError("cell must be 3 lengths or a 3x3 matrix")


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------
def write_xyz(path: str, names, positions, comment: str = "",
              lattice=None) -> None:
    """Write a single-frame XYZ. If ``lattice`` (3x3 Angstrom) is given, the
    comment becomes an extended-XYZ header carrying the cell."""
    if lattice is not None:
        lat = np.asarray(lattice, dtype=float).reshape(9)
        comment = ('Lattice="' + " ".join(f"{v:.10f}" for v in lat) + '" '
                   'Properties=species:S:1:pos:R:3')
    with open(path, "w") as f:
        f.write(f"{len(names)}\n{comment}\n")
        for nm, (x, y, z) in zip(names, positions):
            f.write(f"{nm:<4s} {x:18.10f} {y:18.10f} {z:18.10f}\n")


# --------------------------------------------------------------------------
# Frame selection
# --------------------------------------------------------------------------
def select_frames(traj_xyz: str, outdir: str = "trajectory_inputs",
                  mode: str = "every", every: int = 20,
                  n_random: int = 140, seed: int = 77,
                  skip_equil: int = 0,
                  prefix: str = "trajectory-input",
                  cell_file: Optional[str] = None,
                  cell=None) -> list:
    """Select frames from a multi-frame XYZ trajectory and write them out.

    Parameters
    ----------
    traj_xyz : multi-frame XYZ (e.g. CP2K MD ``trajectory-output.xyz``).
    outdir : output directory for ``<prefix>NNN.xyz`` files.
    mode : ``"every"`` (evenly spaced by ``every``) or ``"random"``.
    skip_equil : drop this many leading (equilibration) frames.
    cell_file : optional CP2K ``.cell`` trajectory (per-frame cell, NPT). Aligned
                by index with the coordinate frames.
    cell : optional fixed cell (3 lengths or 3x3, Angstrom) used when there is no
           ``cell_file``. If neither is given, plain XYZ is written (extract then
           needs a ``cell`` argument).

    Returns the list of selected original frame indices and writes
    ``<outdir>/frame_index_map.txt``.
    """
    frames = read_xyz_frames(traj_xyz)
    ntot = len(frames)
    cell_traj = read_cell_trajectory(cell_file) if cell_file else None
    fixed = _cell_to_3x3(cell) if (cell is not None and cell_file is None) else None

    avail = list(range(skip_equil, ntot))
    if mode == "every":
        picked = avail[::every]
    elif mode == "random":
        if len(avail) < n_random:
            raise ValueError(
                f"available frames ({len(avail)}) < n_random ({n_random})")
        random.seed(seed)
        picked = sorted(random.sample(avail, n_random))
    else:
        raise ValueError("mode must be 'every' or 'random'")

    os.makedirs(outdir, exist_ok=True)
    index_map = []
    for n, idx in enumerate(picked):
        names, pos, _ = frames[idx]
        if cell_traj is not None:
            lat = cell_traj[idx]
        else:
            lat = fixed
        write_xyz(os.path.join(outdir, f"{prefix}{n:03d}.xyz"), names, pos,
                  lattice=lat)
        index_map.append((n, idx))

    with open(os.path.join(outdir, "frame_index_map.txt"), "w") as f:
        f.write("# output_index   original_frame_index\n")
        for n, idx in index_map:
            f.write(f"{n:03d}   {idx}\n")

    kind = "variable-cell" if cell_traj is not None else (
        "fixed-cell" if fixed is not None else "no-cell")
    print(f"selected {len(picked)} / {ntot} frames ({kind}) -> "
          f"{outdir}/{prefix}000.xyz ...")
    return picked
