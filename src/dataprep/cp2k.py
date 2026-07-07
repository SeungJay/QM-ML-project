"""Generic CP2K single-point calculator (energy + forces) and job helpers.

Note: the production charge-decoupling coulomb term is the coupled CP2K<->LAMMPS
DFT-CES calculation in :mod:`dataprep.dftces` (:func:`~dataprep.dftces.run_coulomb`),
not this single CP2K run. This ``CP2K`` class remains a convenient dependency-free
single-point runner (a stand-in for ``aml.CP2K``). Per structure it:
  1. writes ``coord.xyz`` (positions in Angstrom),
  2. builds a CP2K input from a template, injecting the cell (A/B/C vectors,
     Angstrom) into ``&CELL`` and a ``&TOPOLOGY`` pointing at ``coord.xyz``,
  3. runs CP2K through a :class:`~dataprep.launcher.ProcessLauncher`,
  4. parses the total energy and atomic forces (atomic units) from the CP2K
     output log and stores them as a :class:`~dataprep.structures.Property`.

Assumptions about the template: it is a complete CP2K input whose ``&GLOBAL``
uses ``RUN_TYPE ENERGY_FORCE`` (so the log contains the ``ENERGY|`` line and an
``ATOMIC FORCES`` table), and it has a ``&SUBSYS`` with a ``&CELL`` block. The
user's ``coulomb.inp`` satisfies this.
"""

from __future__ import annotations

from typing import Optional, Union
import os
import re
import shlex
import shutil

import numpy as np

import glob

from .structures import Structure, Structures, Property
from .launcher import ProcessLauncher
from .frames import read_xyz_frames, parse_lattice

ANG2BOHR = 1.0 / 0.529177210903  # Bohr per Angstrom


# --------------------------------------------------------------------------
# Geometry / input writing
# --------------------------------------------------------------------------
def write_coord_xyz(path: str, names, positions_bohr) -> None:
    """Write an XYZ file with positions converted Bohr -> Angstrom."""
    pos_ang = np.asarray(positions_bohr, dtype=float) / ANG2BOHR
    with open(path, "w") as f:
        f.write(f"{len(names)}\n\n")
        for nm, (x, y, z) in zip(names, pos_ang):
            f.write(f"{nm:<4s} {x:16.8f} {y:16.8f} {z:16.8f}\n")


def set_cell(template: str, cell_ang) -> str:
    """Replace the ``&CELL`` contents with A/B/C vectors (Angstrom).

    Existing ``A``/``B``/``C``/``ABC``/``CELL_FILE_NAME`` lines inside ``&CELL``
    are removed and replaced. Everything else (including any ``&TOPOLOGY``) is
    left untouched. Case-insensitive; original indentation is preserved. Use this
    to give each frame its own cell (e.g. NPT single-point inputs)."""
    cell_ang = np.asarray(cell_ang, dtype=float).reshape(3, 3)
    out = []
    in_cell = False
    for line in template.splitlines():
        low = line.strip().lower()
        indent = line[: len(line) - len(line.lstrip())]
        if low.startswith("&cell") and not low.startswith("&cell_file"):
            in_cell = True
            out.append(line)
            ai = indent + "  "
            for lbl, vec in zip("ABC", cell_ang):
                out.append(f"{ai}{lbl} {vec[0]:.10f} {vec[1]:.10f} {vec[2]:.10f}")
            continue
        if in_cell:
            if low.startswith("&end"):
                in_cell = False
                out.append(line)
                continue
            key = low.split()[0] if low.split() else ""
            if key in ("a", "b", "c", "abc", "cell_file_name"):
                continue
            out.append(line)
            continue
        out.append(line)
    return "\n".join(out) + "\n"


def inject_cell_and_topology(template: str, cell_ang, coord_file: str = "coord.xyz",
                             coord_format: str = "XYZ") -> str:
    """Set the ``&CELL`` (via :func:`set_cell`) and insert a ``&TOPOLOGY``
    (pointing at ``coord_file``) at the start of ``&SUBSYS``. Used by the coulomb
    calculator, whose template supplies coordinates via an external file."""
    text = set_cell(template, cell_ang)
    out = []
    for line in text.splitlines():
        out.append(line)
        low = line.strip().lower()
        if low.startswith("&subsys"):
            indent = line[: len(line) - len(line.lstrip())]
            si = indent + "  "
            out.append(f"{si}&TOPOLOGY")
            out.append(f"{si}  COORD_FILE_NAME {coord_file}")
            out.append(f"{si}  COORD_FILE_FORMAT {coord_format}")
            out.append(f"{si}&END TOPOLOGY")
    return "\n".join(out) + "\n"


def run_aimd(input_file: str = "aimd.inp", cmd_cp2k: str = "cp2k.psmp",
             output: str = "aimd.log", directory: str = ".",
             launcher: Optional[ProcessLauncher] = None) -> str:
    """Run one CP2K job (e.g. an AIMD) in ``directory``, writing CP2K's stdout to
    ``output``. Intended to be launched *inside* a SLURM allocation via a
    ``ProcessLauncher`` whose ``mode`` emits an ``srun -n N`` / ``mpirun -np N``
    prefix. Returns the log path."""
    launcher = launcher or ProcessLauncher()
    cmd = shlex.split(cmd_cp2k) + ["-i", input_file]
    r = launcher.run([cmd], directory, check=False)[0]
    log = os.path.join(directory, output)
    with open(log, "w") as f:
        f.write(r.stdout)
    if r.returncode != 0:
        print(f"warning: CP2K exited {r.returncode} in {directory} "
              f"(see {output})\n  stderr: {r.stderr[-1000:]}")
    return log


def run_singlepoints(job_dirs="calculator-*", input_file: str = "step-0.inp",
                     cmd_cp2k: str = "cp2k.psmp", output: str = "step-0.log",
                     forces_name: str = "forces-output.xyz",
                     skip_existing: bool = True,
                     launcher: Optional[ProcessLauncher] = None) -> list:
    """Run a CP2K single point in each job directory (like the ``slrm.sh`` loop).

    ``job_dirs`` : list of dirs or a glob (e.g. ``"calculator-*"``).
    ``input_file`` : resolved relative to each job dir — per-dir ``"step-0.inp"``
                     (from :func:`prepare_singlepoint_jobs`) or a shared
                     ``"../step-0.inp"``.
    ``skip_existing`` : skip dirs that already have ``forces_name`` (resume).

    Runs sequentially; use ``launcher`` with an ``srun``/``mpirun`` mode so each
    job uses the SLURM allocation. Returns the dirs that were run.
    """
    launcher = launcher or ProcessLauncher()
    if isinstance(job_dirs, str):
        job_dirs = sorted(glob.glob(job_dirs))
    done = []
    for d in job_dirs:
        if skip_existing and os.path.isfile(os.path.join(d, forces_name)):
            print(f"skip {d} (has {forces_name})")
            continue
        cmd = shlex.split(cmd_cp2k) + ["-i", input_file]
        r = launcher.run([cmd], d, check=False)[0]
        with open(os.path.join(d, output), "w") as f:
            f.write(r.stdout)
        if r.returncode != 0:
            print(f"warning: CP2K exited {r.returncode} in {d} (see {output})")
        done.append(d)
    return done


def prepare_singlepoint_jobs(frames_dir: str, template: str,
                             jobs_dir: str = ".", job_prefix: str = "calculator",
                             frame_prefix: str = "trajectory-input",
                             coord_name: str = "trajectory-input.xyz",
                             input_name: str = "step-0.inp") -> list:
    """Create per-frame single-point CP2K job directories.

    For each ``<frames_dir>/<frame_prefix>NNN.xyz`` (from :func:`select_frames`),
    make ``<jobs_dir>/<job_prefix>-NNN/`` containing the coordinate XYZ and a CP2K
    input built from ``template``. If the frame's XYZ carries a ``Lattice=``
    header (variable cell / NPT), that per-frame cell is written into the input's
    ``&CELL`` via :func:`set_cell`; otherwise the template's cell is used as-is.

    The template must already reference the coordinate file (its ``&TOPOLOGY
    COORD_FILE_NAME`` should be ``coord_name``). Returns the job directories.
    """
    if os.path.isfile(template):
        with open(template) as f:
            template_text = f.read()
    else:
        template_text = template

    files = sorted(glob.glob(os.path.join(frames_dir, f"{frame_prefix}*.xyz")))
    jobs = []
    for f in files:
        num = os.path.basename(f)[len(frame_prefix):-len(".xyz")]
        jd = os.path.join(jobs_dir, f"{job_prefix}-{num}")
        os.makedirs(jd, exist_ok=True)
        shutil.copyfile(f, os.path.join(jd, coord_name))

        _, _, comment = read_xyz_frames(f)[0]
        lattice = parse_lattice(comment)
        inp = set_cell(template_text, lattice) if lattice is not None else template_text
        with open(os.path.join(jd, input_name), "w") as out:
            out.write(inp)
        jobs.append(jd)
    return jobs


def prepare_jobs_from_data(data, template: str, jobs_dir: str = ".",
                           job_prefix: str = "calculator",
                           coord_name: str = "trajectory-input.xyz",
                           input_name: str = "step-0.inp") -> list:
    """Build per-structure CP2K job dirs directly from a ``.data`` dataset.

    For each structure in ``data`` (a :class:`~dataprep.structures.Structures` or
    a ``.data`` path) make ``<jobs_dir>/<job_prefix>-NNN/`` containing the
    coordinate XYZ (positions Bohr -> Angstrom) and a CP2K input built from
    ``template`` with that structure's cell written into ``&CELL``.

    This lets a stage build its own job dirs from ``input-SR.data`` alone (no need
    to carry ``trajectory_inputs/`` forward) — used by the charge-decouple stage.
    """
    from .structures import Structures
    if isinstance(data, str):
        data = Structures.from_file(data)
    if os.path.isfile(template):
        with open(template) as f:
            template_text = f.read()
    else:
        template_text = template

    width = max(3, len(str(len(data) - 1)))
    jobs = []
    for i, s in enumerate(data):
        jd = os.path.join(jobs_dir, f"{job_prefix}-{i:0{width}d}")
        os.makedirs(jd, exist_ok=True)
        write_coord_xyz(os.path.join(jd, coord_name), s.names, s.positions)
        cell_ang = s.cell / ANG2BOHR                      # Bohr -> Angstrom
        with open(os.path.join(jd, input_name), "w") as out:
            out.write(set_cell(template_text, cell_ang))
        jobs.append(jd)
    return jobs


# --------------------------------------------------------------------------
# Output parsing
# --------------------------------------------------------------------------
_ENERGY_RE = re.compile(
    r"ENERGY\|\s*Total FORCE_EVAL.*?:\s*(-?\d+\.\d+)", re.IGNORECASE)


def parse_energy_forces(log_text: str, n_atoms: int):
    """Parse total energy (Hartree) and atomic forces (Hartree/Bohr, shape
    (n_atoms, 3)) from a CP2K ENERGY_FORCE output log."""
    m = list(_ENERGY_RE.finditer(log_text))
    if not m:
        raise ValueError("could not find 'ENERGY| Total FORCE_EVAL' in CP2K log")
    energy = float(m[-1].group(1))

    lines = log_text.splitlines()
    forces = None
    for i, line in enumerate(lines):
        if "ATOMIC FORCES in" in line:
            rows = []
            for row in lines[i + 1:]:
                s = row.strip()
                if not s or s.startswith("#"):
                    continue
                if s.upper().startswith("SUM OF ATOMIC FORCES"):
                    break
                parts = row.split()
                if len(parts) < 6:
                    continue
                # columns: Atom Kind Element X Y Z  -> forces are last 3
                rows.append([float(parts[-3]), float(parts[-2]), float(parts[-1])])
                if len(rows) == n_atoms:
                    break
            forces = np.array(rows)
            break
    if forces is None or forces.shape[0] != n_atoms:
        raise ValueError(
            f"could not parse {n_atoms} force rows from 'ATOMIC FORCES' table")
    return energy, forces


# --------------------------------------------------------------------------
# Calculator
# --------------------------------------------------------------------------
class CP2K:
    def __init__(self, template: str, cmd_cp2k: str = "cp2k.psmp",
                 directory: str = "cp2k_calc", keep_directories: bool = False,
                 launcher: Optional[ProcessLauncher] = None,
                 label: str = "calc", coord_format: str = "XYZ"):
        """
        template : path to a CP2K input template, or the template text itself.
        cmd_cp2k : CP2K executable (run via the launcher).
        directory : base directory for per-structure run folders.
        keep_directories : keep the per-structure folders after parsing.
        launcher : a ProcessLauncher (default: serial, no MPI prefix).
        """
        if os.path.isfile(template):
            with open(template) as f:
                self.template_text = f.read()
        else:
            self.template_text = template
        self.cmd_cp2k = cmd_cp2k
        self.directory = directory
        self.keep_directories = keep_directories
        self.launcher = launcher or ProcessLauncher()
        self.label = label
        self.coord_format = coord_format

    def run(self, structures: Union[Structures, list],
            label_prop: str = "coulomb") -> Structures:
        """Run a CP2K single point for each structure; store results under
        ``label_prop`` in each structure's ``.properties``."""
        os.makedirs(self.directory, exist_ok=True)
        width = max(4, len(str(len(structures))))
        for i, struc in enumerate(structures):
            d = os.path.join(self.directory, f"{self.label}-{i:0{width}d}")
            os.makedirs(d, exist_ok=True)

            write_coord_xyz(os.path.join(d, "coord.xyz"),
                            struc.names, struc.positions)
            cell_ang = struc.cell / ANG2BOHR
            inp = inject_cell_and_topology(
                self.template_text, cell_ang, "coord.xyz", self.coord_format)
            with open(os.path.join(d, "step-0.inp"), "w") as f:
                f.write(inp)

            cmd = shlex.split(self.cmd_cp2k) + ["-i", "step-0.inp", "-o", "step-0.log"]
            self.launcher.run([cmd], d)

            with open(os.path.join(d, "step-0.log")) as f:
                log = f.read()
            energy, forces = parse_energy_forces(log, struc.n_atoms)
            struc.properties[label_prop] = Property(energy, forces)

            if not self.keep_directories:
                shutil.rmtree(d, ignore_errors=True)
        return structures
