"""DFT-CES coulomb term: the electrostatic energy/forces to be decoupled.

This reproduces the production ``run.sh`` procedure — for each structure, the
electrostatic (coulomb) term is computed by a coupled CP2K <-> LAMMPS DFT-CES
calculation, not a single CP2K single point. Per structure (7 steps):

  1. build inputs   (create_dftces_inputs): from the single-point CP2K input,
     make cp2k.inp (external potential ON) + cp2k_initial.inp (OFF), a LAMMPS
     data file for all atoms, and a carbon-only XYZ for CP2K.
  2. initial CP2K   (electrode only, no external potential) -> V_hartree cube,
     x2 -> V_ryd_initial.cube (Hartree -> Rydberg for the LAMMPS grid).
  3. initial LAMMPS (electrolyte point charges in the electrode potential)
     -> MDrho.cube (electrolyte charge density).
  4. chg2pot        MDrho.cube -> pot.cube (electrostatic potential of the
     electrolyte; read by CP2K as the external potential).
  5. main CP2K      (electrode in the electrolyte potential) -> V_hartree cube,
     x2 -> V_ryd.cube, and forces-output.xyz.
  6. final LAMMPS   (electrolyte in the updated potential) -> final.lammpstrj,
     PotEng / f_solvGrid in the log.
  7. extract        combine CP2K + LAMMPS into the coulomb energy/forces ->
     summary.data.

The results are gathered into a :class:`~dataprep.structures.Structures`; only
their energy/forces are used by the charge-decoupling subtraction.

Environment (module load for CP2K vs LAMMPS, MPI, conda) is the user's
responsibility — pass ``cmd_cp2k`` / ``cmd_lammps`` as the executables (or as
wrapper scripts that set up their own environment). CP2K and LAMMPS are run
through a :class:`~dataprep.launcher.ProcessLauncher`.
"""

from __future__ import annotations

from typing import Optional
import glob
import os
import shlex

import numpy as np

from cubetools import Cube, multiply
from poisson import chg2pot as _chg2pot

from .structures import Structure, Structures, Property
from .launcher import ProcessLauncher

KCALMOL2HA = 1.0 / 627.509474
ANG2BOHR = 1.0 / 0.529177210903

# element <-> LAMMPS atom type (edit for your system)
DEFAULT_TYPE_MAP = {"C": 1, "O": 2, "H": 3, "Na": 4, "Cl": 5}


# --------------------------------------------------------------------------
# Step 1 — build the DFT-CES inputs (ports createdatalammps.py)
# --------------------------------------------------------------------------
def create_dftces_inputs(job_dir: str, step_inp: str = "step-0.inp",
                         coord_xyz: str = "trajectory-input.xyz",
                         type_map: Optional[dict] = None) -> dict:
    """From the single-point CP2K input + coordinate XYZ in ``job_dir``, write
    ``cp2k.inp``, ``cp2k_initial.inp``, ``lammps-data.txt`` and
    ``carbon-input.xyz``. Returns the parsed box lengths (Angstrom)."""
    type_map = type_map or DEFAULT_TYPE_MAP
    inp_path = os.path.join(job_dir, step_inp)

    modified, lattice, in_cell = [], {}, False
    with open(inp_path) as f:
        for line in f:
            # Uncomment the external-potential directives only — a directive is a
            # line that, once the leading '#'/spaces are removed, *starts with*
            # the keyword (so prose comments mentioning it are left untouched).
            bare = line.lstrip("#").strip().upper()
            if (bare.startswith("&EXTERNAL_POTENTIAL")
                    or bare.startswith("&END EXTERNAL_POTENTIAL")
                    or bare.startswith("READ_FROM_CUBE")):
                line = line.lstrip("#")
            if "COORD_FILE_NAME" in line and not line.strip().startswith("#"):
                line = "      COORD_FILE_NAME carbon-input.xyz\n"
            if line.strip().startswith("PROJECT"):
                line = "  PROJECT step-0\n"
            modified.append(line)

            if "&CELL" in line:
                in_cell = True
            elif "&END CELL" in line:
                in_cell = False
            elif "PERIODIC XYZ" in line:
                pass
            elif in_cell:
                v = line.split()
                if not v:
                    pass
                elif v[0].upper() == "ABC":
                    lattice["A"], lattice["B"], lattice["C"] = (
                        float(v[1]), float(v[2]), float(v[3]))
                elif v[0].upper() == "A":
                    lattice["A"] = float(v[1])
                elif v[0].upper() == "B":
                    lattice["B"] = float(v[2])
                elif v[0].upper() == "C":
                    lattice["C"] = float(v[3])

    # cp2k.inp — external potential ON (embedding run)
    with open(os.path.join(job_dir, "cp2k.inp"), "w") as f:
        f.writelines(modified)

    # cp2k_initial.inp — external potential block removed, project 'initial'
    # Match only real directive lines (stripped of leading '#'/spaces, *starts
    # with* the keyword) so prose comments mentioning "&EXTERNAL_POTENTIAL" don't
    # trip the skip and wipe the top of the file.
    initial, skip = [], False
    for line in modified:
        bare = line.lstrip("#").strip().upper()
        if bare.startswith("&EXTERNAL_POTENTIAL"):
            skip = True
            continue
        if bare.startswith("&END EXTERNAL_POTENTIAL"):
            skip = False
            continue
        if skip:
            continue
        line = line.replace("PROJECT step-0", "PROJECT initial")
        line = line.replace("FILENAME =forces-output.xyz",
                            "FILENAME = initial_forces-output.xyz")
        line = line.replace("FILENAME forces-output.xyz",
                            "FILENAME initial_forces-output.xyz")
        initial.append(line)
    with open(os.path.join(job_dir, "cp2k_initial.inp"), "w") as f:
        f.writelines(initial)

    # atoms from the coordinate XYZ
    names, positions, carbons = [], [], []
    with open(os.path.join(job_dir, coord_xyz)) as f:
        f.readline(); f.readline()
        for line in f:
            p = line.split()
            if len(p) < 4:
                continue
            names.append(p[0])
            positions.append((float(p[1]), float(p[2]), float(p[3])))
            if p[0] == "C":
                carbons.append((float(p[1]), float(p[2]), float(p[3])))

    # lammps-data.txt (all atoms, point charges set to 0 here; real charges are
    # assigned in the LAMMPS input)
    ntypes = max(type_map.values())
    with open(os.path.join(job_dir, "lammps-data.txt"), "w") as f:
        f.write("LAMMPS Description\n\n")
        f.write(f"{len(names)} atoms\n{ntypes} atom types\n\n")
        f.write(f"0.0 {lattice['A']} xlo xhi\n")
        f.write(f"0.0 {lattice['B']} ylo yhi\n")
        f.write(f"0.0 {lattice['C']} zlo zhi\n\n")
        f.write("Masses\n\n")
        for el, t in sorted(type_map.items(), key=lambda kv: kv[1]):
            f.write(f"{t} {_MASS.get(el, 1.0):.4f}  # {el}\n")
        f.write("\nAtoms\n\n")
        for i, (nm, (x, y, z)) in enumerate(zip(names, positions), start=1):
            f.write(f"{i} 1 {type_map.get(nm, 1)} 0.00 {x} {y} {z}\n")

    # carbon-input.xyz (electrode only, for CP2K)
    with open(os.path.join(job_dir, "carbon-input.xyz"), "w") as f:
        f.write(f"{len(carbons)}\nelectrode carbons only\n")
        for (x, y, z) in carbons:
            f.write(f"C {x} {y} {z}\n")

    return lattice


_MASS = {"C": 12.0107, "O": 15.9994, "H": 1.00794, "Na": 22.9898, "Cl": 35.453}


# --------------------------------------------------------------------------
# Step 7 — combine CP2K + LAMMPS into the coulomb energy/forces
# --------------------------------------------------------------------------
def _read_forces_output(path):
    with open(path) as f:
        f.readline()
        energy = float(f.readline().split(",")[2].split("=")[1].strip())
        forces = []
        for line in f:
            p = line.split()
            if len(p) >= 4:
                forces.append((float(p[1]), float(p[2]), float(p[3])))
    return energy, forces


def _read_lammpstrj(path):
    bounds, atoms, reading = [], [], False
    with open(path) as f:
        for line in f:
            if "ITEM: BOX BOUNDS" in line:
                for _ in range(3):
                    bounds.append([float(x) for x in next(f).split()])
            elif "ITEM: ATOMS" in line:
                reading = True
            elif reading:
                p = line.split()
                if not p or "ITEM:" in line:
                    reading = False
                    continue
                atoms.append({"id": int(p[0]), "type": int(p[1]),
                              "coords": (float(p[2]), float(p[3]), float(p[4])),
                              "forces": (float(p[5]), float(p[6]), float(p[7]))})
    return bounds, atoms


def _read_lammps_final_out(path):
    poteng = f_solv = None
    with open(path) as f:
        for line in f:
            if "PotEng   =" in line:
                poteng = float(line.split()[2])
            elif "f_solvGrid =" in line:
                f_solv = float(line.split()[2])
    return poteng, f_solv


def read_coulomb_result(job_dir: str, type_map: Optional[dict] = None,
                        forces_file: str = "forces-output.xyz",
                        trj_file: str = "final.lammpstrj",
                        lammps_out: str = "lammps.final.out") -> Structure:
    """Combine the CP2K forces/energy + LAMMPS forces/energy of one job into the
    coulomb (electrostatic) :class:`Structure` (energy Hartree, forces a.u.).
    Ports the charge-decouple ``E_F_extract.py``."""
    type_map = type_map or DEFAULT_TYPE_MAP
    inv = {v: k for k, v in type_map.items()}

    e_cp2k, cp2k_forces = _read_forces_output(os.path.join(job_dir, forces_file))
    bounds, atoms = _read_lammpstrj(os.path.join(job_dir, trj_file))
    poteng, f_solv = _read_lammps_final_out(os.path.join(job_dir, lammps_out))
    atoms.sort(key=lambda a: a["id"])

    tot_e = e_cp2k + poteng * KCALMOL2HA - f_solv * KCALMOL2HA

    forces = []
    for i, a in enumerate(atoms):
        lf = tuple(f * KCALMOL2HA / ANG2BOHR for f in a["forces"])
        if i < len(cp2k_forces):
            forces.append(tuple(c + l for c, l in zip(cp2k_forces[i], lf)))
        else:
            forces.append(lf)

    names = [inv.get(a["type"], "X") for a in atoms]
    positions = np.array([a["coords"] for a in atoms])  # Angstrom (E/F only used)
    cell = np.diag([b[1] - b[0] for b in bounds[:3]])
    struc = Structure(names, positions, cell=cell)
    struc.properties["reference"] = Property(energy=tot_e, forces=np.array(forces))
    return struc


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def _run(launcher, cmd, args, workdir, logname):
    full = shlex.split(cmd) + list(args)
    r = launcher.run([full], workdir, check=False)[0]
    with open(os.path.join(workdir, logname), "w") as f:
        f.write(r.stdout)
    if r.returncode != 0:
        print(f"warning: '{full[0]}' exited {r.returncode} in {workdir} "
              f"(see {logname})")
    return r


def _cube_multi(infile, factor, outfile):
    multiply(Cube.from_file(infile), float(factor)).to_file(
        outfile, comment=(" Cubefile rescaled", " lattice unit Bohr"))


def run_coulomb(job_dirs="ref-calc-coulomb/calculator-*",
                cmd_cp2k: str = "cp2k.psmp", cmd_lammps: str = "lmp_mpi",
                launcher: Optional[ProcessLauncher] = None,
                lammps_initial: str = "lammps_initial.in",
                lammps_final: str = "lammps_final.in",
                initial_hartree_cube: str = "initial-V_hartree.cube-v_hartree-1_0.cube",
                main_hartree_cube: str = "step-0-V_hartree.cube-v_hartree-1_0.cube",
                mdrho_cube: str = "MDrho.cube",
                type_map: Optional[dict] = None,
                skip_existing: bool = True,
                out_data: Optional[str] = None) -> Structures:
    """Run the DFT-CES coulomb calculation in each job directory (7 steps each),
    writing ``summary.data`` per dir and returning the gathered coulomb
    :class:`Structures`.

    Each job dir must contain ``step-0.inp`` and ``trajectory-input.xyz`` (from
    :func:`~dataprep.cp2k.prepare_singlepoint_jobs`). ``lammps_initial`` /
    ``lammps_final`` are the LAMMPS DFT-CES input templates (resolved to absolute
    paths). ``skip_existing`` skips dirs that already have ``summary.data``.

    ``mdrho_cube`` is the electrolyte charge-density cube written by the initial
    LAMMPS ``grid`` command (default ``MDrho.cube``); it is fed to ``chg2pot``.
    ``initial_hartree_cube`` / ``main_hartree_cube`` are the CP2K V_hartree cube
    names — set these to match your CP2K PROJECT + ``&V_HARTREE_CUBE FILENAME``.
    """
    launcher = launcher or ProcessLauncher()
    lammps_initial = os.path.abspath(lammps_initial)
    lammps_final = os.path.abspath(lammps_final)
    if isinstance(job_dirs, str):
        job_dirs = sorted(glob.glob(job_dirs))

    results = Structures()
    for d in job_dirs:
        summ = os.path.join(d, "summary.data")
        if skip_existing and os.path.isfile(summ):
            print(f"skip {d} (has summary.data)")
            results.extend(Structures.from_file(summ))
            continue

        create_dftces_inputs(d, type_map=type_map)                       # 1
        _run(launcher, cmd_cp2k, ["-i", "cp2k_initial.inp"], d, "cp2k_initial.log")  # 2
        _cube_multi(os.path.join(d, initial_hartree_cube), 2,
                    os.path.join(d, "V_ryd_initial.cube"))
        _run(launcher, cmd_lammps, ["-in", lammps_initial], d, "lammps.initial.out")  # 3
        _chg2pot(Cube.from_file(os.path.join(d, mdrho_cube))).to_file(  # 4
            os.path.join(d, "pot.cube"),
            comment=(" Cubefile created from chg2pot",
                     " ES potential, lattice unit Bohr, grid unit Ha"))
        _run(launcher, cmd_cp2k, ["-i", "cp2k.inp"], d, "cp2k.log")       # 5
        _cube_multi(os.path.join(d, main_hartree_cube), 2,
                    os.path.join(d, "V_ryd.cube"))
        _run(launcher, cmd_lammps, ["-in", lammps_final], d, "lammps.final.out")      # 6

        struc = read_coulomb_result(d, type_map=type_map)                # 7
        Structures([struc]).to_file(summ, label_prop="reference")
        results.append(struc)

    if out_data is not None:
        results.to_file(out_data, label_prop="reference")
    return results
