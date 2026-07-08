"""Stage 3 — charge decoupling (DFT-CES coulomb term, then subtract).

Only needs input-SR.data (from stage 2) plus the coulomb templates in this
folder. Reproduces the production run.sh + subtract_decouple.py flow:
  1. build coulomb job dirs (ref-calc-coulomb/calculator-NNN/) from input-SR.data,
  2. run the coupled CP2K<->LAMMPS DFT-CES coulomb calc in each (7 steps),
  3. subtract the coulomb energy/forces from the full DFT reference.

CP2K and LAMMPS must be built and callable. If they need different module
environments, point `cmd_cp2k` / `cmd_lammps` at wrapper scripts that set up
their own environment. Launch inside a SLURM allocation.
"""

import os

from dataprep import (prepare_jobs_from_data, run_coulomb, decouple_files,
                      ProcessLauncher)

# ---- ARCHER2 launcher ------------------------------------------------------
# The same launcher prefix is applied to both CP2K and LAMMPS. On ARCHER2 the
# recommended srun flags are added via this callable. Signature is fixed by
# ProcessLauncher: (i_slot, n_core_task, size_node) -> prefix string.
#
# Cores are read from the SLURM allocation (SLURM_NTASKS = nodes x
# tasks-per-node), so run.slurm is the single source of truth. The fallback is
# only used when running outside SLURM.
NCORES = int(os.environ.get("SLURM_NTASKS", 128))


def archer2_srun(i_slot, n_core_task, size_node):
    return (f"srun --hint=nomultithread --distribution=block:block "
            f"-n {n_core_task} ")


# CP2K on ARCHER2 is a central install (MPI-only cp2k.popt):
CP2K = "/work/y07/shared/apps/core/cp2k/cp2k-9.1.0/exe/ARCHER2/cp2k.popt"
# Bundled DFT-CES LAMMPS - use an ABSOLUTE path (run_coulomb executes it from
# inside each calculator-NNN/ dir, so a relative path would not resolve). REPLACE:
LMP = "/work/e05/e05/<user>/.../QM-ML-project/n2p2-v2.1.3-committee-nnp-extpot/bin/lmp_mpi"

launcher = ProcessLauncher(mode=archer2_srun, n_slots=1, n_core_task=NCORES)

# 1. build the coulomb job dirs straight from input-SR.data (geometry + cell come
#    from the dataset), using the coulomb CP2K template (external-potential block
#    present; forces + V_HARTREE_CUBE on)
prepare_jobs_from_data("input-SR.data", "coulomb-step-0.inp",
                       jobs_dir="ref-calc-coulomb", job_prefix="calculator")

# 2. DFT-CES coulomb calc in each dir -> summary.data, gathered into QMML.data
run_coulomb(
    "ref-calc-coulomb/calculator-*",
    cmd_cp2k=CP2K, cmd_lammps=LMP, launcher=launcher,
    lammps_initial="lammps_initial.in",   # DFT-CES LAMMPS templates
    lammps_final="lammps_final.in",
    out_data="ref-calc-coulomb/QMML.data",
    skip_existing=True,                   # resumable
)

# 3. subtract: input-SR - QMML(coulomb) -> input-SR-QMML.data (non-ES target)
decouple_files("input-SR.data", "ref-calc-coulomb/QMML.data",
               "input-SR-QMML.data")
print("wrote input-SR-QMML.data")
