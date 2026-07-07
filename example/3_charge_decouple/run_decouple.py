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

from dataprep import (prepare_jobs_from_data, run_coulomb, decouple_files,
                      ProcessLauncher)

LAUNCHER = "mpirun"     # "srun" | "mpirun" | "plain"
NCORES = 64             # must match SBATCH -n
CP2K = "cp2k.psmp"
LMP = "../n2p2-v2.1.3-committee-nnp-extpot/bin/lmp_mpi"   # built LAMMPS (DFT-CES)

launcher = ProcessLauncher(mode=LAUNCHER, n_slots=1, n_core_task=NCORES)

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
