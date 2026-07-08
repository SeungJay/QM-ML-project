"""Run a CP2K AIMD inside a SLURM allocation (UK ARCHER2).

The launcher becomes e.g.
    srun --hint=nomultithread --distribution=block:block -n 512 cp2k.popt -i aimd.inp
Produces trajectory-output.xyz and aimd.cell (per-frame cell, from
&MOTION/&PRINT/&CELL with FILENAME =aimd.cell in aimd.inp).
"""

import os

from dataprep import run_aimd, ProcessLauncher

# ---- ARCHER2 launcher ------------------------------------------------------
# The built-in "srun"/"mpirun" presets only add `-n <NCORES>`. On ARCHER2 the
# recommended srun flags (--hint=nomultithread --distribution=block:block) are
# added through this callable launcher instead. Signature is fixed by
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

run_aimd(
    input_file="aimd.inp",       # your CP2K MD input
    cmd_cp2k=CP2K,               # CP2K executable
    output="aimd.log",
    launcher=ProcessLauncher(mode=archer2_srun, n_core_task=NCORES),
)
