"""Run a CP2K AIMD inside a SLURM allocation.

The launcher becomes e.g. `srun -n 64 cp2k.psmp -i aimd.inp`. Produces
trajectory-output.xyz and aimd.cell (per-frame cell, from &MOTION/&PRINT/&CELL
with FILENAME =aimd.cell in aimd.inp).
"""

from dataprep import run_aimd, ProcessLauncher

# ---- pick your MPI launcher: "srun" | "mpirun" | "plain" ----
LAUNCHER = "srun"
NCORES = 64                      # must match SBATCH -n

run_aimd(
    input_file="aimd.inp",       # your CP2K MD input
    cmd_cp2k="cp2k.psmp",        # CP2K executable
    output="aimd.log",
    launcher=ProcessLauncher(mode=LAUNCHER, n_core_task=NCORES),
)
