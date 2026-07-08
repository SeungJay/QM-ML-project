"""Stage 2 (part 1) — build per-frame job dirs and run the single-point DFT.

Step 1 (cheap): prepare_singlepoint_jobs builds calculator-NNN/ from the frames
in trajectory_inputs/ (carried from stage 1) using step-0.inp as the template,
substituting each frame's cell into &CELL.
Step 2 (heavy): run_singlepoints runs CP2K in each dir; it skips dirs that
already have forces-output.xyz, so it is safe to resubmit.

Launch inside a SLURM allocation:  python run_singlepoints.py
"""

import os

from dataprep import prepare_singlepoint_jobs, run_singlepoints, ProcessLauncher

# ---- ARCHER2 launcher ------------------------------------------------------
# Adds the recommended ARCHER2 srun flags via a callable launcher. Signature is
# fixed by ProcessLauncher: (i_slot, n_core_task, size_node) -> prefix string.
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

# 1. build calculator-NNN/ (coord + step-0.inp with per-frame cell)
prepare_singlepoint_jobs("trajectory_inputs", "step-0.inp",
                         jobs_dir=".", job_prefix="calculator")

# 2. run CP2K single point in each (resumable)
run_singlepoints(
    job_dirs="calculator-*",
    input_file="step-0.inp",
    cmd_cp2k=CP2K,
    launcher=ProcessLauncher(mode=archer2_srun, n_core_task=NCORES),
    skip_existing=True,
)
