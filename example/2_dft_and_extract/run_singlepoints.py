"""Stage 2 (part 1) — build per-frame job dirs and run the single-point DFT.

Step 1 (cheap): prepare_singlepoint_jobs builds calculator-NNN/ from the frames
in trajectory_inputs/ (carried from stage 1) using step-0.inp as the template,
substituting each frame's cell into &CELL.
Step 2 (heavy): run_singlepoints runs CP2K in each dir; it skips dirs that
already have forces-output.xyz, so it is safe to resubmit.

Launch inside a SLURM allocation:  python run_singlepoints.py
"""

from dataprep import prepare_singlepoint_jobs, run_singlepoints, ProcessLauncher

# ---- pick your MPI launcher: "srun" | "mpirun" | "plain" ----
LAUNCHER = "srun"
NCORES = 64                      # must match SBATCH -n

# 1. build calculator-NNN/ (coord + step-0.inp with per-frame cell)
prepare_singlepoint_jobs("trajectory_inputs", "step-0.inp",
                         jobs_dir=".", job_prefix="calculator")

# 2. run CP2K single point in each (resumable)
run_singlepoints(
    job_dirs="calculator-*",
    input_file="step-0.inp",
    cmd_cp2k="cp2k.psmp",
    launcher=ProcessLauncher(mode=LAUNCHER, n_core_task=NCORES),
    skip_existing=True,
)
