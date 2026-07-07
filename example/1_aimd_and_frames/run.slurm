#!/bin/bash
# Stage 1 — CP2K AIMD + frame selection.  Submit with:  sbatch run.slurm
# Runs the whole stage in one job: the AIMD (heavy), then the frame selection
# (cheap, serial). --ntasks MUST match NCORES in run_aimd.py.
#SBATCH --job-name=aimd
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --error=stderr_%j.txt
#SBATCH --output=stdout_%j.txt
#SBATCH --export=none

# ---- environment (edit modules / paths for your cluster) --------------------
# module load cp2k/cp2k-9.1.0    # or whatever provides your cp2k binary
# source /path/to/conda/bin/activate    # a Python env with QMML installed

export OMP_NUM_THREADS=1
cd "${SLURM_SUBMIT_DIR}"

# CP2K binary on PATH (or give a full path via cmd_cp2k in run_aimd.py):
# export PATH="/path/to/cp2k/bin:${PATH}"

# 1) AIMD (heavy — launches srun cp2k internally)
python run_aimd.py

# 2) frame selection (cheap, serial — pure Python, no MPI)
python run_select_frames.py
