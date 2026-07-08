#!/bin/bash
# Stage 2 - single-point DFT + extract.  Submit with:  sbatch run.slurm
# SLURM template for UK ARCHER2 (HPE Cray EX). Runs the whole stage in one job:
# the single-point DFT (heavy; resumable - skips calculator-NNN/ dirs that
# already have forces-output.xyz), then the extraction (cheap, serial).
# run_singlepoints.py reads the core count from SLURM_NTASKS (= nodes x
# tasks-per-node), so just set the allocation here.
#SBATCH --job-name=dft_sp
#SBATCH --nodes=4
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --account=e05-react-wal
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --error=stderr_%j.txt
#SBATCH --output=stdout_%j.txt
#SBATCH --export=none

# ---- environment (ARCHER2) --------------------------------------------------
module load cp2k/cp2k-9.1.0

# QMML lives in this Anaconda (base) env - activate it so dataprep imports. The
# job runs with --export=none, so activate inside the job (the login shell's
# PATH is not inherited). Anaconda provides python, so no cray-python needed.
source /work/e05/e05/jay/anaconda3/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

cd "${SLURM_SUBMIT_DIR}"

# CP2K basis/potential data must be visible from the run dir; export if needed:
# export CP2K_DATA_DIR=/work/y07/shared/apps/core/cp2k/cp2k-9.1.0/data

# 1) single-point DFT (heavy - run_singlepoints.py launches srun cp2k internally)
python run_singlepoints.py

# 2) extract into input-SR.data (cheap, serial - pure Python, no MPI)
python run_extract.py
