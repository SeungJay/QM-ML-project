#!/bin/bash
# Stage 1 - CP2K AIMD + frame selection.  Submit with:  sbatch run.slurm
# SLURM template for UK ARCHER2 (HPE Cray EX). Runs the whole stage in one job:
# the AIMD (heavy), then the frame selection (cheap, serial).
# run_aimd.py reads the core count from SLURM_NTASKS (= nodes x tasks-per-node),
# so just set the allocation here.
#SBATCH --job-name=aimd
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

# Make the QMML packages importable on compute nodes. A `pip install --user`
# lands in ~/.local (=/home), which ARCHER2 compute nodes cannot see, so point
# PYTHONPATH at the repo's src/ under /work. EDIT to your repo location.
QMML_REPO=/work/e05/e05/jay/1.Project/8.NNP_EDL/1.Elyte_Bi_Graphenes/5.reproduce/QM-ML-project
export PYTHONPATH="${QMML_REPO}/src:${PYTHONPATH}"

# CP2K basis/potential data (GTH_BASIS_SETS, BASIS_MOLOPT, POTENTIAL, dftd3.dat)
# must be visible from the run dir; export CP2K_DATA_DIR if aimd.inp needs it:
# export CP2K_DATA_DIR=/work/y07/shared/apps/core/cp2k/cp2k-9.1.0/data

# 1) AIMD (heavy - run_aimd.py launches srun cp2k internally)
python run_aimd.py

# 2) frame selection (cheap, serial - pure Python, no MPI)
python run_select_frames.py
