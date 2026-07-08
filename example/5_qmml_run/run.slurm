#!/bin/bash
# Stage 5 - constant-potential QM/ML run.  Submit with:  sbatch run.slurm
# SLURM template for UK ARCHER2 (HPE Cray EX). run_qmml.py drives the LAMMPS +
# CP2K cycle loop; it launches srun internally (mpirun="srun" in run_qmml.py).
#SBATCH --job-name=qmml
#SBATCH --nodes=1
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
# Same toolchain the DFT-CES LAMMPS was built with, plus CP2K.
module load cpe/22.12
module load cray-fftw/3.3.10.3
module load gsl
module load eigen
# module load cp2k/cp2k-9.1.0    # if using the central CP2K in run_qmml.py

# QMML lives in this Anaconda (base) env - activate it so qmml_run imports. The
# job runs with --export=none, so activate inside the job (the login shell's
# PATH is not inherited). Anaconda provides python, so no cray-python needed.
source /work/e05/e05/jay/anaconda3/bin/activate

# Shared libs for the DFT-CES LAMMPS build. REPLACE with YOUR n2p2 build path
# (and custom GSL, if you built one):
export LD_LIBRARY_PATH="/work/e05/e05/<user>/.../QM-ML-project/n2p2-v2.1.3-committee-nnp-extpot/src/interface/lammps-nnp/src:${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
# export LD_LIBRARY_PATH="/work/e05/e05/<user>/0.programs/gsl/lib:${LD_LIBRARY_PATH}"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

cd "${SLURM_SUBMIT_DIR}"

# Make the QMML packages importable on compute nodes (see stage 1 note).
# EDIT to your repo location.
QMML_REPO=/work/e05/e05/jay/1.Project/8.NNP_EDL/1.Elyte_Bi_Graphenes/5.reproduce/QM-ML-project
export PYTHONPATH="${QMML_REPO}/src:${PYTHONPATH}"

# Launch the driver (writes progress to python.txt)
python run_qmml.py > python.txt
