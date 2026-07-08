#!/bin/bash
# Stage 3 - DFT-CES charge decoupling.  Submit with:  sbatch run.slurm
# SLURM template for UK ARCHER2 (HPE Cray EX). run_decouple.py runs the coupled
# CP2K <-> LAMMPS coulomb calc in each job dir, then subtracts it from the DFT
# reference. Resumable - skips dirs that already have summary.data.
# run_decouple.py reads the core count from SLURM_NTASKS (= nodes x
# tasks-per-node), so just set the allocation here.
#SBATCH --job-name=decouple
#SBATCH --nodes=4
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --account=e05-react-wal
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --error=stderr_%j.txt
#SBATCH --output=stdout_%j.txt
#SBATCH --export=none

# ---- environment (ARCHER2) --------------------------------------------------
# This stage needs BOTH CP2K and the bundled LAMMPS (lmp_mpi) callable in the
# same job. If the two require conflicting module environments, point cmd_cp2k /
# cmd_lammps in run_decouple.py at wrapper scripts that each `module load` their
# own env; otherwise loading both here is usually fine.
module load cp2k/cp2k-9.1.0
module load gsl                 # runtime lib for the n2p2/LAMMPS build

# QMML lives in this Anaconda (base) env - activate it so dataprep imports. The
# job runs with --export=none, so activate inside the job (the login shell's
# PATH is not inherited). Anaconda provides python, so no cray-python needed.
source /work/e05/e05/jay/anaconda3/bin/activate

# The DFT-CES LAMMPS needs its build's shared libs on the loader path. Point
# these at YOUR n2p2 build (and custom GSL, if you built one):
# export LD_LIBRARY_PATH="/work/e05/e05/<user>/.../QM-ML-project/n2p2-v2.1.3-committee-nnp-extpot/src/interface/lammps-nnp/src:${LD_LIBRARY_PATH}"
# export LD_LIBRARY_PATH="/work/e05/e05/<user>/0.programs/gsl/lib:${LD_LIBRARY_PATH}"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

cd "${SLURM_SUBMIT_DIR}"

# export CP2K_DATA_DIR=/work/y07/shared/apps/core/cp2k/cp2k-9.1.0/data   # if needed

python run_decouple.py
