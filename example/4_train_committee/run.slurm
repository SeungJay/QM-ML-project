#!/bin/bash
# Stage 4 — n2p2 committee training.  Submit with:  sbatch run.slurm
# NCORES in run_train.py must match the task count here.
#SBATCH --job-name=nnp_committee
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --error=stderr_%j.txt
#SBATCH --output=stdout_%j.txt
#SBATCH --export=none

# ---- environment (edit modules / paths for your cluster) --------------------
# module load ...            # the compiler + MKL/GSL/Eigen n2p2 was built with
# source /path/to/conda/bin/activate    # a Python env with QMML installed

export OMP_NUM_THREADS=1
cd "${SLURM_SUBMIT_DIR}"

# n2p2 training binaries (nnp-scaling, nnp-train) on PATH — from the bundled
# n2p2-v2.1.3-committee-nnp-extpot build:
export PATH="/path/to/n2p2-v2.1.3-committee-nnp-extpot/bin:${PATH}"

python run_train.py
