#!/bin/bash
# Stage 4 - n2p2 committee training.  Submit with:  sbatch run.slurm
# SLURM template for UK ARCHER2 (HPE Cray EX). Trains the committee with the
# bundled n2p2 tools (nnp-scaling, nnp-train). Resumable - members already
# trained are skipped, so if the job hits the wall time just resubmit.
# run_train.py reads the core count from SLURM_NTASKS (= nodes x tasks-per-node),
# so just set the allocation here.
#SBATCH --job-name=nnp_committee
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
# Same toolchain the n2p2 tools were built with.
module load cpe/22.12
module load gsl
module load eigen

# QMML lives in this Anaconda (base) env - activate it so dataprep imports. The
# job runs with --export=none, so activate inside the job (the login shell's
# PATH is not inherited). Anaconda provides python, so no cray-python needed.
source /work/e05/e05/jay/anaconda3/bin/activate

# n2p2 training binaries (nnp-scaling, nnp-train) on PATH - from the bundled
# n2p2-v2.1.3-committee-nnp-extpot build. REPLACE with YOUR build path:
export PATH="/work/e05/e05/<user>/.../QM-ML-project/n2p2-v2.1.3-committee-nnp-extpot/bin:${PATH}"
# If you built against a custom GSL, add its lib too:
# export LD_LIBRARY_PATH="/work/e05/e05/<user>/0.programs/gsl/lib:${LD_LIBRARY_PATH}"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

cd "${SLURM_SUBMIT_DIR}"

# Make the QMML packages importable on compute nodes (see stage 1 note).
# EDIT to your repo location.
QMML_REPO=/work/e05/e05/jay/1.Project/8.NNP_EDL/1.Elyte_Bi_Graphenes/5.reproduce/QM-ML-project
export PYTHONPATH="${QMML_REPO}/src:${PYTHONPATH}"

python run_train.py
