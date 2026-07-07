#!/bin/bash
# Stage 3 — DFT-CES charge decoupling.  Submit with:  sbatch run.slurm
# Runs run_decouple.py: the coupled CP2K <-> LAMMPS coulomb calc in each job dir,
# then subtracts it from the DFT reference. Resumable — skips dirs that already
# have summary.data.
# --ntasks MUST match NCORES in run_decouple.py (note it uses mpirun by default).
#SBATCH --job-name=decouple
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --error=stderr_%j.txt
#SBATCH --output=stdout_%j.txt
#SBATCH --export=none

# ---- environment (edit modules / paths for your cluster) --------------------
# module load cp2k/cp2k-9.1.0    # CP2K (electrode DFT)
# source /path/to/conda/bin/activate    # a Python env with QMML installed

export OMP_NUM_THREADS=1
cd "${SLURM_SUBMIT_DIR}"

# Both CP2K and the bundled LAMMPS must be callable — CP2K binary and the LAMMPS
# executable (…/n2p2-v2.1.3-committee-nnp-extpot/bin/lmp_mpi) are passed via
# cmd_cp2k / cmd_lammps in run_decouple.py. Put them on PATH here if needed:
# export PATH="/path/to/cp2k/bin:${PATH}"
# export LD_LIBRARY_PATH="/path/to/n2p2/.../lammps-nnp/src:${LD_LIBRARY_PATH}"

python run_decouple.py
