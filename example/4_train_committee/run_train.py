#!/usr/bin/env python
"""Stage 4 — train the n2p2 committee potential (no active learning).

A committee = `n_members` ordinary n2p2 NNPs trained on the same data set, each
with a different random seed. `train_committee` drives the bundled n2p2 tools
(`nnp-scaling` then `nnp-train`) once per member, picks each member's best epoch
from its `learning-curve.out`, and assembles the committee directory LAMMPS
expects:

    committee/
        nnp-data-1/  input.nn  scaling.data  weights.<Z>.data ...
        nnp-data-2/  ...
        ...

Stage 5 then points `pair_style nnp dir "…/committee/"` at it.

Only the bundled n2p2 binaries are used (no external Python package). Launch
inside a SLURM allocation — see run.slurm.
"""

from dataprep import train_committee, ProcessLauncher

# ---- pick your MPI launcher: "srun" | "mpirun" | "plain" ----
LAUNCHER = "srun"
NCORES = 128                      # must match SBATCH tasks

train_committee(
    data_file="input-SR-QMML.data",   # decoupled training set from stage 3
    template_nn="input.nn",           # n2p2 template ({n_elements}/{elements}/
                                      #                 {seed}/{n_epoch})
    out_dir="committee",              # committee dir (nnp-data-1..N inside)
    elements=("O", "H", "C", "Na", "Cl"),   # REPLACE for your system (fixed order)
    n_members=8,                      # committee size
    n_epoch=100,                      # training epochs per member
    n_bins=500,                       # nnp-scaling histogram bins
    seed0=1,                          # member i uses random_seed = seed0 + i
    metric="force",                   # best-epoch by test RMSE: "force"|"energy"|"last"
    cmd_scaling="nnp-scaling",        # bundled n2p2 tools (on PATH)
    cmd_train="nnp-train",
    launcher=ProcessLauncher(mode=LAUNCHER, n_core_task=NCORES),
    skip_existing=True,               # resumable: skip members already trained
)

# Output: committee/  ->  set in stage 5's base.in.lammps:
#     pair_style nnp dir "…/committee/"  showewsum 0  showew no  resetew yes  ...
