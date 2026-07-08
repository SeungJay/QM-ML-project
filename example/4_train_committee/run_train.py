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

import os

from dataprep import train_committee, ProcessLauncher

# ---- ARCHER2 launcher ------------------------------------------------------
# Adds the recommended ARCHER2 srun flags via a callable launcher. Signature is
# fixed by ProcessLauncher: (i_slot, n_core_task, size_node) -> prefix string.
#
# Cores are read from the SLURM allocation (SLURM_NTASKS = nodes x
# tasks-per-node), so run.slurm is the single source of truth. The fallback is
# only used when running outside SLURM.
NCORES = int(os.environ.get("SLURM_NTASKS", 128))


def archer2_srun(i_slot, n_core_task, size_node):
    return (f"srun --hint=nomultithread --distribution=block:block "
            f"-n {n_core_task} ")

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
    launcher=ProcessLauncher(mode=archer2_srun, n_core_task=NCORES),
    skip_existing=True,               # resumable: skip members already trained
)

# Output: committee/ — member 0 at the top level (input.nn with committee_mode /
# committee_data, scaling.data, weights.<Z>.data) + members 1..N-1 in
# committee/nnp-data-1..N-1/. Point stage 5's base.in.lammps at it:
#     pair_style nnp dir "…/committee/"  showewsum 0  showew no  resetew yes  ...
