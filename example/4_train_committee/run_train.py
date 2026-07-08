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
# Committee members train IN PARALLEL — one node each (like the old AML setup).
# Allocate one node per member in run.slurm (nodes = N_MEMBERS). Each member's
# srun then uses one node's worth of cores, and SLURM spreads the concurrent
# steps across the nodes. Cores-per-member = total tasks / n_members.
N_MEMBERS = 2                                             # committee size (example)
TOTAL = int(os.environ.get("SLURM_NTASKS", N_MEMBERS * 128))
CORES_PER_MEMBER = max(1, TOTAL // N_MEMBERS)             # ~one node per member


def archer2_srun(i_slot, n_core_task, size_node):
    return (f"srun --hint=nomultithread --distribution=block:block "
            f"-n {n_core_task} ")

train_committee(
    data_file="input-SR-QMML.data",   # decoupled training set from stage 3
    template_nn="input.nn",           # n2p2 template ({n_elements}/{elements}/
                                      #                 {seed}/{n_epoch})
    out_dir="committee",              # committee dir (member 0 top level + nnp-data-1..)
    elements=("O", "H", "C", "Na", "Cl"),   # REPLACE for your system (fixed order)
    n_members=N_MEMBERS,              # committee size
    n_parallel=N_MEMBERS,             # train all members at once (one node each)
    n_epoch=100,                      # training epochs per member
    n_bins=100,                       # nnp-scaling bins (AML hardcoded 100); optional
    seed0=1,                          # member i uses random_seed = seed0 + i
    metric="last",                    # AML used the final epoch; "force"/"energy" = best-RMSE
    cmd_scaling="nnp-scaling",        # bundled n2p2 tools (on PATH)
    cmd_train="nnp-train",
    launcher=ProcessLauncher(mode=archer2_srun, n_core_task=CORES_PER_MEMBER),
    skip_existing=True,               # resumable: skip members already trained
    keep_train_dir=False,             # True keeps each member's <dir>.train scratch
                                      # (per-epoch weights, nnp logs). learning-curve.out
                                      # is kept in the member dir either way.
)

# Output: committee/ — member 0 at the top level (input.nn with committee_mode /
# committee_data, scaling.data, weights.<Z>.data) + members 1..N-1 in
# committee/nnp-data-1..N-1/. Point stage 5's base.in.lammps at it:
#     pair_style nnp dir "…/committee/"  showewsum 0  showew no  resetew yes  ...
