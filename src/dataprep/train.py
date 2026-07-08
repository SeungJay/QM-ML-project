"""Train an n2p2 committee potential directly (no active learning).

A committee neural network potential (C-NNP) is simply ``n_members`` ordinary
n2p2 NNPs trained on the same data set, each with a different random seed. This
module drives the standard n2p2 tools — ``nnp-scaling`` then ``nnp-train`` — once
per member, then collects each member's best-epoch weights into the directory
layout LAMMPS expects::

    <out_dir>/
        nnp-data-1/  input.nn  scaling.data  weights.<Z>.data ...
        nnp-data-2/  ...
        ...

That directory is what stage 5 points `pair_style nnp dir "<out_dir>"` at.

Everything here shells out to the n2p2 binaries bundled with QMML
(``n2p2-v2.1.3-committee-nnp-extpot``) through :class:`ProcessLauncher`; there is
no dependency on any external Python package. All numbers stay in the atomic
units of the ``.data`` file — n2p2 does its own internal unit handling.
"""

from __future__ import annotations

import glob
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Sequence

from .launcher import ProcessLauncher
from .symfunc import make_symfunctions

# Atomic numbers for the weights file names (weights.%03d.data, %03d = Z).
_ATOMIC_NUMBER = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Ru": 44,
    "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51,
    "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "Pt": 78, "Au": 79,
    "Hg": 80, "Pb": 82,
}


def _fill_template(template: str, *, elements: Sequence[str], seed: int,
                   n_epoch: int) -> str:
    """Substitute the placeholders in an n2p2 ``input.nn`` template.

    Fills ``{n_elements}`` / ``{elements}`` / ``{seed}`` / ``{n_epoch}`` (the AML
    placeholders, format specs like ``{n_elements:d}`` / ``{elements:s}`` also
    work) and the symmetry-function block ``{acsf}`` (AML's name; ``{symmetry_
    functions}`` is accepted as an alias) — the full element-resolved ACSF set
    generated for ``elements`` by :func:`dataprep.symfunc.make_symfunctions`. A
    template that lists its symmetry functions explicitly is still supported
    (unused keywords are ignored)."""
    acsf = make_symfunctions(elements)
    return template.format(n_elements=len(elements),
                           elements=" ".join(elements),
                           seed=seed, n_epoch=n_epoch,
                           acsf=acsf, symmetry_functions=acsf)


def select_best_epoch(learning_curve: str, metric: str = "force") -> int:
    """Return the epoch with the lowest **test-set** RMSE in an n2p2
    ``learning-curve.out``.

    metric : ``"force"`` -> RMSE_Ftest_pu, ``"energy"`` -> RMSEpa_Etest_pu,
             ``"last"``   -> the final epoch (no selection).

    The column is located by name from the ``#`` header, so it is robust to
    n2p2 writing different sets of metrics.
    """
    lines = Path(learning_curve).read_text().splitlines()

    # Map column name -> 1-based index from header lines like "#    5 RMSE_Ftest_pu".
    col = {}
    for ln in lines:
        m = re.match(r"#\s*(\d+)\s+(\S+)", ln)
        if m:
            col[m.group(2)] = int(m.group(1)) - 1  # to 0-based

    data = [ln.split() for ln in lines if ln.strip() and not ln.startswith("#")]
    if not data:
        raise ValueError(f"no data rows in {learning_curve}")
    epoch_col = col.get("epoch", 0)

    if metric == "last":
        return int(float(data[-1][epoch_col]))

    name = {"force": "RMSE_Ftest_pu",
            "energy": "RMSEpa_Etest_pu"}[metric]
    if name not in col:
        # fall back to the other metric, then to the last epoch
        alt = "RMSEpa_Etest_pu" if metric == "force" else "RMSE_Ftest_pu"
        name = alt if alt in col else None
    if name is None:
        return int(float(data[-1][epoch_col]))

    c = col[name]
    best = min(data, key=lambda r: float(r[c]))
    return int(float(best[epoch_col]))


def train_member(data_file: str, template_nn: str, member_dir: str, *,
                 elements: Sequence[str], seed: int, n_epoch: int = 100,
                 n_bins: int = 500, metric: str = "force",
                 cmd_scaling: str = "nnp-scaling", cmd_train: str = "nnp-train",
                 launcher: Optional[ProcessLauncher] = None,
                 keep_train_dir: bool = False) -> str:
    """Train one committee member and assemble its ``nnp-data-*`` directory.

    Runs ``nnp-scaling`` then ``nnp-train`` in a scratch dir, picks the best
    epoch, and writes ``input.nn`` + ``scaling.data`` + ``weights.<Z>.data`` into
    ``member_dir``. Returns ``member_dir``.
    """
    launcher = launcher or ProcessLauncher(mode="plain")
    member_dir = os.path.abspath(member_dir)
    os.makedirs(member_dir, exist_ok=True)
    train_dir = member_dir + ".train"
    os.makedirs(train_dir, exist_ok=True)

    template = Path(template_nn).read_text()
    input_nn = _fill_template(template, elements=elements, seed=seed,
                              n_epoch=n_epoch)
    Path(train_dir, "input.nn").write_text(input_nn)
    shutil.copyfile(data_file, os.path.join(train_dir, "input.data"))

    # 1) symmetry-function scaling  -> scaling.data
    launcher.run([[*_split(cmd_scaling), str(n_bins)]], train_dir, check=True)
    # 2) train the network         -> weights.<Z>.<epoch>.out, learning-curve.out
    launcher.run([_split(cmd_train)], train_dir, check=True)

    best = select_best_epoch(os.path.join(train_dir, "learning-curve.out"),
                             metric=metric)

    # 3) collect final files into member_dir
    shutil.copyfile(os.path.join(train_dir, "input.nn"),
                    os.path.join(member_dir, "input.nn"))
    shutil.copyfile(os.path.join(train_dir, "scaling.data"),
                    os.path.join(member_dir, "scaling.data"))
    for el in elements:
        z = _ATOMIC_NUMBER[el]
        src = os.path.join(train_dir, f"weights.{z:03d}.{best:06d}.out")
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"expected best-epoch weights {src} not found — check that "
                f"`write_weights_epoch 1` is set in input.nn")
        shutil.copyfile(src, os.path.join(member_dir, f"weights.{z:03d}.data"))

    if not keep_train_dir:
        shutil.rmtree(train_dir, ignore_errors=True)
    return member_dir


def train_committee(data_file: str, template_nn: str, out_dir: str = "committee",
                    *, elements: Sequence[str], n_members: int = 8,
                    n_parallel: int = 1,
                    n_epoch: int = 100, n_bins: int = 500, seed0: int = 1,
                    metric: str = "force",
                    cmd_scaling: str = "nnp-scaling",
                    cmd_train: str = "nnp-train",
                    launcher: Optional[ProcessLauncher] = None,
                    skip_existing: bool = True,
                    keep_train_dir: bool = False) -> str:
    """Train an ``n_members``-strong n2p2 committee on ``data_file``.

    data_file   : training set, e.g. ``input-SR-QMML.data`` from stage 3.
    template_nn : an n2p2 ``input.nn`` with {n_elements}/{elements}/{seed}/
                  {n_epoch} placeholders (member seed is ``seed0 + i``). A
                  {symmetry_functions} placeholder, if present, is filled with the
                  element-resolved ACSF block generated for ``elements`` (see
                  :mod:`dataprep.symfunc`); a template listing symmetry functions
                  explicitly also works.
    out_dir     : committee directory to create — this is what stage 5's
                  ``pair_style nnp dir "…"`` points at. Members are laid out the
                  way the committee-n2p2 build reads them (see below).
    elements    : element symbols in fixed order, e.g. ``('O','H','C','Na','Cl')``.
    metric      : best-epoch selection — ``"force"``, ``"energy"`` or ``"last"``.

    Layout produced (committee member ``c`` = ``0 .. n_members-1``, seed
    ``seed0 + c``), matching ``Mode::readNeuralNetworkWeights`` in the bundled
    n2p2::

        out_dir/                     committee member 0 (top level, no subdir)
            input.nn                 committee descriptor (+ committee_mode /
                                     committee_data — read only by LAMMPS /
                                     nnp-comp2, NOT by nnp-train)
            scaling.data  weights.<Z>.data
            nnp-data-1/  weights.<Z>.data   committee member 1
            ...
            nnp-data-<N-1>/ ...             committee member N-1

    This is *not* active learning: each member is trained for ``n_epoch`` epochs,
    its best epoch is kept, and the committee is assembled — done.

    ``n_parallel`` : how many members to train **at the same time**. With
    ``n_parallel = n_members`` and a SLURM allocation of one node per member,
    each member's ``srun`` step lands on its own node (SLURM spreads concurrent
    steps across free nodes), so the whole committee trains in parallel — the way
    the old AML committee did. Give the launcher ``n_core_task`` = cores per
    member (one node's worth), not the whole allocation. ``n_parallel = 1`` (the
    default) trains members one after another on the full allocation.

    Members whose target dir already holds weights are skipped when
    ``skip_existing`` (resumable). Returns ``out_dir``.
    """
    data_file = os.path.abspath(data_file)
    template_nn = os.path.abspath(template_nn)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # collect the members that still need training (committee member 0 lives at
    # the top level; members 1..N-1 in nnp-data-c/)
    todo = []
    for c in range(n_members):
        member_dir = out_dir if c == 0 else os.path.join(out_dir, f"nnp-data-{c}")
        if skip_existing and glob.glob(os.path.join(member_dir, "weights.*.data")):
            print(f"member {c + 1}/{n_members}: already trained, skipping")
            continue
        todo.append((c, member_dir))

    def _train_one(c, member_dir):
        print(f"member {c + 1}/{n_members}: training (seed={seed0 + c}) ...")
        train_member(data_file, template_nn, member_dir,
                     elements=elements, seed=seed0 + c, n_epoch=n_epoch,
                     n_bins=n_bins, metric=metric, cmd_scaling=cmd_scaling,
                     cmd_train=cmd_train, launcher=launcher,
                     keep_train_dir=keep_train_dir)

    workers = max(1, min(int(n_parallel), len(todo)))
    if workers > 1:
        # Train members concurrently. Each train_member blocks on its own srun
        # step; run in threads so the steps overlap and SLURM places them on
        # separate nodes of the allocation.
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_train_one, c, d) for c, d in todo]
            for f in futures:
                f.result()   # propagate the first exception, if any
    else:
        for c, d in todo:
            _train_one(c, d)

    # top-level committee descriptor: member 0's input.nn + the prediction-only
    # committee keywords (written last so it is correct after a resumed run too).
    _write_committee_descriptor(out_dir, template_nn, elements=elements,
                                seed=seed0, n_epoch=n_epoch, n_members=n_members)
    print(f"committee ready: {out_dir}  ({n_members} members, "
          f"{workers}-way parallel)")
    return out_dir


def _write_committee_descriptor(out_dir: str, template_nn: str, *,
                                elements: Sequence[str], seed: int,
                                n_epoch: int, n_members: int) -> str:
    """Write the top-level committee ``input.nn`` LAMMPS / nnp-comp2 read.

    It is member 0's filled ``input.nn`` plus ``committee_mode prediction`` and
    ``committee_data <prefix> <size>``. These two keywords are **prediction only**
    — ``nnp-train`` rejects a committee size > 1 — so they live only here, never in
    the per-member training inputs.
    """
    text = _fill_template(Path(template_nn).read_text(),
                          elements=elements, seed=seed, n_epoch=n_epoch)
    text = text.rstrip("\n") + (
        "\n\n"
        "###############################################################################\n"
        "# COMMITTEE (prediction only — read by LAMMPS pair_style nnp / nnp-comp2,\n"
        "# NOT by nnp-train)\n"
        "###############################################################################\n"
        "committee_mode                  prediction     # average all NNs for prediction\n"
        f"committee_data                  nnp-data {n_members}     # dir prefix + committee size\n")
    out = os.path.join(out_dir, "input.nn")
    Path(out).write_text(text)
    return out


def _split(cmd) -> list:
    """Accept a command as a string or an already-split list."""
    if isinstance(cmd, str):
        return cmd.split()
    return list(cmd)
