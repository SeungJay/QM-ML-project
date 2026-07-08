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
    out_dir     : committee directory to create (``nnp-data-1 .. -N`` inside);
                  this is what stage 5's ``pair_style nnp dir "…"`` points at.
    elements    : element symbols in fixed order, e.g. ``('O','H','C','Na','Cl')``.
    metric      : best-epoch selection — ``"force"``, ``"energy"`` or ``"last"``.

    Members whose ``nnp-data-*`` dir already holds weights are skipped when
    ``skip_existing`` (resumable). Returns ``out_dir``.
    """
    data_file = os.path.abspath(data_file)
    template_nn = os.path.abspath(template_nn)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    for i in range(n_members):
        member_dir = os.path.join(out_dir, f"nnp-data-{i + 1}")
        done = glob.glob(os.path.join(member_dir, "weights.*.data"))
        if skip_existing and done:
            print(f"member {i + 1}/{n_members}: already trained, skipping")
            continue
        print(f"member {i + 1}/{n_members}: training (seed={seed0 + i}) ...")
        train_member(data_file, template_nn, member_dir,
                     elements=elements, seed=seed0 + i, n_epoch=n_epoch,
                     n_bins=n_bins, metric=metric, cmd_scaling=cmd_scaling,
                     cmd_train=cmd_train, launcher=launcher,
                     keep_train_dir=keep_train_dir)
    print(f"committee ready: {out_dir}  ({n_members} members)")
    return out_dir


def _split(cmd) -> list:
    """Accept a command as a string or an already-split list."""
    if isinstance(cmd, str):
        return cmd.split()
    return list(cmd)
