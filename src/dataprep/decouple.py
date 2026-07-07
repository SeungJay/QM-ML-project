"""Charge decoupling: subtract the electrostatic (coulomb) term from DFT data.

Given the full DFT reference data and the coulomb (electrostatic) term over the
same structures, subtract the coulomb energy and forces element-wise to obtain
the short-range (decoupled) reference. Ports ``subtract_decouple.py``.

Only as many frames as the coulomb set provides are processed and written, so a
partially-completed coulomb run still yields a clean (uncontaminated) dataset.
All arithmetic is on the raw stored (atomic-unit) numbers — no conversion.
"""

from __future__ import annotations

from typing import Optional

from .structures import Structures


def subtract(reference: Structures, coulomb: Structures,
             label_ref: str = "reference",
             label_coulomb: str = "reference") -> Structures:
    """Subtract coulomb energy/forces from ``reference``, in place.

    Processes ``min(len(reference), len(coulomb))`` aligned frames and returns a
    :class:`Structures` of just those processed frames (so trailing,
    not-yet-computed reference frames are excluded)."""
    n = min(len(reference), len(coulomb))
    if len(reference) != len(coulomb):
        print(f"note: reference has {len(reference)} frames, coulomb has "
              f"{len(coulomb)}; processing the first {n}")
    for i in range(n):
        struc, cou = reference[i], coulomb[i]
        if struc.n_atoms != cou.n_atoms:
            raise ValueError(f"atom-count mismatch at frame {i}")
        rp = struc.properties[label_ref]
        cp = cou.properties[label_coulomb]
        rp._energy -= cp.energy
        rp._forces -= cp.forces
    return Structures(reference[:n])


def decouple_inplace(structures: Structures, label_ref: str = "reference",
                     label_coulomb: str = "coulomb") -> Structures:
    """Subtract each structure's ``coulomb`` property from its ``reference``
    property, in place. Use when both properties live on the same structures."""
    return subtract(structures, structures,
                    label_ref=label_ref, label_coulomb=label_coulomb)


def decouple_files(reference_data: str, coulomb_data: str, out_data: str,
                   label_ref: str = "reference",
                   label_coulomb: str = "reference") -> Structures:
    """Read the full DFT reference and the coulomb ``.data``, subtract, and write
    the decoupled dataset (only the frames present in the coulomb set).

    reference_data : full DFT reference (e.g. ``input-SR.data``)
    coulomb_data   : coulomb term, e.g. ``ref-calc-coulomb/QMML.data`` from
                     :func:`~dataprep.dftces.run_coulomb`
    out_data       : output (e.g. ``input-SR-QMML.data``)
    """
    ref = Structures.from_file(reference_data, label_prop=label_ref)
    cou = Structures.from_file(coulomb_data, label_prop=label_coulomb)
    clean = subtract(ref, cou, label_ref=label_ref, label_coulomb=label_coulomb)
    clean.to_file(out_data, label_prop=label_ref)
    return clean
