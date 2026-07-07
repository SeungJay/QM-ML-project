"""dataprep — DFT data extraction and charge decoupling for QM/ML training.

Part of the QMML distribution. A dependency-free (numpy-only) reimplementation
of the pieces of the old ``aml`` package used to prepare training data:

  * ``.data`` (RuNNer/n2p2) structures I/O — :class:`Structures`, :class:`Structure`
  * AIMD trajectory frame selection (multi-frame XYZ -> per-frame XYZ)
  * DFT single-point extraction (CP2K forces + coordinate XYZ + cell -> .data)
  * a CP2K single-point calculator for the coulomb/electrostatic term
  * charge decoupling (subtract coulomb energy/forces from the DFT reference)

Typical workflow::

    from dataprep import select_frames, extract_dataset, CP2K, \
        ProcessLauncher, decouple_inplace, Structures

    BOX = (12.35, 12.834, 31.75)   # AIMD cell (Angstrom), from &CELL ABC

    # 0. pick frames from the AIMD trajectory -> per-frame coordinate XYZs
    select_frames("trajectory-output.xyz", "trajectory_inputs",
                  mode="every", every=20)
    #    (then run the single-point CP2K on each frame -> calculator-NNN/)

    # 1. extract DFT single points into one .data
    data = extract_dataset("calculator-*", cell=BOX, out_data="input-SR.data")

    # 2. coulomb-only CP2K over the same structures
    def srun(i, n, s): return f"srun -n {n} "
    calc = CP2K("coulomb.inp", cmd_cp2k="cp2k.popt",
                launcher=ProcessLauncher(mode=srun, n_core_task=128))
    calc.run(data, label_prop="coulomb")

    # 3. decouple and write the short-range reference
    decouple_inplace(data)
    data.to_file("input-LR.data", label_prop="reference")
"""

from __future__ import annotations

from .structures import Structure, Structures, Property
from .frames import (
    select_frames, read_xyz_frames, write_xyz, read_cell_trajectory,
    parse_lattice,
)
from .extract import (
    extract_structure, extract_dataset, read_cp2k_forces, read_xyz,
    read_cell_from_cp2k_input,
)
from .decouple import subtract, decouple_inplace, decouple_files
from .launcher import ProcessLauncher
from .cp2k import (
    CP2K, set_cell, inject_cell_and_topology, prepare_singlepoint_jobs,
    prepare_jobs_from_data, run_aimd, run_singlepoints,
)
from .dftces import (
    run_coulomb, create_dftces_inputs, read_coulomb_result, DEFAULT_TYPE_MAP,
)

__all__ = [
    "Structure", "Structures", "Property",
    "select_frames", "read_xyz_frames", "write_xyz",
    "read_cell_trajectory", "parse_lattice",
    "extract_structure", "extract_dataset", "read_cp2k_forces", "read_xyz",
    "read_cell_from_cp2k_input",
    "subtract", "decouple_inplace", "decouple_files",
    "ProcessLauncher", "CP2K", "set_cell", "inject_cell_and_topology",
    "prepare_singlepoint_jobs", "prepare_jobs_from_data",
    "run_aimd", "run_singlepoints",
    "run_coulomb", "create_dftces_inputs", "read_coulomb_result",
    "DEFAULT_TYPE_MAP",
]
