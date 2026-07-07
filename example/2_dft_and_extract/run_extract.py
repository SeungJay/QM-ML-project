"""Stage 2 (part 2) — extract completed CP2K single-point DFT results into a .data file.

Each job directory has the coordinate XYZ (trajectory-input.xyz, an extended-XYZ
that carries its own cell in the `Lattice` header) and the CP2K forces output
(forces-output.xyz). The per-frame cell is read from the XYZ automatically, so no
cell argument is needed. Edit the settings and run:  python run_extract.py
"""

from dataprep import extract_dataset

extract_dataset(
    job_dirs="calculator-*",                 # glob (or a list of directories)
    coord_name="trajectory-input.xyz",       # ext-XYZ (carries per-frame Lattice)
    forces_name="forces-output.xyz",         # CP2K forces file in each job dir
    out_data="input-SR.data",                # combined reference dataset
)
print("wrote input-SR.data")

# ---- cell fallback -------------------------------------------------------
# The cell is taken from each XYZ's `Lattice` header (written by select_frames),
# so `cell` is normally NOT needed. Pass it only if your coordinate XYZs are
# plain (no Lattice). It applies to ALL frames (fixed cell), in Angstrom:
#
#   extract_dataset("calculator-*", out_data="input-SR.data",
#                   cell=(12.35, 12.834, 31.75))                 # 3 box lengths
#
#   extract_dataset("calculator-*", out_data="input-SR.data",
#                   cell=((12.35, 0.0, 0.0),                     # full 3x3 matrix
#                         (0.0, 12.834, 0.0),
#                         (0.0, 0.0, 31.75)))
#
#   extract_dataset("calculator-*", out_data="input-SR.data",
#                   cell="step-0.inp")           # parse &CELL from a CP2K input
