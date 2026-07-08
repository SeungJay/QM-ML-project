"""Stage 1 (part 2) — pick frames from the AIMD trajectory.

Reads the multi-frame CP2K MD trajectory (trajectory-output.xyz) + its per-frame
cell trajectory (aimd.cell), and writes one extended-XYZ per selected frame into
trajectory_inputs/ (each carries its own Lattice) + frame_index_map.txt.

Cheap — run on a login node. The per-frame job dirs are built in stage 2.
Edit the settings and run:  python run_select_frames.py
"""

from dataprep import select_frames

# ---- how the per-frame cell is set (choose ONE) --------------------------
# (A) per-frame cell from CP2K's .cell trajectory  [recommended, needed for NPT]
CELL_KW = dict(cell_file="aimd.cell")          # FILENAME =aimd.cell in aimd.inp
#
# (B) a fixed box (NVT), no .cell file — give 3 lengths (Angstrom):
# CELL_KW = dict(cell=(12.35, 12.834, 31.75))
#     or a full 3x3 matrix (Angstrom), e.g. a non-orthorhombic cell:
# CELL_KW = dict(cell=((12.35, 0.0, 0.0), (0.0, 12.834, 0.0), (0.0, 0.0, 31.75)))
# --------------------------------------------------------------------------

select_frames(
    traj_xyz="trajectory-output.xyz",   # AIMD trajectory (multi-frame XYZ)
    outdir="trajectory_inputs",
    mode="every",        # "every" (evenly spaced) or "random"
    every=1,             # mode="every": take one frame every N steps
    n_random=140,        # mode="random": number of frames
    seed=77,             # mode="random": RNG seed
    skip_equil=0,        # drop this many leading equilibration frames
    **CELL_KW,
)
