# Example scripts

Copy these into your working directory, edit the settings at the top of each,
and run in order. They import the installed `QMML` packages.

Five numbered folders (run in order). Each holds its run script(s) plus the
example CP2K/LAMMPS/n2p2 inputs it needs:

| folder | scripts | does |
|---|---|---|
| `1_aimd_and_frames/` | `run_aimd.py` + `run.slurm`, `run_select_frames.py` (+ `aimd.inp`, `trajectory-input.xyz`) | AIMD → `trajectory-output.xyz` + `aimd.cell`, then pick frames → `trajectory_inputs/` |
| `2_dft_and_extract/` | `run_singlepoints.py` + `run.slurm`, `run_extract.py` (+ `step-0.inp`) | build job dirs + per-frame single-point DFT → `calculator-NNN/`, then extract → `input-SR.data` |
| `3_charge_decouple/` | `run_decouple.py` + `run.slurm` (+ `coulomb-step-0.inp`, `lammps_initial.in`, `lammps_final.in`) | DFT-CES coulomb (CP2K↔LAMMPS) + subtract → `input-SR-QMML.data` |
| `4_train_committee/` | `run_train.py`, `run.slurm` (+ `input.nn`) | train the n2p2 committee → `committee/` (member 0 at top level + `nnp-data-1..N-1/`) |
| `5_qmml_run/` | `run_qmml.py` (+ `base.cp2k.in`, `base.in.lammps`, `data.cp2k`, `data.lammps`, `run.slurm`) | constant-potential QM/ML run |

Each stage's `run.slurm` runs the whole stage in one submission. In stages 1 and
2 it runs the cheap follow-up script (`run_select_frames.py` / `run_extract.py`)
on a second line after the heavy one, so a single `sbatch` does everything (see
the SLURM section below).

The example inputs are for a NaCl-in-water / graphene-electrode system — edit the
cell, atom types, point charges, basis/potentials, etc. for your own system.

## Passing files between folders

The stages share files, so the simplest setup is **one working directory** where
outputs accumulate. If you prefer a **separate directory per folder**, carry
forward exactly — **① carried in** (from the previous folder), **② you add**
(inputs you supply), **③ produced** (outputs):

| folder | ① carried in | ② you add | ③ produced |
|---|---|---|---|
| 1 AIMD + frames | — | `aimd.inp`, `trajectory-input.xyz` (initial config) | `trajectory-output.xyz`, `aimd.cell`, `trajectory_inputs/` |
| 2 DFT + extract | `trajectory_inputs/` | `step-0.inp` | `calculator-NNN/` (+ `forces-output.xyz`), `input-SR.data` |
| 3 charge decouple | `input-SR.data` | `coulomb-step-0.inp`, `lammps_initial.in`, `lammps_final.in` | `ref-calc-coulomb/QMML.data`, `input-SR-QMML.data` |
| 4 train committee | `input-SR-QMML.data` | `input.nn` (n2p2 template) | `committee/` (member 0 top level + `nnp-data-1..N-1/`) |
| 5 QM/ML run | `committee/` (path in `base.in.lammps`) + initial structure from the training system (see note) | `base.cp2k.in`, `base.in.lammps` | the QM/ML run |

If you keep each stage in its own folder, copy the **① carried in** item from the
previous stage before submitting (run from the `example/` directory):

```bash
# 1 -> 2 : the selected frames
cp -r 1_aimd_and_frames/trajectory_inputs  2_dft_and_extract/

# 2 -> 3 : the short-range DFT dataset
cp 2_dft_and_extract/input-SR.data  3_charge_decouple/

# 3 -> 4 : the charge-decoupled (non-ES) training set
cp 3_charge_decouple/input-SR-QMML.data  4_train_committee/

# 4 -> 5 : the trained committee (or just point base.in.lammps at its path)
cp -r 4_train_committee/committee  5_qmml_run/
```

Running all stages in **one working directory** avoids these copies entirely —
the outputs simply accumulate in place.

Notes:
- **CP2K data files** (`GTH_BASIS_SETS`, `BASIS_MOLOPT`, `POTENTIAL`, `dftd3.dat`)
  must be visible to every CP2K run — put them in the run dir or set
  `CP2K_DATA_DIR` in your SLURM script.
- Folder 2 builds `calculator-NNN/` from `trajectory_inputs/` (using `step-0.inp`)
  then runs CP2K in each. Folder 3 rebuilds its own `ref-calc-coulomb/` job dirs
  from `input-SR.data` (geometry + cell are stored in the dataset), so it needs
  only `input-SR.data`.
- `frame_index_map.txt` (in `trajectory_inputs/`) records which original AIMD
  frame each `NNN` came from.
- **Stage 5 initial structure** (`data.cp2k` = electrode, `data.lammps` = full
  electrode + electrolyte cell) is the configuration the production QM/ML run
  starts from. It is **taken from the training system** — e.g. an equilibrated
  frame of the stage-1 AIMD — not an unrelated input, so the committee potential
  is applied to the same chemistry it was trained on. Prepare these two files
  from that frame (any tool, e.g. OVITO) and keep the cell / atom types / charges
  consistent with `base.cp2k.in` and `base.in.lammps` (`emap`). The `data.cp2k` /
  `data.lammps` shipped here are an example of such a structure — replace them
  with your own. (There is no conversion script; do this once, by hand.)

## Choosing the MPI launcher

`ProcessLauncher(mode=...)` builds the MPI-launch prefix. `mode` is either a
built-in name — `"srun"` → `srun -n NCORES`, `"mpirun"` → `mpirun -np NCORES`,
`"plain"` → no prefix — or a **callable** `(i_slot, n_core_task, size_node) ->
prefix string` for full control.

**These templates are configured for UK ARCHER2** (see below), so the scripts use
a callable that adds the recommended ARCHER2 srun flags (the built-in `"srun"`
preset does not):

```python
import os

# core count comes from the SLURM allocation (nodes x tasks-per-node), so
# run.slurm is the single source of truth; the fallback is only for manual runs
NCORES = int(os.environ.get("SLURM_NTASKS", 128))

def archer2_srun(i_slot, n_core_task, size_node):
    return (f"srun --hint=nomultithread --distribution=block:block "
            f"-n {n_core_task} ")

launcher = ProcessLauncher(mode=archer2_srun, n_core_task=NCORES)
```

On another cluster, drop the callable and use `mode="srun"` / `"mpirun"` / a
callable with your own flags.

## Submitting to SLURM

> **These `run.slurm` templates are written for UK ARCHER2** (HPE Cray EX):
> `--account=e05-react-wal`, `--partition=standard --qos=standard`, the ARCHER2
> `module load` lines (`cp2k/cp2k-9.1.0` for CP2K stages; `cpe/22.12` + `gsl` +
> `eigen` for the LAMMPS/n2p2 stages), and the central CP2K binary
> `/work/y07/shared/apps/core/cp2k/cp2k-9.1.0/exe/ARCHER2/cp2k.popt`. On another
> cluster, edit the `#SBATCH` header, the `module load` lines, and the executable
> paths. Placeholders marked `<user>` / `REPLACE` (venv path, your n2p2 build
> location) must be filled in before submitting.

The heavy steps must run inside a SLURM allocation, so **each stage ships its own
`run.slurm`** — one submission runs the whole stage:

| stage | submit | runs |
|---|---|---|
| 1 | `sbatch 1_aimd_and_frames/run.slurm` | `run_aimd.py` (CP2K MD) → `run_select_frames.py` |
| 2 | `sbatch 2_dft_and_extract/run.slurm` | `run_singlepoints.py` (single-point DFT) → `run_extract.py` |
| 3 | `sbatch 3_charge_decouple/run.slurm` | `run_decouple.py` (CP2K↔LAMMPS coulomb) |
| 4 | `sbatch 4_train_committee/run.slurm` | `run_train.py` (n2p2 training) |
| 5 | `sbatch 5_qmml_run/run.slurm` | `run_qmml.py` (QM/ML run) |

In stages 1 and 2 the single `run.slurm` runs both steps in sequence, each on its
own line — the heavy script (`run_aimd.py` / `run_singlepoints.py`) then the cheap
one (`run_select_frames.py` / `run_extract.py`). If you'd rather run the cheap
step by hand, delete/comment its line and just `python run_select_frames.py` /
`python run_extract.py` on a login node afterwards.

Each `run.slurm` is a template: edit the `#SBATCH` header, the `module load`
lines, and the `PATH` to your CP2K / LAMMPS / n2p2 binaries. The `.py` scripts
read the core count from `SLURM_NTASKS` (= `nodes` × `tasks-per-node`), so the
allocation in `run.slurm` is the single source of truth — no need to keep a
separate `NCORES` in sync. Several steps are resumable —
`run_singlepoints.py` skips `calculator-NNN/` dirs with `forces-output.xyz`,
`run_decouple.py` skips dirs with `summary.data`, and `run_train.py` skips
committee members already trained — so you can just resubmit until everything
finishes.
