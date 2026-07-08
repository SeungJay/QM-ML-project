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
| `4_train_committee/` | `run_train.py`, `run.slurm` (+ `input.nn`) | train the n2p2 committee → `committee/nnp-data-1..N/` |
| `5_qmml_run/` | `run_qmml.py` (+ `base.cp2k.in`, `base.in.lammps`, `data.cp2k`, `data.lammps`, `run.slurm`) | constant-potential QM/ML run |

Each stage's `run.slurm` runs the whole stage in one submission. In stages 1 and
2 it runs the cheap follow-up script (`run_select_frames.py` / `run_extract.py`)
on a second line after the heavy one, so a single `sbatch` does everything (see
the SLURM section below).

The example inputs are for a NaCl-in-water / graphene-electrode system — edit the
cell, atom types, point charges, basis/potentials, etc. for your own system.

## Stage 4 — committee training (n2p2, no active learning)

A committee is just `n_members` ordinary n2p2 NNPs trained on the same data set
with different random seeds. `train_committee()` drives the **bundled** n2p2
tools (`nnp-scaling` then `nnp-train`) once per member, picks each member's best
epoch from its `learning-curve.out`, and assembles the directory LAMMPS wants:

```
committee/nnp-data-1/  input.nn  scaling.data  weights.<Z>.data ...
committee/nnp-data-2/  ...
```

No external Python package is needed — only the n2p2 binaries from the bundled
`n2p2-v2.1.3-committee-nnp-extpot` build must be on `PATH` (`run.slurm` sets it).
`input.nn` is the n2p2 template (its `{n_elements}/{elements}/{seed}/{n_epoch}`
are filled per member; keep `write_weights_epoch 1` so per-epoch weights exist to
select from).

**Before the QM/ML run (folder 5):** in `base.in.lammps`, point `pair_style nnp
dir "…/committee/"` at the committee trained in stage 4, and check the point
charges / `emap`. In `base.cp2k.in`, the `&GLOBAL PROJECT` + cube `FILENAME`s
determine the cube names that `run_qmml.py`'s `cp2k_V_file` / `cp2k_e_file` must
match. Submit with `run.slurm` (runs `python run_qmml.py`).

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
| 4 train committee | `input-SR-QMML.data` | `input.nn` (n2p2 template) | `committee/nnp-data-1..N/` |
| 5 QM/ML run | `committee/` (path in `base.in.lammps`) | `base.cp2k.in`, `base.in.lammps`, `data.cp2k`, `data.lammps` | the QM/ML run |

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
