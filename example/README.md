# Example scripts

Copy these into your working directory, edit the settings at the top of each,
and run in order. They import the installed `QMML` packages.

Four numbered folders (run in order). Each holds its run script(s) plus the
example CP2K/LAMMPS inputs it needs:

| folder | scripts | does |
|---|---|---|
| `1_aimd_and_frames/` | `run_aimd.py`, `run_select_frames.py` (+ `aimd.inp`, `trajectory-input.xyz`) | AIMD → `trajectory-output.xyz` + `aimd.cell`, then pick frames → `trajectory_inputs/` |
| `2_dft_and_extract/` | `run_singlepoints.py`, `run_extract.py` (+ `step-0.inp`) | build job dirs + per-frame single-point DFT → `calculator-NNN/`, then extract → `input-SR.data` |
| `3_charge_decouple/` | `run_decouple.py` (+ `coulomb-step-0.inp`, `lammps_initial.in`, `lammps_final.in`) | DFT-CES coulomb (CP2K↔LAMMPS) + subtract → `input-SR-QMML.data` |
| `4_qmml_run/` | `run_qmml.py` (+ `base.cp2k.in`, `base.in.lammps`, `data.cp2k`, `data.lammps`, `run.slurm`) | constant-potential QM/ML run |

Within a folder, `run_aimd.py` / `run_singlepoints.py` are SLURM jobs (heavy DFT);
`run_select_frames.py` / `run_extract.py` are cheap (login node). Between folders 2
and 3, train the MLFF (n2p2, external) on `input-SR-QMML.data`.

The example inputs are for a NaCl-in-water / graphene-electrode system — edit the
cell, atom types, point charges, basis/potentials, etc. for your own system.

**Before the QM/ML run (folder 4):** in `base.in.lammps`, set the `pair_style nnp
dir "…/final_model/"` path to your **trained committee NNP**, and check the point
charges / `emap`. In `base.cp2k.in`, the `&GLOBAL PROJECT` + cube `FILENAME`s
determine the cube names that `run_qmml.py`'s `cp2k_V_file` / `cp2k_e_file` must
match. Submit with `run.slurm` (a SLURM wrapper that runs `python run_qmml.py`).

## Passing files between folders

The stages share files, so the simplest setup is **one working directory** where
outputs accumulate. If you prefer a **separate directory per folder**, carry
forward exactly — **① carried in** (from the previous folder), **② you add**
(inputs you supply), **③ produced** (outputs):

| folder | ① carried in | ② you add | ③ produced |
|---|---|---|---|
| 1 AIMD + frames | — | `aimd.inp`, `trajectory-input.xyz` (initial config) | `trajectory-output.xyz`, `aimd.cell`, `trajectory_inputs/` |
| 2 DFT + extract | `trajectory_inputs/` | `step-0.inp` | `calculator-NNN/` (+ `forces-output.xyz`), `input-SR.data` |
| — MLFF train | `input-SR-QMML.data`* | (n2p2 training config, external) | committee NNP potential |
| 3 charge decouple | `input-SR.data` | `coulomb-step-0.inp`, `lammps_initial.in`, `lammps_final.in` | `ref-calc-coulomb/QMML.data`, `input-SR-QMML.data` |
| 4 QM/ML run | trained NNP (path in `base.in.lammps`) | `base.cp2k.in`, `base.in.lammps`, `data.cp2k`, `data.lammps` | the QM/ML run |

\* the MLFF training happens after folder 3 (it consumes `input-SR-QMML.data`).

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

Scripts that run CP2K/LAMMPS expose a toggle at the top:

```python
LAUNCHER = "srun"    # "srun" | "mpirun" | "plain"
NCORES   = 64        # must match the SLURM  #SBATCH -n
```

`"srun"` → `srun -n NCORES`, `"mpirun"` → `mpirun -np NCORES`, `"plain"` → no
prefix.

## Submitting to SLURM

The DFT/LAMMPS steps must run inside a SLURM allocation. A minimal wrapper:

```bash
#!/bin/bash
#SBATCH -J qmml
#SBATCH -n 64                 # must match NCORES in the .py
#SBATCH -p long
#SBATCH --time 120:00:00
#SBATCH -o %x.o.%j
#SBATCH -e %x.e.%j

source ~/.bashrc
# load your CP2K / LAMMPS modules here

cd $SLURM_SUBMIT_DIR
python 1_aimd_and_frames/run_aimd.py   # or 2_dft_and_extract/run_singlepoints.py,
                                        #    3_charge_decouple/run_decouple.py,
                                        #    4_qmml_run/run_qmml.py
```

Then `sbatch that_script.sh`. `run_singlepoints.py` is resumable — it skips
`calculator-NNN/` dirs that already have `forces-output.xyz`, so you can resubmit
until all frames finish.
