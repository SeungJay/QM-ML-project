# QMML

A QM/ML electrochemistry toolkit: build training data from DFT, and run
constant-potential QM/ML (MLFF + on-the-fly electrostatics) simulations.

It is a dependency-light (numpy + scipy) reimplementation of an older
`aml` + C/C++ toolchain — **no compilation, no FFTW, no `cp2k_input_tools`**.

## Packages

The distribution `QMML` installs four independently importable packages:

| package | what it does | import |
|---|---|---|
| `dataprep` | AIMD frame selection, DFT extraction, `.data` (RuNNer/n2p2) I/O, CP2K coulomb calc, charge decoupling, n2p2 committee training | `from dataprep import ...` |
| `cubetools` | Gaussian-cube I/O + arithmetic, sigmoid, dipole correction, planar average | `from cubetools import ...` |
| `poisson` | `chg2pot` Poisson solver (charge density → potential) | `from poisson import chg2pot` |
| `qmml_run` | in-process cube `pipeline` + the LAMMPS/CP2K QM/ML driver (`Config`, `run`) | `from qmml_run import ...` |

Dependencies: `poisson` uses `cubetools`; `qmml_run` uses both. `dataprep` is
standalone.

## The full workflow

```
 1  AIMD + frames       run_aimd()                  [example/1_aimd_and_frames/]
      │   trajectory-output.xyz + aimd.cell
      │   select_frames()  -> trajectory_inputs/
      ▼
 2  DFT + extract       run_singlepoints()          [example/2_dft_and_extract/]
      │   prepare_singlepoint_jobs() -> calculator-NNN/forces-output.xyz
      │   extract_dataset()  -> input-SR.data
      ▼
 3  charge decouple     run_coulomb() + decouple    [example/3_charge_decouple/]
      │   DFT-CES coulomb (CP2K↔LAMMPS); full DFT − coulomb -> input-SR-QMML.data
      ▼
 4  train committee     train_committee()           [example/4_train_committee/]
      │   n2p2 committee (short-range, non-ES target); ES re-added by mean-field
      │   coupling at run time. Uses the bundled n2p2 tools.
      ▼
 5  QM/ML run           run(Config(...))            [example/5_qmml_run/]
```

Example scripts for every step are in `example/` (see `example/README.md`).
CP2K MD and DFT are executed via `run_aimd()` / `run_singlepoints()`, which
launch CP2K through the `ProcessLauncher` (`mode="srun"`/`"mpirun"`). Run them
**inside a SLURM allocation** — a one-line `sbatch` wrapper is shown in
`example/README.md`.

## Install

```bash
git clone https://github.com/SeungJay/QM-ML-project.git
cd QM-ML-project
pip install .             # installs the Python packages (numpy, scipy pulled in)
```

`pip install .` installs the four Python packages only (`dataprep`, `cubetools`,
`poisson`, `qmml_run`) and the cube CLIs (`chg2pot`, `cube_add`, …). The two
external programs are set up separately (next section); the LAMMPS source lives
in the cloned repo, so cloning (not just `pip install`) is the intended route.

Repository layout:

```
QM-ML-project/
├── src/                              Python packages (installed by pip)
│   ├── dataprep/  cubetools/  poisson/  qmml_run/
├── example/                          run_*.py scripts
├── n2p2-v2.1.3-committee-nnp-extpot/ LAMMPS source — build with its README (`make`)
├── pyproject.toml   README.md
```

CP2K you install yourself.

## External programs (install separately)

QMML orchestrates two external codes; it does **not** bundle them. Install them
yourself and make them available in your run environment (e.g. `module load`),
then point the config at the executables (`cmd_cp2k=...`, LAMMPS executable in
`qmml_run.Config`). Committee training uses the n2p2 tools that come with the
bundled LAMMPS build (below) — no extra package.

- **CP2K** — runs the AIMD and the single-point / coulomb DFT. Developed against
  **CP2K 9.1**, but any version works as long as it supports the external
  potential (`&DFT/&EXTERNAL_POTENTIAL  READ_FROM_CUBE`). Give the binary via
  `cmd_cp2k="cp2k.psmp"` (on `PATH`) or a full path; no registration needed.

- **LAMMPS with n2p2 committee + external potential** — runs the MLFF MD in the
  constant-potential QM/ML loop (`qmml_run`). **Everything needed to install
  LAMMPS is in the bundled `n2p2-v2.1.3-committee-nnp-extpot/` folder** — you do
  not install LAMMPS separately. Follow that folder's `README.md`: two `make`
  steps (`src/` builds the n2p2 core; `src/interface/` downloads LAMMPS
  `stable_29Sep2021` + the DFT-CES patch and builds `bin/lmp_mpi`). Requires an
  MPI C++ compiler, GSL and Eigen3. Then point
  `qmml_run.Config(lmp_executable="…/n2p2-v2.1.3-committee-nnp-extpot/bin/lmp_mpi")`
  at the built binary. The same build also provides the n2p2 **training**
  binaries (`nnp-scaling`, `nnp-train`) used by stage 4 — put its `bin/` on
  `PATH`.

> These HPC codes are **not** installed by `pip` — they must be built against
> your cluster's MPI/toolchain (usually via `module load` + `make`). Building
> them inside `pip install` would link against the wrong libraries; keep them
> separate.

---

# Data preparation (`dataprep`)

All quantities are stored in **atomic units** (positions/cell in Bohr, energy in
Hartree, forces in Hartree/Bohr), matching the RuNNer/n2p2 `.data` convention.
`.data` files are read/written with no unit conversion.

### Run the AIMD (CP2K MD)

`run_aimd()` launches CP2K on your MD input through the launcher, producing
`trajectory-output.xyz` and the per-frame cell trajectory `aimd.cell`.
Launch it inside a SLURM allocation (`example/1_aimd_and_frames/run_aimd.py` is the settings + call):

```python
from dataprep import run_aimd, ProcessLauncher
run_aimd("aimd.inp", cmd_cp2k="cp2k.psmp",
         launcher=ProcessLauncher(mode="srun", n_core_task=64))
```

Pick the launcher with `mode="srun"`, `"mpirun"`, or `"plain"` (no MPI) — they
expand to `srun -n N` / `mpirun -np N`. `n_core_task` (N) must match the SLURM
`#SBATCH -n`.

> **Always enable cell output** — for **both NVT and NPT** — so the cell travels
> with every frame (one uniform scheme; for NVT it's simply constant). Add to
> your AIMD `&MOTION`:
> ```
> &MOTION
>   &PRINT
>     &CELL
>       FILENAME =aimd.cell
>       &EACH
>         MD 1
>       &END EACH
>     &END CELL
>   &END PRINT
> &END MOTION
> ```
> `FILENAME =aimd.cell` fixes the output name (otherwise it is `<PROJECT>-N.cell`).
> Frame selection (below) reads this file via `cell_file="aimd.cell"`.

### Select frames from the AIMD trajectory

Input: the multi-frame `trajectory-output.xyz` and the per-frame cell trajectory
`aimd.cell`. Output: one **extended-XYZ** per selected frame (each carries
its own `Lattice`), plus `frame_index_map.txt`.

```python
from dataprep import select_frames

select_frames("trajectory-output.xyz", "trajectory_inputs",
              cell_file="aimd.cell",           # CP2K per-frame cell (always)
              mode="every", every=20)            # or mode="random", n_random=140
```

The same call works for NVT and NPT — the cell just happens to be constant for
NVT. Each frame becomes an extended-XYZ carrying its own `Lattice="…"`, so every
downstream step gets the right cell automatically. (CP2K's XYZ reader ignores the
comment, so these files are still valid coordinate inputs.)

**Fixed box, no `.cell` file?** Pass `cell` instead of `cell_file` to bake a
constant Lattice into every frame (Angstrom):

```python
# 3 box lengths (orthorhombic):
select_frames("trajectory-output.xyz", "trajectory_inputs",
              cell=(12.35, 12.834, 31.75), mode="every", every=20)

# or a full 3x3 matrix (any cell):
select_frames("trajectory-output.xyz", "trajectory_inputs",
              cell=((12.35, 0, 0), (0, 12.834, 0), (0, 0, 31.75)),
              mode="every", every=20)
```

Enabling cell output (`cell_file`, above) is still the recommended, uniform path
(and the only correct one for NPT).

### Run the single-point DFT

First build the per-frame job directories (cheap, on a login node), then run
CP2K in each inside a SLURM allocation. Each job produces `forces-output.xyz`
(energy + forces, printed via `&FORCE_EVAL/&PRINT/&FORCES` or
`&MOTION/&PRINT/&FORCES`).

```python
from dataprep import prepare_singlepoint_jobs
prepare_singlepoint_jobs("trajectory_inputs", "step-0.inp",
                         job_prefix="calculator")   # -> calculator-NNN/{coord, step-0.inp}
```

`prepare_singlepoint_jobs` copies each frame's coords and writes a CP2K input
from your template with that frame's cell substituted into `&CELL` (keeping the
rest, including `&TOPOLOGY`, unchanged) — so every single point (NVT or NPT) uses
its own cell.

Then run CP2K in every job dir inside a SLURM allocation (via
`example/2_dft_and_extract/run_singlepoints.py`). `run_singlepoints()` loops over `calculator-*`
and **skips dirs that already have `forces-output.xyz`** — so it is safe to
resubmit until all frames finish.

### Extract DFT results into `.data`

Reads coordinates + the per-frame cell from each job's `trajectory-input.xyz`
(the `Lattice` header) and energy + forces from `forces-output.xyz`. Because the
cell travels with every frame, no cell argument is needed.

```python
from dataprep import extract_dataset

extract_dataset("calculator-*", out_data="input-SR.data")
```

**Fallback** — if your coordinate XYZs are plain (no `Lattice` header), pass
`cell` (applies to all frames, Angstrom):

```python
# 3 box lengths (orthorhombic):
extract_dataset("calculator-*", out_data="input-SR.data",
                cell=(12.35, 12.834, 31.75))

# full 3x3 matrix (any cell):
extract_dataset("calculator-*", out_data="input-SR.data",
                cell=((12.35, 0, 0), (0, 12.834, 0), (0, 0, 31.75)))

# or parse &CELL from a CP2K input file:
extract_dataset("calculator-*", out_data="input-SR.data", cell="step-0.inp")
```

### Charge decoupling (remove the electrostatics)

This step removes the **electrostatic (ES) energy and forces** from the full DFT
reference, so the MLFF is trained only on the **short-range, non-electrostatic**
part (the long-range ES is re-added by the mean-field coupling at run time).
Concretely: `input-SR-QMML = input-SR (full DFT) − coulomb(ES)`, per atom.

The coulomb (ES) term is computed by a **coupled CP2K ↔ LAMMPS DFT-CES
calculation** for each structure — not a single CP2K run. Per structure,
`run_coulomb()` does 7 steps: build inputs → initial CP2K (electrode only) →
`cube_multi` → initial LAMMPS (electrolyte charge density) → `chg2pot` → main
CP2K (electrode in the electrolyte potential) → `cube_multi` → final LAMMPS →
combine into the coulomb energy/forces (`summary.data`). CP2K treats the electrode
(carbon), LAMMPS the electrolyte as point charges; the two are coupled through the
electrostatic grid.

```python
from dataprep import prepare_jobs_from_data, run_coulomb, decouple_files, \
    ProcessLauncher

# 1. build coulomb job dirs straight from input-SR.data (geometry + cell are in it)
prepare_jobs_from_data("input-SR.data", "coulomb-step-0.inp",
                       jobs_dir="ref-calc-coulomb", job_prefix="calculator")

# 2. DFT-CES coulomb calc in each dir -> summary.data, gathered into QMML.data
run_coulomb("ref-calc-coulomb/calculator-*",
            cmd_cp2k="cp2k.psmp",
            cmd_lammps="…/n2p2-v2.1.3-committee-nnp-extpot/bin/lmp_mpi",
            launcher=ProcessLauncher(mode="mpirun", n_core_task=64),
            lammps_initial="lammps_initial.in", lammps_final="lammps_final.in",
            out_data="ref-calc-coulomb/QMML.data")

# 3. subtract: input-SR - QMML(coulomb) -> input-SR-QMML.data
decouple_files("input-SR.data", "ref-calc-coulomb/QMML.data",
               "input-SR-QMML.data")
```

`run_coulomb()` runs CP2K and LAMMPS through the launcher and uses `cube_multi`
(cubetools) and `chg2pot` (poisson) internally; it **skips dirs that already have
`summary.data`** (resumable). CP2K and LAMMPS need to be built and callable — if
they require different module environments, point `cmd_cp2k` / `cmd_lammps` at
wrapper scripts that set up their own environment. The compute inputs
(`coulomb-step-0.inp`, `lammps_initial.in`, `lammps_final.in`, the atom-type map)
are yours and define exactly what the coulomb term is; only the subtraction in
step 3 uses just the resulting energy/forces (`decouple_files` processes only the
frames that finished, so a partial run still yields a clean dataset).

### Train the committee NNP

`input-SR-QMML.data` is the training target. A committee is just `n_members`
ordinary n2p2 NNPs trained on it with different random seeds — no active
learning. `train_committee()` runs the bundled n2p2 tools (`nnp-scaling` then
`nnp-train`) once per member through the launcher, picks each member's best epoch
from its `learning-curve.out`, and assembles the `committee/nnp-data-1..N/`
layout LAMMPS expects.

```python
from dataprep import train_committee, ProcessLauncher

train_committee(
    "input-SR-QMML.data", "input.nn", out_dir="committee",
    elements=("O", "H", "C", "Na", "Cl"),   # fixed order
    n_members=8, n_epoch=100, n_bins=500, seed0=1,
    metric="force",                          # best epoch by test RMSE
    launcher=ProcessLauncher(mode="srun", n_core_task=128),
)
```

`input.nn` is the n2p2 template (`{n_elements}/{elements}/{seed}/{n_epoch}` are
filled per member; keep `write_weights_epoch 1`). Only the bundled n2p2 binaries
are used (on `PATH`); there is no external Python dependency. The resulting
`committee/` is what `qmml_run` loads at run time via `pair_style nnp dir …` in
the LAMMPS input. It is resumable — members already trained are skipped. See
`example/4_train_committee/`.

### Example scripts

Example scripts are in `example/`, one numbered folder per stage (see
`example/README.md`). Edit the settings at the top of each and run with `python`.

---

# Cube tools (`cubetools`, `poisson`)

```python
from cubetools import Cube, add, subtract, multiply, \
    sigmoid_profile, dipole_correction, planar_average
from poisson import chg2pot

chg = Cube.from_file("chgden.cube")    # Cube with .data as an (n0,n1,n2) ndarray
pot = chg2pot(chg)                      # Poisson solve: density -> potential
pot.to_file("pot.cube")

z, avg = planar_average(pot, axis=2)    # planar-averaged profile along z
added  = add(Cube.from_file("a.cube"), Cube.from_file("b.cube"))
corr   = dipole_correction(chg, axis=2, position=120)
```

`Cube.data` is a plain NumPy array, so custom operations are one-liners.

### Command-line tools

Same arguments and default output filenames as the original C/C++ binaries:

| command | usage | output |
|---|---|---|
| `chg2pot` | `chg2pot chgden.cube dir[1\|2\|3]` | `pot.cube`, `pot.{x,y,z}.avg` |
| `cube_add` | `cube_add a.cube b.cube` | `add.cube` |
| `cube_sub` | `cube_sub a.cube b.cube` | `subtracted.cube` |
| `cube_multi` | `cube_multi a.cube 2.0` | `multiplied.cube` |
| `cube_sigmoid` | `cube_sigmoid a.cube target z_ini z_fin steepness` | `z_sig.cube` |
| `cube_avg` | `cube_avg a.cube dir[1\|2\|3]` | `cube.{x,y,z}.avg` |
| `dipc` | `dipc chgden.cube dir[1\|2\|3] position` | `dipc.cube` |

### Fast binary I/O

For multi-step cube pipelines, keep intermediates in binary (~100x faster than
text) and only write `.cube` at the end for external tools:

```python
pot = chg2pot(Cube.from_file("chgden.cube"))
pot.save("pot.npz")                                  # fast intermediate
corr = dipole_correction(Cube.load("pot.npz"), axis=2, position=240)
corr.to_file("final.cube")                           # text only when needed
```

---

# QM/ML driver (`qmml_run`)

The constant-potential LAMMPS + CP2K cycle loop. `qmml_run.pipeline` provides
in-process replacements for the cube binaries (same output filenames), so the
driver needs no compiled cube tools. `example/5_qmml_run/run_qmml.py` is just configuration:

```python
from qmml_run import Config, run

cfg = Config(
    mpirun="srun", lmp_executable="...", cp2k_executable="cp2k.popt",
    steepness=0.4, anode_z_pos=..., cathod_z_pos=...,
    anode_grid=..., cathode_grid=...,
    runstep=100, heatstep=30000, savecube=1000, saverestart=100,
    max_cycles=10000, del_phi_0=...,
)
run(cfg)
```

Run with `python example/5_qmml_run/run_qmml.py` after filling in the values marked `REPLACE`.
Only the cube post-processing runs in-process; LAMMPS and CP2K stay as
subprocesses.

You can also call the pipeline directly:

```python
from qmml_run import pipeline as cube
cube.cube_multi("v_hartree.cube", 2)   # -> multiplied.cube
cube.dipc("MDrho.cube", 3, 10)         # -> dipc.cube
cube.chg2pot("dipc.cube", 3)           # -> pot.cube
cube.cube_avg("MDpot.cube", 3)         # -> cube.z.avg
```
