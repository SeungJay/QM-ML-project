# n2p2 (committee) + DFT-CES for LAMMPS

This repository is a distribution of **n2p2 v2.1.3** extended with two features and
a build system that sets up the LAMMPS interface automatically:

1. **Committee neural network potentials (C-NNP).** On-the-fly committee
   disagreement of energies and forces inside LAMMPS, active-learning stopping
   criteria, and the `nnp-comp2` tool. This is *not* part of the official n2p2
   releases. The methodology follows the C-NNP work of C. Schran, K. Brezina and
   O. Marsalek.
2. **DFT-CES coupling.** A `grid` command and a `fix gridforce` in LAMMPS that let
   classical MD run in an external QM electrostatic grid (DFT-CES, originally by
   H.-K. Lim, M-design group @ KAIST). These are added to LAMMPS through a patch.

The n2p2 core code here is unchanged committee-n2p2 (v2.1.3). Only the LAMMPS
interface build was reworked (see below); nothing else was modified.

---

## Why this repository exists

The committee-enabled n2p2 v2.1.3 used here was obtained from GitHub while we were
developing this work. That original repository no longer appears to be available,
so we redistribute the version we used here — unmodified in its n2p2 core — for
reproducibility and so that others can build and use it. The committee code is not
part of the official n2p2 releases (see the Credits below for the upstream projects
and methodology it derives from).

---

## Requirements

- A C++ compiler and GNU Make
- **MPI** providing `mpic++` / `mpicxx`
  (e.g. `conda install -c conda-forge openmpi`, or `brew install open-mpi`)
- GSL and Eigen3 (needed by n2p2)
- `wget` **or** `curl`, plus `tar` and `patch`
- Internet access on the **first** interface build (LAMMPS + the DFT-CES patch are
  downloaded automatically)

---

## Build

### 1. Build the n2p2 core library and tools

```bash
cd src
make
```

This produces `lib/libnnp*.a` and the `bin/nnp-*` tools (including `nnp-comp2`).

### 2. Build the LAMMPS interface (committee NNP + DFT-CES)

```bash
cd src/interface
make
```

Running `make` here performs the whole setup automatically:

1. downloads LAMMPS `stable_29Sep2021` from `github.com/lammps/lammps`
2. extracts it as `./lammps-nnp`
3. downloads the DFT-CES patch (`stable_29Sep2021.patch`) from
   `github.com/SeungJay/DFT-CES`
4. applies the patch — adds `grid.cpp/.h`, `fix_gridforce.cpp/.h` and modifies
   `domain.*`, `input.*`, `run.cpp` (enables the `grid` command and `fix gridforce`)
5. links n2p2 (`lammps-nnp/lib/nnp`) and installs the committee NNP pair style
   (`USER-NNP` → `pair_nnp`)
6. sets the MPI compiler in `MAKE/Makefile.mpi`
7. builds with the `user-nnp`, `molecule` and `kspace` packages → `lmp_mpi`,
   copied to `bin/`
8. removes the downloaded archive (`stable_29Sep2021.tar.gz`) and patch
   (`stable_29Sep2021.patch`)

When it finishes it prints a summary of exactly these steps.

The resulting `bin/lmp_mpi` supports `pair_style nnp` (with committee-disagreement
keywords), the `grid` command, `fix gridforce`, and `kspace_style pppm` /
`pair_style coul/long` for long-range electrostatics.

To rebuild from scratch:

```bash
cd src/interface
make clean && make
```

---

## Notes

- `src/interface/makefile` is the automated build described above.
- LAMMPS is **not** vendored in this repository — it is fetched at build time.
  This keeps the repo small; the trade-off is that the first build needs a network
  connection.
- The build was verified for the LAMMPS `stable_29Sep2021` release. `stable_29Oct2020`
  is avoided because of a `gather_atoms` bug
  ([matsci thread](https://matsci.org/t/lammps-users-typeerror-on-gather-atoms-in-python-examples/39076)).

---

## Credits

- **n2p2** — Andreas Singraber, University of Vienna:
  <https://github.com/CompPhysVienna/n2p2>
- **Committee NNP (C-NNP)** — C. Schran, K. Brezina, O. Marsalek,
  *J. Chem. Phys.* **153**, 104105 (2020); see also
  <https://github.com/MarsalekGroup/aml>
- **DFT-CES** — H.-K. Lim et al. (M-design group, KAIST); patch:
  <https://github.com/SeungJay/DFT-CES>

Original n2p2 README is preserved as [`README.upstream.md`](README.upstream.md).

## License

GPL-3.0-or-later, following n2p2. See [`LICENSE`](LICENSE).
