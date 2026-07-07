# Build environment — UK ARCHER2

Verified on ARCHER2 (project e05). Load these modules on a **login node**,
then run `make` here (`src/`) and in `src/interface/`.

```bash
module load cpe/22.12
module load cray-fftw/3.3.10.3
module load cray-python
module load gsl
module load eigen
```

Notes:
- The compiler is the Cray wrapper `CC` (set in `makefile.gnu`); BLAS for
  `-DEIGEN_USE_BLAS` comes from cray-libsci (default with cpe).
- Build on a **login node** — `src/interface` downloads LAMMPS + the DFT-CES
  patch from GitHub, and ARCHER2 compute nodes have no external network.
- If `CC` does not resolve to a working C++ compiler, `module load PrgEnv-gnu`
  first.
