"""QM/ML electrochemistry run — configuration + launch.

All logic lives in the qmml_run package. Values below are filled for the example
system (NaCl-in-water / 1-layer graphene, matching base.cp2k.in / base.in.lammps
/ data.cp2k / data.lammps in this folder). Edit for your system, then run:

    python run_qmml.py           # inside a SLURM allocation (see run.slurm)

`REPLACE` = set before a real run. Absolute executable paths are cluster-specific.
"""

from qmml_run import Config, run

cfg = Config(
    # ---------------------------------------------------------------- #
    # Executables / MPI launcher   (edit the paths for your cluster)
    # ---------------------------------------------------------------- #
    mpirun="srun",                                  # MPI launcher (srun / mpirun)
    lmp_executable="/work/e05/e05/jay/1.Project/8.NNP_EDL/1.Elyte_Bi_Graphenes/5.reproduce/QM-ML-project/n2p2-v2.1.3-committee-nnp-extpot/bin/lmp_mpi",  # built LAMMPS
    cp2k_executable="/work/y07/shared/apps/core/cp2k/cp2k-9.1.0/exe/ARCHER2/cp2k.popt",  # CP2K binary

    # ---------------------------------------------------------------- #
    # Input templates / CP2K output cube names
    #   the cube names = CP2K PROJECT + FILENAME (from base.cp2k.in), so they
    #   must match your base.cp2k.in &GLOBAL PROJECT and &V_HARTREE_CUBE /
    #   &E_DENSITY_CUBE FILENAME.
    # ---------------------------------------------------------------- #
    lammpsInput="base.in.lammps",                   # LAMMPS input template
    cp2kInput="base.cp2k.in",                       # CP2K input template
    cp2k_V_file="confined-NaClWatCarb-revPBE-D3-V_hartree.cube-v_hartree-1_0.cube",
    cp2k_e_file="confined-NaClWatCarb-revPBE-D3-Val_den.cube-ELECTRON_DENSITY-1_0.cube",

    # ---------------------------------------------------------------- #
    # Dipole correction
    # ---------------------------------------------------------------- #
    dipc_dir="3",                                   # dipole axis: 1=x, 2=y, 3=z
    dipc_grid="10",                                 # grid index for the correction plane
    cp2k_dipc_pos=0.0,                              # CP2K dipole position (Angstrom)
    try_diff_dip=200,                               # retries if dipole correction errors out
    d_dipc_pos=0.01,                                # retry step size (Angstrom)

    # ---------------------------------------------------------------- #
    # Potentiostat (sigmoidal potential adjustment)
    # ---------------------------------------------------------------- #
    potentiostat=True,                              # apply sigmoidal potential correction
    steepness=0.4,                                  # sigmoid steepness (0.4 recommended)
    # z of the anode/cathode planes where the sigmoid potential ramps. These are
    # SYSTEM-SPECIFIC (values here are from the 4-layer production run). Choose
    # them so the sigmoid's sloped (transition) region does NOT overlap the
    # electrode charge density — the ramp must sit where there is no electrode
    # electron density, otherwise the applied potential distorts the electrode.
    anode_z_pos=14.18,                              # anode plane z (bohr)
    cathod_z_pos=45.825855,                         # cathode plane z (bohr)
    prefactor=1.0,                                  # scales the sigmoidal function

    # ---------------------------------------------------------------- #
    # Potential-difference reading (planar-averaged profile grid indices)
    # ---------------------------------------------------------------- #
    # Grid indices on the z-averaged profile where the converged potential is
    # read to get del_phi. SYSTEM-SPECIFIC. Pick indices that (a) do NOT overlap
    # either electrode's charge and (b) sit where the potential is FLAT (a
    # plateau) — check the z-averaged profile and confirm both before trusting
    # the potential difference.
    anode_grid=150,                                 # flat-region grid index near anode
    cathode_grid=530,                               # flat-region grid index near cathode

    # ---------------------------------------------------------------- #
    # Running parameters
    # ---------------------------------------------------------------- #
    runstep=100,                                    # MD steps per production cycle
    heatstep=30000,                                 # MD steps for the initial heating run
    savecube=1000,                                  # keep QM grid every N cycles (else deleted)
    saverestart=100,                                # save a restart every N cycles
    max_cycles=10000,                               # total number of QM/ML cycles
    del_phi_0=1.0,                                  # target potential (V)

    # ---------------------------------------------------------------- #
    # Continuation run
    # ---------------------------------------------------------------- #
    ini_cycle=0,                                    # starting cycle (0 = fresh run)
    sum_net_phi_0=0.0,                              # if ini_cycle > 0: previous "will_apply" value (V)
    start_from="run_mode",                          # "qm" | "ml" | "qmpp" | "mlpp" | "run_mode"

    # ---------------------------------------------------------------- #
    # Debugging
    # ---------------------------------------------------------------- #
    debugging=False,                                # if True, keep extra intermediate cubes
)

if __name__ == "__main__":
    run(cfg)
