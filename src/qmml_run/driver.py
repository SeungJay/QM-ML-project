"""QM/ML electrochemistry driver (LAMMPS + CP2K cycle loop).

All the logic that used to live at module scope in ``qmml.py`` is here: the
helper functions and the main cycle loop (:func:`run`). Cube post-processing
runs in-process via :mod:`qmml_run.pipeline`; LAMMPS/CP2K and file shuffling
run as subprocesses, exactly as before.

Your ``qmml.py`` becomes just configuration plus a run call::

    from qmml_run import Config, run

    cfg = Config(
        steepness=0.4,
        anode_z_pos=...,      # bohr
        cathod_z_pos=...,     # bohr
        anode_grid=...,
        cathode_grid=...,
        runstep=100,
        heatstep=30000,
        savecube=1000,
        saverestart=100,
        max_cycles=10000,
        del_phi_0=...,        # V
    )
    run(cfg)
"""

from __future__ import annotations

import sys
import subprocess
from dataclasses import dataclass

from . import pipeline as cube

# constants
Ha2eV = 27.211386245988


@dataclass
class Config:
    """All inputs for a QM/ML run. Fields left as ``None`` must be set by you."""

    # executables / MPI launcher
    mpirun: str = "srun"
    lmp_executable: str = "n2p2-committee-nnp/bin/lmp_mpi"
    cp2k_executable: str = "cp2k.popt"

    # input templates / CP2K output cube names
    lammpsInput: str = "base.in.lammps"
    cp2kInput: str = "base.cp2k.in"
    cp2k_V_file: str = "v_hartree-1_0.cube"
    cp2k_e_file: str = "ELECTRON_DENSITY-1_0.cube"

    # dipole correction
    dipc_dir: str = "3"
    dipc_grid: str = "10"          # need not match the CP2K dipole
    cp2k_dipc_pos: float = 0.0     # dipole position for CP2K (Angstrom)
    try_diff_dip: int = 200        # retries if dipole correction errors out
    d_dipc_pos: float = 0.01       # step size for the retry position (Angstrom)

    # potentiostat
    potentiostat: bool = True
    steepness: float = None        # ~0.4 recommended
    anode_z_pos: float = None      # bohr
    cathod_z_pos: float = None     # bohr
    prefactor: float = 1.0         # scales the sigmoidal function

    # potential-difference reading
    anode_grid: int = None         # vacuum grid index at anode
    cathode_grid: int = None       # vacuum grid index at cathode

    # running parameters
    runstep: int = None
    heatstep: int = None
    savecube: int = None           # save QM grid every this many steps
    saverestart: int = None        # save restart every this many steps
    max_cycles: int = None
    del_phi_0: float = None        # target potential (V)

    # continuation run
    ini_cycle: int = 0
    sum_net_phi_0: float = 0.0     # if ini_cycle > 0, use previous will_apply value
    start_from: str = "run_mode"   # "qm" | "ml" | "qmpp" | "mlpp" | "run_mode"

    # debugging
    debugging: bool = False


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------
def run_lammps(cfg: Config, input_file, cycle):
    subprocess.run(
        [cfg.mpirun, cfg.lmp_executable, "-in", input_file],
        stdout=open(f"summary/lammps_log/log_{cycle}cycle.txt", "w"),
        stderr=subprocess.STDOUT,
        check=False,
    )


def check_lammps_success(output_file):
    success_indicator = "Total wall time:"
    with open(output_file, 'r') as file:
        for line in file:
            if success_indicator in line:
                return True
    return False


def run_cp2k(cfg: Config, input_file, cycle):
    subprocess.run(
        [cfg.mpirun, cfg.cp2k_executable, "-i", input_file],
        stdout=open(f"summary/cp2k_log/log_{cycle}cycle.txt", "w"),
        stderr=subprocess.STDOUT,
        check=False,
    )


def check_cp2k_success(output_file):
    success_indicator = "PROGRAM ENDED AT"
    with open(output_file, 'r') as file:
        for line in file:
            if success_indicator in line:
                return True
    return False


def check_cp2k_dipole(output_file):
    success_indicator = "Dipole correction needs more vacuum space above the surface"
    with open(output_file, 'r') as file:
        for line in file:
            if success_indicator in line:
                return False
    return True


def read_pot_diff(file_path, anode_grid, cathode_grid):
    diffs = []
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for i in range(len(lines) - 1):
                value1 = float(lines[i].split()[1])
                value2 = float(lines[i + 1].split()[1])
                diff = value2 - value1
                diffs.append(diff)

            pot_i = None
            pot_f = None

            for i in range(anode_grid, 0, -1):
                if abs(diffs[i]) <= 2e-6:
                    pot_i = lines[i].split()[1]
                    break

            for i in range(cathode_grid, len(lines)):
                if abs(diffs[i]) <= 2e-6:
                    pot_f = lines[i].split()[1]
                    break

            if pot_i is None or pot_f is None:
                raise ValueError("Potential values not found within the specified grids.")

        return float(pot_f) - float(pot_i)
    except Exception as e:
        with open('QMMLoutput.txt', 'a') as f:
            f.write(f"Error in read_pot_diff: {e}\n")
        return None


def save_last_cycle(cycle):
    savelist = ["lammps_input.in", "runlammps.xyz", "qmml.restart", "cp2k_input.in"]

    for file in savelist:
        subprocess.run(["cp", file, f"summary/last_cycle/{file}"], check=True)

    subprocess.run(["cp", f"V_ryd_{cycle}.cube", "summary/last_cycle/V_ryd_last.cube"], check=True)
    subprocess.run(["cp", f"val_{cycle}.cube", "summary/last_cycle/val_last.cube"], check=True)

    cubelist = ["MDpot.cube", "dipc.cube", "MDrho.cube", "z_sig.cube"]
    for file in cubelist:
        subprocess.run(["cp", file, f"summary/last_cycle/{file}"], check=True)


# --------------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------------
def run(cfg: Config):
    """Run the full QM/ML cycle loop using the given configuration."""

    directories = ["trj", "txt", "cube", "restart", "avg",
                   "lammps_log", "cp2k_log", "last_cycle"]
    for directory in directories:
        subprocess.run(["mkdir", "-p", f"summary/{directory}"], check=True)

    # mutable run state
    sum_net_phi = cfg.sum_net_phi_0 / Ha2eV
    start_from = cfg.start_from

    for cycle in range(cfg.ini_cycle, cfg.max_cycles):
        if start_from == "run_mode" or start_from == "qm":
            message = f"Starting cycle {cycle} of {cfg.max_cycles} \n"
            with open('QMMLoutput.txt', 'a') as f:
                f.write(message)

        # Heat LAMMPS
        if cycle == 0:
            message = "Heating LAMMPS..."
            with open('QMMLoutput.txt', 'a') as f:
                f.write(message)
            subprocess.run(["cp", cfg.lammpsInput, "lammps_heat.in"], check=True)
            subprocess.run(["sed", "-i", f"s/run STEP/run {cfg.heatstep}/", "lammps_heat.in"], check=True)
            subprocess.run(["sed", "-i", "s/fix             solvGrid elyte gridforce -1 1//", "lammps_heat.in"], check=True)
            subprocess.run(["sed", "-i", "s/fix_modify      solvGrid energy yes//", "lammps_heat.in"], check=True)
            subprocess.run(["sed", "-i", "s/grid            V_file.cube//", "lammps_heat.in"], check=True)
            subprocess.run(["sed", "-i", "s/f_solvGrid//", "lammps_heat.in"], check=True)
            subprocess.run(["sed", "-i", "s/fix NVT elyte nvt temp 300 300 100.0/fix NVT elyte nvt temp 10 300 100.0/", "lammps_heat.in"], check=True)

            run_lammps(cfg, 'lammps_heat.in', cycle)
            subprocess.run(["mv", "Total.lammpstrj", "summary/heat.lammpstrj"], check=True)
            subprocess.run(["mv", "committee-energy.txt", "summary/txt/committee-energy_heat.txt"], check=True)

            # Check if LAMMPS was successful
            subprocess.run(["cat", f"summary/lammps_log/log_{cycle}cycle.txt"], stdout=open("temp.output", "w"), check=True)
            if check_lammps_success("temp.output") == False:
                message = "LAMMPS calculation failed. Check the output file for details.\n"
                with open('QMMLoutput.txt', 'a') as f:
                    f.write(message)
                sys.exit(1)

            message = "Done.\n"
            with open('QMMLoutput.txt', 'a') as f:
                f.write(message)

        # Run CP2K
        if start_from == "qm" or start_from == "run_mode":
            message = "Running CP2K..."
            with open('QMMLoutput.txt', 'a') as f:
                f.write(message)

            subprocess.run(["cp", cfg.cp2kInput, "cp2k_input.in"], check=True)
            if cycle != 0:
                subprocess.run(["sed", "-i", "s/COORD_FILE_NAME data.cp2k/COORD_FILE_NAME runlammps.xyz/", "cp2k_input.in"], check=True)
                subprocess.run(["sed", "-i", "/&DFT/a \    &EXTERNAL_POTENTIAL\\\n        READ_FROM_CUBE T\\\n    &END EXTERNAL_POTENTIAL", "cp2k_input.in"], check=True)
            subprocess.run(["sed", "-i", f"s/SURF_DIP_POS Dipc_position/SURF_DIP_POS {cfg.cp2k_dipc_pos}/", "cp2k_input.in"], check=True)
            run_cp2k(cfg, 'cp2k_input.in', cycle)

            # Check if CP2K was successful
            subprocess.run(["cat", f"summary/cp2k_log/log_{cycle}cycle.txt"], stdout=open("temp.output", "w"), check=True)
            if check_cp2k_success("temp.output") == False and check_cp2k_dipole("temp.output") == True:
                message = "CP2K calculation failed. Check the output file for details.\n"
                with open('QMMLoutput.txt', 'a') as f:
                    f.write(message)
                sys.exit(1)

            elif check_cp2k_success("temp.output") == False and check_cp2k_dipole("temp.output") == False:
                message = "CP2K calculation failed. Check the output file for details. It may be due to dipole correction. \n"
                with open('QMMLoutput.txt', 'a') as f:
                    f.write(message)

                for n_try_dip in range(0, cfg.try_diff_dip):
                    message = f"Trying again with different dipole correction position: {n_try_dip + 1} / {cfg.try_diff_dip}\n"
                    with open('QMMLoutput.txt', 'a') as f:
                        f.write(message)

                    subprocess.run(["cp", cfg.cp2kInput, "cp2k_input.in"], check=True)
                    if cycle != 0:
                        subprocess.run(["sed", "-i", "s/COORD_FILE_NAME data.cp2k/COORD_FILE_NAME runlammps.xyz/", "cp2k_input.in"], check=True)
                        subprocess.run(["sed", "-i", "/&DFT/a \    &EXTERNAL_POTENTIAL\\\n        READ_FROM_CUBE T\\\n    &END EXTERNAL_POTENTIAL", "cp2k_input.in"], check=True)
                    temp_cp2k_dipc_pos = cfg.cp2k_dipc_pos + (n_try_dip + 1) * cfg.d_dipc_pos
                    subprocess.run(["sed", "-i", f"s/SURF_DIP_POS Dipc_position/SURF_DIP_POS {temp_cp2k_dipc_pos}/", "cp2k_input.in"], check=True)

                    run_cp2k(cfg, 'cp2k_input.in', cycle)

                    # Check if CP2K was successful
                    subprocess.run(["cat", f"summary/cp2k_log/log_{cycle}cycle.txt"], stdout=open("temp.output", "w"), check=True)
                    if check_cp2k_success("temp.output") == True:
                        message = f"The updated dipole position is {temp_cp2k_dipc_pos} Angstrom.\n"
                        with open('QMMLoutput.txt', 'a') as f:
                            f.write(message)
                        break

            message = "Done..."
            with open('QMMLoutput.txt', 'a') as f:
                f.write(message)
            start_from = "run_mode"

        # Post-process CP2K
        if start_from == "qmpp" or start_from == "run_mode":
            # convert V_file to Ryd units
            cube.cube_multi(cfg.cp2k_V_file, 2)  # to make Ryd units
            subprocess.run(["mv", "multiplied.cube", f"V_ryd_{cycle}.cube"], check=True)
            subprocess.run(["mv", cfg.cp2k_e_file, f"val_{cycle}.cube"], check=True)

            cube.cube_avg(cfg.cp2k_V_file, 3)
            subprocess.run(["mv", "cube.z.avg", f"summary/avg/qmV_{cycle}_au.avg"], check=True)
            cube.cube_avg(f"val_{cycle}.cube", 3)
            subprocess.run(["mv", "cube.z.avg", f"summary/avg/val_{cycle}.avg"], check=True)

            # Calculate the instant potential difference
            if cycle > 0:
                cube.cube_sub("MDpot.cube", cfg.cp2k_V_file)  # cp2k potential is inverse physical convention
                cube.cube_avg("subtracted.cube", 3)

                del_phi = read_pot_diff('cube.z.avg', cfg.anode_grid, cfg.cathode_grid)
            else:
                cube.cube_avg(cfg.cp2k_V_file, 3)

                dp = read_pot_diff('cube.z.avg', cfg.anode_grid, cfg.cathode_grid)
                del_phi = -1.0 * dp if dp is not None else None  # inverse convention

            if del_phi is not None:
                net_phi = (del_phi - cfg.del_phi_0 / Ha2eV)  # in Ha units
                message = f"del_phi0: {cfg.del_phi_0 :.5f} V, del_phi: {del_phi * Ha2eV :.5f} V, deviates: {net_phi * Ha2eV :.5f} V \n"
                with open('QMMLoutput.txt', 'a') as f:
                    f.write(message)
            else:
                message = ("Potential values not found: no flat point within the "
                           f"anode_grid={cfg.anode_grid} / cathode_grid={cfg.cathode_grid} "
                           "search ranges of cube.z.avg. Set these to flat, "
                           "non-electrode grid indices for your system (both must be "
                           "< the number of lines in cube.z.avg).\n")
                with open('QMMLoutput.txt', 'a') as f:
                    f.write(message)
                sys.exit()

            start_from = "run_mode"

        # Run LAMMPS
        if start_from == "ml" or start_from == "run_mode":
            message = "Running LAMMPS..."
            with open('QMMLoutput.txt', 'a') as f:
                f.write(message)

            subprocess.run(["cp", cfg.lammpsInput, "lammps_input.in"], check=True)
            subprocess.run(["sed", "-i", "/read_data /c\\read_restart qmml.restart", "lammps_input.in"], check=True)
            subprocess.run(["sed", "-i", f"s/run STEP/run {cfg.runstep}/", "lammps_input.in"], check=True)
            subprocess.run(["sed", "-i", f"s/grid            V_file.cube/grid            V_ryd_{cycle}.cube/", "lammps_input.in"], check=True)

            run_lammps(cfg, 'lammps_input.in', cycle)
            subprocess.run(["mv", "Total.lammpstrj", f"summary/trj/qmml{cycle}cycle.lammpstrj"], check=True)
            subprocess.run(["mv", "committee-energy.txt", f"summary/txt/committee-energy{cycle}cycle.txt"], check=True)

            # Check if LAMMPS was successful
            subprocess.run(["cat", f"summary/lammps_log/log_{cycle}cycle.txt"], stdout=open("temp.output", "w"), check=True)
            if check_lammps_success("temp.output") == False:
                message = "LAMMPS calculation failed. Check the output file for details."
                with open('QMMLoutput.txt', 'a') as f:
                    f.write(message)
                sys.exit(1)

            message = "Done..."
            with open('QMMLoutput.txt', 'a') as f:
                f.write(message)
            start_from = "run_mode"

        # Post-process LAMMPS
        if start_from == "mlpp" or start_from == "run_mode":

            # Potential process
            cube.dipc("MDrho.cube", cfg.dipc_dir, cfg.dipc_grid)  # follow physical convention
            cube.chg2pot("dipc.cube", 3)  # generate pot.cube in Ha units
            subprocess.run(["mv", "pot.cube", "MDpot.cube"], check=True)
            if cfg.debugging:
                subprocess.run(["cp", "MDpot.cube", f"summary/cube/MDpot{cycle}.cube"], check=True)
            cube.cube_avg("MDpot.cube", 3)
            subprocess.run(["mv", "cube.z.avg", f"summary/avg/elyt_V_{cycle}_au.avg"], check=True)

            # Save the last cycle
            # save_last_cycle(cycle)

            if cycle % cfg.savecube != 0:
                subprocess.run(["rm", "-f", f"V_ryd_{cycle}.cube"], check=True)
                subprocess.run(["rm", "-f", f"val_{cycle}.cube"], check=True)
            else:
                subprocess.run(["mv", f"V_ryd_{cycle}.cube", f"summary/cube/V_ryd_{cycle}.cube"], check=True)
                subprocess.run(["mv", f"val_{cycle}.cube", f"summary/cube/val_{cycle}.cube"], check=True)
            if cycle % cfg.saverestart == 0:
                subprocess.run(["cp", "qmml.restart", f"summary/restart/qmml{cycle}cycle.restart"], check=True)

            # Calculate the instant potential difference
            if cfg.potentiostat:
                cube.cube_sub("MDpot.cube", cfg.cp2k_V_file)  # cp2k potential is inverse physical convention
                cube.cube_avg("subtracted.cube", 3)

                del_phi = read_pot_diff('cube.z.avg', cfg.anode_grid, cfg.cathode_grid)
                if del_phi is not None:
                    net_phi = (del_phi - cfg.del_phi_0 / Ha2eV)  # in Ha units
                    sum_net_phi += net_phi  # initial value is sum_net_phi_0
                    message = f"del_phi0: {cfg.del_phi_0 :.5f} V, del_phi: {del_phi * Ha2eV :.5f} V, deviates: {net_phi * Ha2eV :.5f} V, will_apply: {sum_net_phi * Ha2eV :.5f} V  \n"
                    with open('QMMLoutput.txt', 'a') as f:
                        f.write(message)
                    # MDpot.cube is only read for its header; the sigmoid
                    # potential compensates the deviated potential (potentiostat)
                    cube.cube_sigmoid("MDpot.cube", cfg.prefactor * sum_net_phi, cfg.anode_z_pos, cfg.cathod_z_pos, cfg.steepness)
                    cube.cube_add("z_sig.cube", "MDpot.cube")
                    subprocess.run(["mv", "add.cube", "pot.cube"], check=True)
                    cube.cube_avg("z_sig.cube", 3)
                    subprocess.run(["mv", "cube.z.avg", f"summary/avg/z_sig_{cycle}_au.avg"], check=True)

                    if cfg.debugging:
                        subprocess.run(["cp", "z_sig.cube", f"summary/cube/z_sig{cycle}.cube"], check=True)

                else:
                    message = "Potential values not found.\n"
                    with open('QMMLoutput.txt', 'a') as f:
                        f.write(message)
                    sys.exit()
            else:
                subprocess.run(["cp", "MDpot.cube", "pot.cube"], check=True)

            message = f"Cycle {cycle} completed.\n"
            with open('QMMLoutput.txt', 'a') as f:
                f.write(message)

            start_from = "run_mode"

    message = "Simulation completed.\n"
    with open('QMMLoutput.txt', 'a') as f:
        f.write(message)
