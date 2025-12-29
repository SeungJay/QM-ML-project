import os
import sys
import subprocess

mpirun="srun"
lmp_executable="n2p2-committee-nnp/bin/lmp_mpi" 
cp2k_executable="cp2k.popt"
lammpsInput="base.in.lammps"
cp2kInput="base.cp2k.in"
cp2k_V_file="v_hartree-1_0.cube"
cp2k_e_file="ELECTRON_DENSITY-1_0.cube"

cube_multi="/cube_multi"
cube_add="/cube_add"
cube_sub="/cube_sub"
cube_avg="/cube_avg"
chg2pot="/chg2pot"

dipc2="/dipc"
dipc_dir="3"
dipc_grid="10" # mannually set (don't need to be matched with cp2k dipc) 
cp2k_dipc_pos = 0.0 # mannually set (dipole position for CP2K; Angstrom units) 
try_diff_dip = 200 # mannually set (number of tries for dipole correction position if default errors out)
d_dipc_pos = 0.01 # mannually set (step size for dipole correction position; Angstrom units)

# for potentiostat
potentiostat = True # if True, the potential will be adjusted by the sigmoidal function
cube_sig="/cube_sigmoid"
steepness= # mannually set 0.4 is recommedend
anode_z_pos =  # bohr units
cathod_z_pos = # bohr units
prefactor = 1.0 # mannually set (to scale sigmoidal function)

#for read potential difference
anode_grid =     # vacuum at anode atomic position grid
cathode_grid =   # vacuum at cathode atomic position grid

# running parameters
runstep= #100
heatstep= #30000
savecube= #1000 # save qm grid every this step.
saverestart= #100 # save restart every this step.
max_cycles = #10000  # Define the maximum number of cycles
del_phi_0 = #potentialvalue # mannually set in V unit

#for the continuation run
ini_cycle = 0
sum_net_phi_0 = 0.0 # if ini_cycle >0, this must be the value from previous cycle value (will_apply: ... V)
start_from = "run_mode" # "qm" or "ml" or "qmpp" or "mlpp" or "run_mode" > 0

#debugging
debugging = False

# constants
Ha2eV = 27.211386245988
sum_net_phi = sum_net_phi_0 / Ha2eV

directories = ["trj", "txt", "cube", "restart", "avg","lammps_log","cp2k_log","last_cycle"]
for directory in directories:
    #os.system(f"mkdir -p summary/{directory}")
    subprocess.run(["mkdir", "-p", f"summary/{directory}"], check=True)

def run_lammps(input_file, cycle):
    # Replace 'lmp_executable' with the path to your LAMMPS executable
    # and 'input_file' with the path to your LAMMPS input file.
    subprocess.run(
        [mpirun, lmp_executable, "-in", input_file],
        #stdout=open("summary/log.lammps.QMML", "a"),
        stdout=open(f"summary/lammps_log/log_{cycle}cycle.txt", "w"),
        stderr=subprocess.STDOUT,
        #check=True
        check=False
    )

def check_lammps_success(output_file):
    success_indicator = "Total wall time:"
    with open(output_file, 'r') as file:
        for line in file:
            if success_indicator in line:
                return True
    return False

def run_cp2k(input_file, cycle):
    # Replace 'cp2k_executable' with the path to your CP2K executable
    # and 'input_file' with the path to your CP2K input file.
    subprocess.run(
        [mpirun, cp2k_executable, "-i", input_file],
        #stdout=open("summary/log.cp2k.QMML", "a"),
        stdout=open(f"summary/cp2k_log/log_{cycle}cycle.txt", "w"),
        stderr=subprocess.STDOUT,
        #check=True
        check=False
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

for cycle in range(ini_cycle,max_cycles):
    if start_from == "run_mode" or start_from == "qm":
        message = f"Starting cycle {cycle} of {max_cycles} \n"
        with open('QMMLoutput.txt', 'a') as f:
            f.write(message)

    # Heat LAMMPS
    if cycle == 0:
        message = "Heating LAMMPS..."
        with open('QMMLoutput.txt', 'a') as f:
            f.write(message)
        subprocess.run(["cp", lammpsInput, "lammps_heat.in"], check=True)
        subprocess.run(["sed", "-i", f"s/run STEP/run {heatstep}/", "lammps_heat.in"], check=True)
        subprocess.run(["sed", "-i", "s/fix             solvGrid elyte gridforce -1 1//", "lammps_heat.in"], check=True)
        subprocess.run(["sed", "-i", "s/fix_modify      solvGrid energy yes//", "lammps_heat.in"], check=True)
        subprocess.run(["sed", "-i", "s/grid            V_file.cube//", "lammps_heat.in"], check=True)
        subprocess.run(["sed", "-i", "s/f_solvGrid//", "lammps_heat.in"], check=True)
        subprocess.run(["sed", "-i", "s/fix NVT elyte nvt temp 300 300 100.0/fix NVT elyte nvt temp 10 300 100.0/", "lammps_heat.in"], check=True)
    
        run_lammps('lammps_heat.in',cycle)
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
    if  start_from == "qm" or start_from=="run_mode":
        message = "Running CP2K..."
        with open('QMMLoutput.txt', 'a') as f:
            f.write(message)

        subprocess.run(["cp", cp2kInput, "cp2k_input.in"], check=True)
        if cycle != 0:
            subprocess.run(["sed", "-i", "s/COORD_FILE_NAME data.cp2k/COORD_FILE_NAME runlammps.xyz/", "cp2k_input.in"], check=True)
            subprocess.run(["sed", "-i", "/&DFT/a \    &EXTERNAL_POTENTIAL\\\n        READ_FROM_CUBE T\\\n    &END EXTERNAL_POTENTIAL", "cp2k_input.in"], check=True)
        subprocess.run(["sed", "-i", f"s/SURF_DIP_POS Dipc_position/SURF_DIP_POS {cp2k_dipc_pos}/", "cp2k_input.in"], check=True)
        run_cp2k('cp2k_input.in',cycle)

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

            for n_try_dip in range(0, try_diff_dip):
                message = f"Trying again with different dipole correction position: {n_try_dip + 1} / {try_diff_dip}\n"
                with open('QMMLoutput.txt', 'a') as f:
                    f.write(message)
                
                subprocess.run(["cp", cp2kInput, "cp2k_input.in"], check=True)
                if cycle != 0:
                    subprocess.run(["sed", "-i", "s/COORD_FILE_NAME data.cp2k/COORD_FILE_NAME runlammps.xyz/", "cp2k_input.in"], check=True)
                    subprocess.run(["sed", "-i", "/&DFT/a \    &EXTERNAL_POTENTIAL\\\n        READ_FROM_CUBE T\\\n    &END EXTERNAL_POTENTIAL", "cp2k_input.in"], check=True)
                temp_cp2k_dipc_pos = cp2k_dipc_pos + (n_try_dip+1) * d_dipc_pos
                subprocess.run(["sed", "-i", f"s/SURF_DIP_POS Dipc_position/SURF_DIP_POS {temp_cp2k_dipc_pos}/", "cp2k_input.in"], check=True)
                
                run_cp2k('cp2k_input.in',cycle)

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
    if start_from == "qmpp" or start_from=="run_mode":
        # convert V_file to Ryd units    
        subprocess.run([cube_multi, cp2k_V_file, "2"], check=True) # to make Ryd units
        subprocess.run(["mv", "multiplied.cube", f"V_ryd_{cycle}.cube"], check=True)
        subprocess.run(["mv", cp2k_e_file, f"val_{cycle}.cube"], check=True)

        subprocess.run([cube_avg, cp2k_V_file, "3"], check=True)
        subprocess.run(["mv", "cube.z.avg", f"summary/avg/qmV_{cycle}_au.avg"], check=True)
        subprocess.run([cube_avg, f"val_{cycle}.cube", "3"], check=True)
        subprocess.run(["mv", "cube.z.avg", f"summary/avg/val_{cycle}.avg"], check=True)

        # Calculate the instant potential difference
        if cycle > 0:
            subprocess.run([cube_sub, "MDpot.cube", cp2k_V_file], check=True) # since cp2k potential is inverse physical convention
            subprocess.run([cube_avg, "subtracted.cube", "3"], check=True)

            del_phi = read_pot_diff('cube.z.avg', anode_grid, cathode_grid )
        else: 
            subprocess.run([cube_avg, cp2k_V_file, "3"], check=True)

            del_phi = -1.0 * read_pot_diff('cube.z.avg', anode_grid, cathode_grid ) # since cp2k potential is inverse physical convention

        if del_phi is not None:
            net_phi = ( del_phi - del_phi_0 / Ha2eV )  # in Ha units
            message = f"del_phi0: {del_phi_0 :.5f} V, del_phi: {del_phi * Ha2eV :.5f} V, deviates: {net_phi * Ha2eV :.5f} V \n"
            with open('QMMLoutput.txt', 'a') as f:
                f.write(message)
        else:
            message = "Potential values not found.\n"
            with open('QMMLoutput.txt', 'a') as f:
                f.write(message)
            sys.exit()

        start_from = "run_mode"

    # Run LAMMPS
    if start_from == "ml" or start_from=="run_mode":
        message = "Running LAMMPS..."
        with open('QMMLoutput.txt', 'a') as f:
            f.write(message)

        subprocess.run(["cp", lammpsInput, "lammps_input.in"], check=True)
        subprocess.run(["sed", "-i", "/read_data /c\\read_restart qmml.restart", "lammps_input.in"], check=True)
        subprocess.run(["sed", "-i", f"s/run STEP/run {runstep}/", "lammps_input.in"], check=True)
        subprocess.run(["sed", "-i", f"s/grid            V_file.cube/grid            V_ryd_{cycle}.cube/", "lammps_input.in"], check=True)
    
        run_lammps('lammps_input.in',cycle)
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
    if start_from == "mlpp" or start_from=="run_mode":

        # Potential process
        subprocess.run([dipc2, "MDrho.cube", dipc_dir, dipc_grid], check=True) # to follow the physical convention
        subprocess.run([chg2pot, "dipc.cube", "3"], check=True) # generate pot.cube in Ha units
        subprocess.run(["mv", "pot.cube", "MDpot.cube"], check=True)
        if debugging:
            subprocess.run(["cp", "MDpot.cube", f"summary/cube/MDpot{cycle}.cube"], check=True)
        subprocess.run([cube_avg, "MDpot.cube", "3"], check=True)
        subprocess.run(["mv", "cube.z.avg", f"summary/avg/elyt_V_{cycle}_au.avg"], check=True)

        # Save the last cycle
        #save_last_cycle(cycle)

        if cycle % savecube != 0:
            subprocess.run(["rm", "-f", f"V_ryd_{cycle}.cube"], check=True)
            subprocess.run(["rm", "-f", f"val_{cycle}.cube"], check=True)
        else:
            subprocess.run(["mv", f"V_ryd_{cycle}.cube", f"summary/cube/V_ryd_{cycle}.cube"], check=True)
            subprocess.run(["mv", f"val_{cycle}.cube", f"summary/cube/val_{cycle}.cube"], check=True)
        if cycle % saverestart == 0:
            subprocess.run(["cp", "qmml.restart", f"summary/restart/qmml{cycle}cycle.restart"], check=True)

        # Calculate the instant potential difference
        if potentiostat:
            subprocess.run([cube_sub, "MDpot.cube", cp2k_V_file], check=True) # since cp2k potential is inverse physical convention
            subprocess.run([cube_avg, "subtracted.cube", "3"], check=True)

            del_phi = read_pot_diff('cube.z.avg', anode_grid, cathode_grid )
            if del_phi is not None:
                net_phi = ( del_phi - del_phi_0 / Ha2eV )  # in Ha units
                sum_net_phi += net_phi # initial sum_net_phi is set as sum_net_phi_0 in the head of the script
                message = f"del_phi0: {del_phi_0 :.5f} V, del_phi: {del_phi * Ha2eV :.5f} V, deviates: {net_phi * Ha2eV :.5f} V, will_apply: {sum_net_phi * Ha2eV :.5f} V  \n"
                with open('QMMLoutput.txt', 'a') as f:
                    f.write(message)
                # to adjust the potential, MDpot.cube is just to read the header
                # the sigmoid potential is to compensate the deviated potential (potentiostat)
                subprocess.run([cube_sig, "MDpot.cube", f"{prefactor * sum_net_phi:.10f}", str(anode_z_pos), str(cathod_z_pos), str(steepness)], check=True)
                subprocess.run([cube_add, "z_sig.cube", "MDpot.cube"], check=True)
                subprocess.run(["mv", "add.cube", "pot.cube"], check=True)
                subprocess.run([cube_avg, "z_sig.cube", "3"], check=True)
                subprocess.run(["mv", "cube.z.avg", f"summary/avg/z_sig_{cycle}_au.avg"], check=True)

                if debugging:
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
