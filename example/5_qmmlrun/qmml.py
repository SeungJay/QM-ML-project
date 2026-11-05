"""
qmml.py -- CP2K + LAMMPS QM/ML driver, constant applied bias,
            chained through the 20-minute short QoS.

Each cycle is split into stages (heat -> qm -> qmpp -> ml -> mlpp).  After
every stage the next one is recorded in qmml.progress.json.  Before starting a
stage the script checks the time left in the job (QMML_DEADLINE, set by
run.pbs); if the stage probably will not fit, it exits with code 99 and
run.pbs resubmits itself.  The next job resumes from qmml.progress.json, so
the resume point is no longer set by hand.

BEFORE YOU SUBMIT, SET THESE (search for "CHECK:"):
  1. max_cycles, runstep
  2. fixed_bias_V                              -- applied bias, volts
  3. anode_grid, cathode_grid                  -- plateau read points
  4. anode_z_pos, cathod_z_pos                 -- step position

Every cycle the step profile of amplitude fixed_bias_V is added to MDpot.cube
and handed to CP2K as pot.cube.  There is no feedback: del_phi is only
measured (post-SCF diagnostic) and logged.
"""

import os
import sys
import subprocess

mpirun="srun"
#lmp_executable="/work/e05/e05/jay/0.programs/DFT-CES/DFT-CES_MD/lammps-11Aug17/src/lmp_mpi"
lmp_executable="/work/e05/e05/jay/0.programs/n2p2-committee-nnp/bin/lmp_mpi"
cp2k_executable="/work/e05/e05/jay/0.programs/cp2k-9.1/exe/ARCHER2/cp2k.popt"
lammpsInput="base.in.lammps"
cp2kInput="base.cp2k.in"
cp2k_V_file="confined-NaClWatCarb-revPBE-D3-V_hartree.cube-v_hartree-1_0.cube"
cp2k_e_file="confined-NaClWatCarb-revPBE-D3-Val_den.cube-ELECTRON_DENSITY-1_0.cube"

cube_multi="/work/e05/e05/jay/script/cube_multi"
cube_add="/work/e05/e05/jay/script/cube_add"
cube_sub="/work/e05/e05/jay/script/cube_sub"
cube_avg="/work/e05/e05/jay/script/cube_avg"
chg2pot="/work/e05/e05/jay/0.programs/poisson/chg2pot"

dipc2="/work/e05/e05/jay/0.programs/poisson/dipc2"
dipc_dir="3"
dipc_grid="10" # mannually set (don't need to be matched with cp2k dipc)
cp2k_dipc_pos = 0.0 # mannually set (dipole position for CP2K; Angstrom units)
try_diff_dip = 200 # mannually set (number of tries for dipole correction position if default errors out)
d_dipc_pos = 0.01 # mannually set (step size for dipole correction position; Angstrom units)

# ======================================================================
# APPLIED BIAS
# ======================================================================
# STEP-FUNCTION PROFILE.  cube_step takes (cube, amp_Ha, z_step) and writes
# the profile to z_sig.cube (renamed to bias_step.cube, built once per run).
# The step sits midway between the electrodes.
cube_sig="/work/e05/e05/jay/script/cube_step"
anode_z_pos = 14.78 # bohr units
cathod_z_pos = 27.42 # bohr units
z_step_pos   = (anode_z_pos + cathod_z_pos) / 2.0   # = 21.1 bohr

# Constant applied bias (volts).  Never updated during the run.
fixed_bias_V = biasvalue  # CHECK

# On a dipole-correction abort, retry with progressively gentler Broyden
# mixing instead of moving the sigmoid / dipole plane.
#
# base.cp2k.in carries the placeholder "ALPHA   Alpha_value"; the value below is
# substituted in for every CP2K run, so this is the only place ALPHA is set.
# The retry sweep starts one step BELOW alpha_base -- a retry at the base value
# would just repeat the setting that already failed.  Values <= 0 are dropped,
# so raising n_try_alpha too far is harmless.
alpha_base  = 0.1      # ALPHA used for the normal run
d_alpha     = 0.0125   # step down on each retry
n_try_alpha = 7        # -> 0.0875, 0.075, 0.0625, 0.05, 0.0375, 0.025, 0.0125
retry_alphas = [a for a in (round(alpha_base - (n + 1) * d_alpha, 6)
                            for n in range(n_try_alpha)) if a > 0]

#for read potential difference
anode_grid =  165   # anode atomic position grid
cathode_grid = 314 # cathode atomic position grid

# running parameters
runstep=runstepvalue
heatstep=30000
savecube=1000 # save qm grid every this step.
saverestart=100 # save restart every this step.
max_cycles = maxcyclevalue  # CHECK

# ----------------------------------------------------------------------
# short-QoS chaining
# ----------------------------------------------------------------------
import json
import time
progress_file = "qmml.progress.json"
STAGES = ("heat", "qm", "qmpp", "ml", "mlpp")
EXIT_RESUBMIT = 99          # run.pbs resubmits on this exit code only
# Unix time by which this job must have finished; unset -> no limit.
job_deadline = float(os.environ.get("QMML_DEADLINE", "inf"))
# Guessed stage durations (s) until a real one has been measured.
default_stage_s = dict(heat=600, qm=900, qmpp=60, ml=120, mlpp=60)
stage_margin = 1.2          # start a stage only if left > margin*estimate + safety_s
safety_s = 60
max_kills = 2               # give up if the same stage is killed this often

#debugging
debugging = False

# constants
Ha2eV = 27.211386245988

directories = ["trj", "txt", "cube", "restart", "avg","lammps_log","cp2k_log"]
for directory in directories:
    subprocess.run(["mkdir", "-p", f"summary/{directory}"], check=True)


def log(message):
    with open('QMMLoutput.txt', 'a') as f:
        f.write(message)


def load_progress():
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            return json.load(f)
    # Fresh run: reuse a pre-made heat (cp.sh copies ../heat/qmml.restart
    # here) instead of heating again; the heat is bias- and k-point-independent.
    first = "qm" if os.path.exists("qmml.restart") else "heat"
    return dict(cycle=0, stage=first, running=None, kills=0, durations={})


def save_progress():
    tmp = progress_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump(prog, f, indent=1)
    os.replace(tmp, progress_file)     # atomic: never a half-written file


def stage_due(cycle, stage):
    """Heat only on cycle 0; on the resumed cycle, skip stages already done."""
    if stage == "heat" and cycle != 0:
        return False
    if cycle > resume_cycle:
        return True
    return STAGES.index(stage) >= STAGES.index(resume_stage)


def enter_stage(cycle, stage):
    """Exit for resubmission if `stage` probably won't finish in this job.
    The first stage of a job always runs, so the chain always progresses."""
    global ran_any
    # Never estimate below the default: a CP2K run that restarted from an
    # already-converged wfn (cycle 0, or a rerun after a kill) finishes in a
    # minute or two and would otherwise make a normal SCF look short.
    past = prog["durations"].get(stage, [])
    need = stage_margin * max(past + [default_stage_s[stage]]) + safety_s
    left = job_deadline - time.time()
    if ran_any and left < need:
        log(f"[chain] {left:.0f} s left < {need:.0f} s needed for {stage} "
            f"of cycle {cycle}; exiting for resubmission.\n")
        sys.exit(EXIT_RESUBMIT)
    prog["running"] = stage
    prog["t0"] = time.time()
    save_progress()
    ran_any = True


def leave_stage(stage, next_cycle, next_stage):
    dt = time.time() - prog.pop("t0")
    prog["durations"][stage] = (prog["durations"].get(stage, []) + [round(dt, 1)])[-5:]
    prog.update(cycle=next_cycle, stage=next_stage, running=None, kills=0)
    save_progress()


def run_lammps(input_file, cycle):
    subprocess.run(
        [mpirun, lmp_executable, "-in", input_file],
        stdout=open(f"summary/lammps_log/log_{cycle}cycle.txt", "w"),
        stderr=subprocess.STDOUT,
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
    subprocess.run(
        [mpirun, cp2k_executable, "-i", input_file],
        stdout=open(f"summary/cp2k_log/log_{cycle}cycle.txt", "w"),
        stderr=subprocess.STDOUT,
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

def check_cube(path, ref):
    """Guard against a truncated cube.

    cube_add / cube_sigmoid / chg2pot / dipc2 all write with fprintf and never
    check fprintf or fclose, so a short write (quota or ENOSPC) leaves a
    half-written cube and the tool still exits 0 -- subprocess check=True
    cannot see it.  CP2K then dies much later with
    "Bad value during floating point read".  Catch it here instead.
    """
    if not os.path.exists(path):
        msg = f"ERROR: {path} was not created.\n"
    elif open(path, 'rb').read()[-1:] != b'\n':
        msg = f"ERROR: {path} is truncated (no trailing newline, {os.path.getsize(path)} bytes). Out of disk space?\n"
    elif os.path.getsize(path) < 0.99 * os.path.getsize(ref):
        msg = (f"ERROR: {path} is truncated ({os.path.getsize(path)} bytes vs "
               f"{os.path.getsize(ref)} for {ref}). Out of disk space?\n")
    else:
        return True
    log(msg)
    sys.stderr.write(msg)
    sys.exit(1)


# ----------------------------------------------------------------------
# reading planar-averaged potentials
# ----------------------------------------------------------------------
def _read_avg(file_path):
    vals = []
    with open(file_path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 2:
                vals.append(float(p[1]))
    return vals

# Progressive tolerances for the plateau search.  The original code used a
# single 2e-6 and called sys.exit() when no flat point was found before the
# scan ran off the end of the array -- one bad frame then threw away the whole
# run (a 2455-cycle job died exactly this way, as
# "Error in read_pot_diff: list index out of range").  A frame whose profile is
# momentarily rough is not a reason to stop; widen the tolerance, and fall back
# to a window average, before giving up.
PLATEAU_TOLS = (2e-6, 1e-5, 5e-5, 2e-4)
PLATEAU_FALLBACK_WIDTH = 10     # grid points averaged if no plateau is found


def _find_plateau(vals, start, direction, tol):
    """Walk outward from `start` until the profile goes flat -- i.e. we are
    inside the metal, where the planar average no longer varies."""
    n = len(vals)
    i = max(0, min(start, n - 2))
    while 0 <= i < n - 1:
        if abs(vals[i + 1] - vals[i]) <= tol:
            return vals[i]
        i += direction
    return None


def _plateau_or_window(vals, start, direction, label):
    """Plateau at the tightest tolerance that works; window average if none."""
    for tol in PLATEAU_TOLS:
        v = _find_plateau(vals, start, direction, tol)
        if v is not None:
            if tol != PLATEAU_TOLS[0]:
                log(f"WARNING: {label} plateau only found at tol {tol:g} "
                    f"(usual {PLATEAU_TOLS[0]:g}); profile is rough this frame.\n")
            return v
    w = PLATEAU_FALLBACK_WIDTH
    lo = max(min(start, len(vals) - 1) - w, 0)
    hi = min(max(start, 0) + w + 1, len(vals))
    if hi - lo < 3:
        return None
    log(f"WARNING: no {label} plateau at any tolerance; using the mean over "
        f"grid {lo}..{hi - 1} instead. Check {label}_grid and the sigmoid "
        f"transition width.\n")
    return sum(vals[lo:hi]) / (hi - lo)

def read_pot_diff(file_path):
    """Return phi_cathode - phi_anode in HARTREE, or None on failure."""
    try:
        vals = _read_avg(file_path)
        if len(vals) < 5:
            raise ValueError(f"{file_path} has only {len(vals)} rows.")
        pot_a = _plateau_or_window(vals, anode_grid, -1, "anode")
        pot_c = _plateau_or_window(vals, cathode_grid, +1, "cathode")
        if pot_a is None or pot_c is None:
            raise ValueError("Potential values not found within the specified grids.")
        return pot_c - pot_a
    except Exception as e:
        log(f"Error in read_pot_diff: {e}\n")
        return None


# ----------------------------------------------------------------------
# writing the bias into a cube
# ----------------------------------------------------------------------
bias_cube = "bias_step.cube"   # step profile alone, built once per run
_bias_cube_ready = False

def build_bias_cube(amp_V, out="pot.cube"):
    """Writes `out` = MDpot + step profile of amplitude amp_V (volts).

    The bias is constant and the grid never changes, so the step profile is
    built only once per run (on the first call, since cube_step needs
    MDpot.cube as the grid template) and reused every cycle after that."""
    global _bias_cube_ready
    if not _bias_cube_ready:
        subprocess.run([cube_sig, "MDpot.cube",
                        f"{amp_V / Ha2eV:.10f}",
                        str(z_step_pos)],
                       check=True)
        subprocess.run(["mv", "z_sig.cube", bias_cube], check=True)
        check_cube(bias_cube, "MDpot.cube")
        subprocess.run([cube_avg, bias_cube, "3"], check=True)
        subprocess.run(["mv", "cube.z.avg", "summary/avg/z_sig_au.avg"], check=True)
        log(f"Built {bias_cube} once for U = {amp_V:+.5f} V\n")
        _bias_cube_ready = True
    subprocess.run([cube_add, bias_cube, "MDpot.cube"], check=True)
    subprocess.run(["mv", "add.cube", out], check=True)
    check_cube(out, "MDpot.cube")


# ----------------------------------------------------------------------
# --- new-run marker ---------------------------------------------------
# QMMLoutput.txt stays append-only across restarts ON PURPOSE: qmml-cycles-c
# ties every value to the preceding "Starting cycle N" banner and lets a
# repeated cycle overwrite the earlier attempt, so every run has to live in
# this one file -- rotating it would hide all earlier cycles from the CSV.
# So mark where each run begins instead of splitting the file.  The text
# below is deliberately inert to that parser: it contains none of the six
# strings it keys on ("Starting cycle", "Running ", "del_phi", "cell:",
# "[pstat/", "post-SCF diagnostic:").
prog = load_progress()
resume_cycle, resume_stage = prog["cycle"], prog["stage"]
ran_any = False

import datetime as _dt
log("\n" + "=" * 72 + "\n"
    + f"== new run  {_dt.datetime.now():%Y-%m-%d %H:%M:%S}"
      f"  job {os.environ.get('SLURM_JOB_ID', '-')}"
      f"  resume: cycle={resume_cycle} stage={resume_stage}\n"
    + "=" * 72 + "\n")

log(f"CONSTANT-BIAS run, U = {fixed_bias_V:+.5f} V\n")

if prog["running"] is not None:
    # The previous job died inside this stage (walltime or crash); it reruns.
    prog["kills"] += 1
    log(f"[chain] previous job was killed during {prog['running']} of cycle "
        f"{resume_cycle} ({prog['kills']} time(s)); rerunning it.\n")
    if prog["kills"] >= max_kills:
        log(f"[chain] {prog['running']} was killed {prog['kills']} times in a row; "
            f"it does not fit in one job.  Stopping the chain.\n")
        save_progress()
        sys.exit(1)
    prog["running"] = None
    save_progress()


for cycle in range(resume_cycle, max_cycles):
    if stage_due(cycle, "qm"):
        log(f"Starting cycle {cycle} of {max_cycles} \n")

    # Heat LAMMPS
    if stage_due(cycle, "heat"):
        enter_stage(cycle, "heat")
        log("Heating LAMMPS...")
        subprocess.run(["cp", lammpsInput, "lammps_heat.in"], check=True)
        subprocess.run(["sed", "-i", f"s/run STEP/run {heatstep}/", "lammps_heat.in"], check=True)
        subprocess.run(["sed", "-i", "s/fix             solvGrid elyte gridforce -1 1//", "lammps_heat.in"], check=True)
        subprocess.run(["sed", "-i", "s/fix_modify      solvGrid energy yes//", "lammps_heat.in"], check=True)
        subprocess.run(["sed", "-i", "s/grid            V_file.cube//", "lammps_heat.in"], check=True)
        subprocess.run(["sed", "-i", "s/f_solvGrid//", "lammps_heat.in"], check=True)
        subprocess.run(["sed", "-i", "s/fix NVT elyte nvt temp 300 300 100.0/fix NVT elyte nvt temp 10 300 100.0/", "lammps_heat.in"], check=True)

        run_lammps('lammps_heat.in', "heat")
        subprocess.run(["mv", "Total.lammpstrj", "summary/heat.lammpstrj"], check=True)
        subprocess.run(["mv", "committee-energy.txt", "summary/txt/committee-energy_heat.txt"], check=True)

        # Check if LAMMPS was successful
        subprocess.run(["cat", "summary/lammps_log/log_heatcycle.txt"], stdout=open("temp.output", "w"), check=True)
        if check_lammps_success("temp.output") == False:
            log("LAMMPS calculation failed. Check the output file for details.\n")
            sys.exit(1)

        log("Done.\n")
        leave_stage("heat", cycle, "qm")

    # Run CP2K
    if stage_due(cycle, "qm"):
        enter_stage(cycle, "qm")
        log("Running CP2K...")

        subprocess.run(["cp", cp2kInput, "cp2k_input.in"], check=True)
        if cycle != 0:
            subprocess.run(["sed", "-i", "s/COORD_FILE_NAME data.cp2k/COORD_FILE_NAME runlammps.xyz/", "cp2k_input.in"], check=True)
            subprocess.run(["sed", "-i", "/&DFT/a \    &EXTERNAL_POTENTIAL\\\n        READ_FROM_CUBE T\\\n    &END EXTERNAL_POTENTIAL", "cp2k_input.in"], check=True)
        subprocess.run(["sed", "-i", f"s/SURF_DIP_POS Dipc_position/SURF_DIP_POS {cp2k_dipc_pos}/", "cp2k_input.in"], check=True)
        subprocess.run(["sed", "-i", f"s/^\\( *ALPHA *\\).*/\\1{alpha_base}/", "cp2k_input.in"], check=True)
        run_cp2k('cp2k_input.in',cycle)

        # Check if CP2K was successful
        subprocess.run(["cat", f"summary/cp2k_log/log_{cycle}cycle.txt"], stdout=open("temp.output", "w"), check=True)
        if check_cp2k_success("temp.output") == False and check_cp2k_dipole("temp.output") == True:
            log("CP2K calculation failed. Check the output file for details.\n")
            sys.exit(1)

        elif check_cp2k_success("temp.output") == False and check_cp2k_dipole("temp.output") == False:
            log("CP2K calculation failed. It may be due to dipole correction. \n")

            # CP2K re-checks the vacuum region on EVERY KS build, including the
            # wild intermediate densities early in the SCF.  A frame therefore
            # aborts on a transient, not because the converged density is bad.
            # So retry with gentler mixing: moving the sigmoid or the dipole
            # plane changes the physics of the run, whereas ALPHA only changes
            # how the SCF travels to the same answer.  pot.cube is left alone.
            scf_success = False
            for n_try, alpha in enumerate(retry_alphas):
                log(f"Trying again with gentler SCF mixing: ALPHA {alpha} "
                    f"({n_try + 1} / {len(retry_alphas)})\n")

                subprocess.run(["cp", cp2kInput, "cp2k_input.in"], check=True)
                if cycle != 0:
                    subprocess.run(["sed", "-i", "s/COORD_FILE_NAME data.cp2k/COORD_FILE_NAME runlammps.xyz/", "cp2k_input.in"], check=True)
                    subprocess.run(["sed", "-i", "/&DFT/a \    &EXTERNAL_POTENTIAL\\\n        READ_FROM_CUBE T\\\n    &END EXTERNAL_POTENTIAL", "cp2k_input.in"], check=True)
                subprocess.run(["sed", "-i", f"s/SURF_DIP_POS Dipc_position/SURF_DIP_POS {cp2k_dipc_pos}/", "cp2k_input.in"], check=True)
                subprocess.run(["sed", "-i", f"s/^\\( *ALPHA *\\).*/\\1{alpha}/", "cp2k_input.in"], check=True)

                run_cp2k('cp2k_input.in', cycle)

                subprocess.run(["cat", f"summary/cp2k_log/log_{cycle}cycle.txt"], stdout=open("temp.output", "w"), check=True)
                if check_cp2k_success("temp.output") == True:
                    log(f"Succeeded with ALPHA {alpha}.\n")
                    scf_success = True
                    break

            if scf_success == False:
                log(f"All {len(retry_alphas)} mixing retries failed. Stopping.\n")
                sys.exit(1)
        log("Done...")
        leave_stage("qm", cycle, "qmpp")

    # Post-process CP2K
    if stage_due(cycle, "qmpp"):
        enter_stage(cycle, "qmpp")
        # convert V_file to Ryd units
        subprocess.run([cube_multi, cp2k_V_file, "2"], check=True) # to make Ryd units
        subprocess.run(["mv", "multiplied.cube", f"V_ryd_{cycle}.cube"], check=True)
        if os.path.exists(cp2k_e_file):   # already moved if this stage is being rerun
            subprocess.run(["mv", cp2k_e_file, f"val_{cycle}.cube"], check=True)

        subprocess.run([cube_avg, cp2k_V_file, "3"], check=True)
        subprocess.run(["mv", "cube.z.avg", f"summary/avg/qmV_{cycle}_au.avg"], check=True)
        subprocess.run([cube_avg, f"val_{cycle}.cube", "3"], check=True)
        subprocess.run(["mv", "cube.z.avg", f"summary/avg/val_{cycle}.avg"], check=True)

        # Diagnostic only: the instantaneous drop right after the SCF.  The
        # controller does NOT act here -- it acts once per cycle, after LAMMPS.
        if cycle > 0:
            subprocess.run([cube_sub, "MDpot.cube", cp2k_V_file], check=True) # since cp2k potential is inverse physical convention
            subprocess.run([cube_avg, "subtracted.cube", "3"], check=True)
            del_phi = read_pot_diff('cube.z.avg')
        else:
            subprocess.run([cube_avg, cp2k_V_file, "3"], check=True)
            del_phi = read_pot_diff('cube.z.avg') # since cp2k potential is inverse physical convention
            if del_phi is not None:
                del_phi = -1.0 * del_phi

        if del_phi is None:
            log("Potential values not found.\n")
            sys.exit()
        log(f"post-SCF diagnostic: del_phi {del_phi * Ha2eV:+.5f} V\n")
        leave_stage("qmpp", cycle, "ml")

    # Run LAMMPS
    if stage_due(cycle, "ml"):
        enter_stage(cycle, "ml")
        log("Running LAMMPS...")

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
            log("LAMMPS calculation failed. Check the output file for details.")
            sys.exit(1)

        log("Done...")
        leave_stage("ml", cycle, "mlpp")

    # Post-process LAMMPS
    if stage_due(cycle, "mlpp"):
        enter_stage(cycle, "mlpp")

        # Potential process
        subprocess.run([dipc2, "MDrho.cube", dipc_dir, dipc_grid], check=True) # to follow the physical convention
        check_cube("dipc.cube", "MDrho.cube")
        subprocess.run([chg2pot, "dipc.cube", "3"], check=True) # generate pot.cube in Ha units
        subprocess.run(["mv", "pot.cube", "MDpot.cube"], check=True)
        check_cube("MDpot.cube", "MDrho.cube")
        if debugging:
            subprocess.run(["cp", "MDpot.cube", f"summary/cube/MDpot{cycle}.cube"], check=True)
        subprocess.run([cube_avg, "MDpot.cube", "3"], check=True)
        subprocess.run(["mv", "cube.z.avg", f"summary/avg/elyt_V_{cycle}_au.avg"], check=True)

        if cycle % savecube != 0:
            subprocess.run(["rm", "-f", f"V_ryd_{cycle}.cube"], check=True)
            subprocess.run(["rm", "-f", f"val_{cycle}.cube"], check=True)
        else:
            subprocess.run(["mv", f"V_ryd_{cycle}.cube", f"summary/cube/V_ryd_{cycle}.cube"], check=True)
            subprocess.run(["mv", f"val_{cycle}.cube", f"summary/cube/val_{cycle}.cube"], check=True)
        if cycle % saverestart == 0:
            subprocess.run(["cp", "qmml.restart", f"summary/restart/qmml{cycle}cycle.restart"], check=True)

        # ---- APPLIED BIAS (constant) --------------------------------------
        build_bias_cube(fixed_bias_V, out="pot.cube")

        log(f"Cycle {cycle} completed.\n")
        leave_stage("mlpp", cycle + 1, "qm")

log("Simulation completed.\n")
