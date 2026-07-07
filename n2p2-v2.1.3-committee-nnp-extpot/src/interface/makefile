#!/bin/make -f

###############################################################################
# PROJECT PATHS
###############################################################################
PROJECT_DIR=../..
PROJECT_BIN=$(PROJECT_DIR)/bin
PROJECT_INCLUDE=$(PROJECT_DIR)/include
PROJECT_LIB=$(PROJECT_DIR)/lib


###############################################################################
# GENERAL SETTINGS
###############################################################################
# Default compiler, may be overridden by master makefile or command line.
COMP=gnu

# Default build mode, may be overridden by master makefile or command line.
# Possible modes are "static" and "test".
MODE=static

# LAMMPS version to download and build.
# NOTE: switched from stable_29Oct2020 to stable_29Sep2021 because of a
#       gather_atoms bug in 29Oct2020
#       (https://matsci.org/t/lammps-users-typeerror-on-gather-atoms-in-python-examples/39076)
LAMMPS_VERSION=stable_29Sep2021

# DFT-CES patch (Seung-Jae Shin, https://github.com/SeungJay/DFT-CES)
DFTCES_PATCH=stable_29Sep2021.patch
DFTCES_PATCH_URL=https://raw.githubusercontent.com/SeungJay/DFT-CES/main/DFT-CES_MD/stable_29Sep2021/stable_29Sep2021.patch

# Include global (project-wide) settings.
include $(PROJECT_DIR)/src/makefile.$(COMP)


###############################################################################
# RULES
###############################################################################
.PHONY: all lammps-nnp clean-lammps-nnp clean-lammps-nnp-keep-file

all: lammps-nnp

clean: clean-lammps-nnp

##############
# lammps-nnp #
##############
LAMMPS_DEBUG := $(filter-out -pedantic-errors,$(PROJECT_DEBUG))
lammps-nnp:
	# --- 1) Download pristine LAMMPS source from GitHub ------------------
	if [ ! -e $(LAMMPS_VERSION).tar.gz ]; then \
		command -v wget >/dev/null 2>&1 \
		&& wget https://github.com/lammps/lammps/archive/$(LAMMPS_VERSION).tar.gz \
		|| curl -L -o $(LAMMPS_VERSION).tar.gz https://github.com/lammps/lammps/archive/$(LAMMPS_VERSION).tar.gz; \
	fi
	# --- 2) Extract and rename to lammps-nnp (skip if it already exists) -
	if [ ! -d lammps-nnp ]; then \
		tar -xzf $(LAMMPS_VERSION).tar.gz && mv lammps-$(LAMMPS_VERSION) lammps-nnp; \
	fi
	# --- 3) Download the DFT-CES patch ----------------------------------
	if [ ! -e $(DFTCES_PATCH) ]; then \
		command -v wget >/dev/null 2>&1 \
		&& wget -O $(DFTCES_PATCH) $(DFTCES_PATCH_URL) \
		|| curl -L -o $(DFTCES_PATCH) $(DFTCES_PATCH_URL); \
	fi
	# --- 4) Apply the DFT-CES patch (skip if already applied) -----------
	if [ ! -e lammps-nnp/src/grid.cpp ]; then \
		cd lammps-nnp && patch -p1 < ../$(DFTCES_PATCH); \
	fi
	# --- 5) Link n2p2 library + install committee NNP pair style --------
	ln -sf $(PROJECT_DIR)/../../ lammps-nnp/lib/nnp
	if [ ! -d lammps-nnp/src/USER-NNP ]; then \
		cp -r ./LAMMPS/src/USER-NNP lammps-nnp/src/; \
	fi
	# --- 6) Inject MPI compiler settings into Makefile.mpi --------------
	@sed -i.bak "s/^CC .*$$/CC      =       $(PROJECT_MPICC)/" lammps-nnp/src/MAKE/Makefile.mpi
	@sed -i.bak "s/^CCFLAGS .*$$/CCFLAGS =       $(PROJECT_CFLAGS) $(PROJECT_CFLAGS_MPI)/" lammps-nnp/src/MAKE/Makefile.mpi
	@sed -i.bak "s/^LINK .*$$/LINK =          $(PROJECT_MPICC)/" lammps-nnp/src/MAKE/Makefile.mpi
	@sed -i.bak "s/^LINKFLAGS .*$$/LINKFLAGS =     $(PROJECT_CFLAGS) $(PROJECT_CFLAGS_MPI)/" lammps-nnp/src/MAKE/Makefile.mpi
	if [ "$(MODE)" = "test" ]; then \
		sed -i.bak "/^CCFLAGS =/   s/$$/ $(LAMMPS_DEBUG) $(PROJECT_TEST)/" lammps-nnp/src/MAKE/Makefile.mpi; \
		sed -i.bak "/^LINKFLAGS =/ s/$$/ $(LAMMPS_DEBUG) $(PROJECT_TEST)/" lammps-nnp/src/MAKE/Makefile.mpi; \
	fi
	@rm lammps-nnp/src/MAKE/Makefile.mpi.bak
	# --- 7) Build LAMMPS (committee NNP + molecule + DFT-CES) -----------
	cd lammps-nnp/src/ && $(MAKE) yes-user-nnp && $(MAKE) yes-molecule && $(MAKE) mpi
	cp lammps-nnp/src/lmp_mpi $(PROJECT_BIN)/
	# --- 8) Report what was done ---------------------------------------
	@echo ""
	@echo "==================================================================="
	@echo " lammps-nnp is ready."
	@echo "==================================================================="
	@echo " What this makefile just did:"
	@echo "  1. Downloaded LAMMPS $(LAMMPS_VERSION) from github.com/lammps/lammps"
	@echo "  2. Extracted it as ./lammps-nnp"
	@echo "  3. Downloaded the DFT-CES patch ($(DFTCES_PATCH))"
	@echo "     from $(DFTCES_PATCH_URL)"
	@echo "  4. Applied the patch: added grid.cpp/.h and fix_gridforce.cpp/.h,"
	@echo "     modified domain.*, input.*, run.cpp"
	@echo "     -> enables the 'grid' command and 'fix gridforce' for DFT-CES"
	@echo "  5. Linked n2p2 (lammps-nnp/lib/nnp) and installed the committee"
	@echo "     NNP pair style (USER-NNP -> pair_nnp)"
	@echo "  6. Set the MPI compiler to $(PROJECT_MPICC) in Makefile.mpi"
	@echo "  7. Built with packages user-nnp + molecule -> lmp_mpi"
	@echo "     Executable copied to $(PROJECT_BIN)/lmp_mpi"
	@echo "==================================================================="
	@echo ""

clean-lammps-nnp:
	$(RM) -r lammps-nnp
	$(RM) $(LAMMPS_VERSION).tar.gz
	$(RM) $(DFTCES_PATCH)
	$(RM) $(PROJECT_BIN)/lmp_mpi

clean-lammps-nnp-keep-file:
	$(RM) -r lammps-nnp
	$(RM) $(PROJECT_BIN)/lmp_mpi
