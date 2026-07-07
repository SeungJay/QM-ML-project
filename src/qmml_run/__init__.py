"""qmml_run — QM/ML electrochemistry driver (LAMMPS + CP2K cycle loop).

Part of the QMML distribution. Provides the :class:`Config` dataclass and the
:func:`run` driver, plus the in-process cube :mod:`~qmml_run.pipeline`.

    from qmml_run import Config, run
    run(Config(steepness=0.4, ...))
"""

from __future__ import annotations

from .driver import Config, run
from . import pipeline

__all__ = ["Config", "run", "pipeline"]
