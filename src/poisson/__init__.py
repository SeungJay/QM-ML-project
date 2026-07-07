"""poisson — Poisson solver (charge density -> electrostatic potential).

Part of the QMML distribution. Depends on :mod:`cubetools` for the ``Cube``
container. Import directly::

    from poisson import chg2pot
    from cubetools import Cube

    pot = chg2pot(Cube.from_file("chgden.cube"))
"""

from __future__ import annotations

from .poisson import chg2pot

__all__ = ["chg2pot"]
