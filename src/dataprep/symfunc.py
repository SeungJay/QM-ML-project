"""Atom-centred symmetry function (ACSF) generation for n2p2 ``input.nn``.

A dependency-free port of the generation + formatting logic from the ``aml``
package (``aml/acsf.py``). Symmetry-function *parameters* are generated from a
systematic scheme (shifted radial + centred angular) rather than hardcoded, and
the same set is assigned to every element pair/triple, so the resulting
``symfunction_short`` lines adapt to whatever ``elements`` you pass.

The default scheme (:func:`generate_radial_angular_default`) reproduces the
standard 10-radial / 4-angular set used in this work:
  radial  : shifted, n=10, r_0=0.14 Angstrom, r_max=r_c-0.14 Angstrom, r_c=12 Bohr
  angular : centred, r_0=2.8 Angstrom, lambda=+/-1, zeta={1,4}, r_c=12 Bohr
Lengths are emitted in Bohr (RuNNer/n2p2 atomic-unit convention).

References:
  Gastegger et al., J. Chem. Phys. 148, 241709 (2018)  [wACSF]
  Imbalzano et al., J. Chem. Phys. 148, 241730 (2018)  [auto SF selection]
"""

from __future__ import annotations

import itertools
from collections import namedtuple
from typing import Sequence

# 1 Angstrom in Bohr (atomic units) — n2p2/RuNNer .data & input.nn use Bohr.
ANGSTROM = 1.0 / 0.52917721067

# parameters of a radial ACSF (G2) and an angular ACSF (G3)
RadialSF = namedtuple("RadialSF", ["eta", "mu", "r_c"])
AngularSF = namedtuple("AngularSF", ["lam", "zeta", "eta", "r_c", "mu"])


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
def generate_radial_shifted(n: int, r_0: float, r_max: float,
                            r_c: float) -> list:
    """A systematic set of ``n`` shifted radial ACSFs (fixed eta, spaced mu)."""
    dr = (r_max - r_0) / (n - 1)
    eta = 1.0 / (2.0 * dr ** 2)
    return [RadialSF(eta=eta, mu=r_0 + i * dr, r_c=r_c) for i in range(n)]


def generate_angular_centered(n: int, r_0: float, r_max: float,
                              zeta: float, r_c: float) -> list:
    """A systematic set of centred angular ACSFs (mu=0), both lambda signs."""
    if n % 2 != 0:
        raise ValueError("`n` must be even.")
    dr = (r_max - r_0) / (n - 1)
    angulars = []
    for i in range(n // 2):
        ri = r_0 + i * dr
        eta = 1.0 / (3.0 * ri ** 2)
        for lam in (-1.0, 1.0):
            angulars.append(AngularSF(lam=lam, zeta=zeta, mu=0.0, eta=eta, r_c=r_c))
    return angulars


def generate_radial_angular_default():
    """The "default" 10 radial + 4 angular ACSF parameter sets used in this work."""
    r_c = 12.0
    r_0_radial = 0.14 * ANGSTROM
    r_max_radial = r_c - 0.14 * ANGSTROM
    r_0_angular = 2.8 * ANGSTROM
    radials = generate_radial_shifted(n=10, r_0=r_0_radial,
                                      r_max=r_max_radial, r_c=r_c)
    angulars = (generate_angular_centered(2, r_0_angular, r_max_radial, 1.0, r_c)
                + generate_angular_centered(2, r_0_angular, r_max_radial, 4.0, r_c))
    return radials, angulars


# --------------------------------------------------------------------------- #
# formatting (RuNNer/n2p2 input.nn)
# --------------------------------------------------------------------------- #
def _fmt_radial_block(radials: Sequence, e1: str, e2: str) -> str:
    return "\n".join(
        f"symfunction_short {e1} 2 {e2} {r.eta:f} {r.mu:f} {r.r_c:f}"
        for r in radials)


def _fmt_angular_block(angulars: Sequence, e1: str, e2: str, e3: str) -> str:
    return "\n".join(
        f"symfunction_short {e1} 3 {e2} {e3} {a.eta:f} {a.lam: f} {a.zeta:f} {a.r_c:f}"
        for a in angulars)


def format_symfunctions(radials: Sequence, angulars: Sequence,
                        elements: Sequence[str]) -> str:
    """Format radial + angular ACSFs for every element pair/triple.

    The same ``radials`` set is used for each (central, neighbour) pair and the
    same ``angulars`` set for each (central, neighbour1, neighbour2) triple, with
    only neighbour pairs where index(n2) >= index(n1) (n2p2 convention).
    """
    lines = []
    for e1 in elements:
        lines += ["#", f"# Radial symmetry functions for {e1}", "#",
                  "# <element-central> 2 <element-neighbor> <eta> <rshift> <rcutoff>", ""]
        for e2 in elements:
            lines += [f"# {e1} - {e2}", _fmt_radial_block(radials, e1, e2), ""]
        lines.append("")
    for e1 in elements:
        lines += ["#", f"# Angular symmetry functions for {e1}", "#",
                  "# <element-central> 3 <element-neighbor1> <element-neighbor2> "
                  "<eta> <lambda> <zeta> <rcutoff> <<rshift>>", ""]
        for i2, e2 in enumerate(elements):
            for i3, e3 in enumerate(elements):
                if i3 < i2:
                    continue
                lines += [f"# {e1} - {e2}-{e3}",
                          _fmt_angular_block(angulars, e1, e2, e3), ""]
        lines.append("")
    return "\n".join(lines)


def make_symfunctions(elements: Sequence[str]) -> str:
    """Convenience: the default ACSF set formatted for ``elements``.

    This is the string that fills the ``{symmetry_functions}`` placeholder in an
    ``input.nn`` template.
    """
    radials, angulars = generate_radial_angular_default()
    return format_symfunctions(radials, angulars, elements)
