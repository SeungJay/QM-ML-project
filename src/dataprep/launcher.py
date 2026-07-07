"""Process launcher for running external commands (e.g. CP2K).

A small dependency-free stand-in for ``aml.launcher.ProcessLauncher``. It builds
an MPI-launch prefix (via a ``mode`` callable or a preset) and runs commands in a
working directory. Multiple structures can be run concurrently with ``n_slots``
worker threads.

Example
-------
    def srun(i_slot, n_core_task, size_node):
        return f"srun -n {n_core_task} "

    launcher = ProcessLauncher(mode=srun, n_slots=1, n_core_task=128)
    launcher.run([["cp2k.popt", "-i", "step-0.inp", "-o", "step-0.log"]], directory)
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union
import shlex
import subprocess


class LauncherError(RuntimeError):
    pass


class ProcessLauncher:
    def __init__(self, mode: Union[str, Callable] = "plain",
                 n_slots: int = 1, n_core_task: int = 1,
                 size_node: Optional[int] = None):
        """
        mode : one of the built-in names ``"srun"``, ``"mpirun"``, ``"plain"``
               (no prefix), or a callable ``(i_slot, n_core_task, size_node) ->
               prefix string`` for full control. The built-ins expand to
               ``srun -n <n_core_task>`` / ``mpirun -np <n_core_task>``.
        n_slots : number of concurrent worker slots.
        n_core_task : cores/processes per task (used by the built-in prefixes).
        """
        self.mode = mode
        self.n_slots = int(n_slots)
        self.n_core_task = int(n_core_task)
        self.size_node = size_node

    # ------------------------------------------------------------------
    def prefix(self, i_slot: int = 0) -> str:
        if callable(self.mode):
            return self.mode(i_slot, self.n_core_task, self.size_node)
        m = (self.mode or "plain").lower()
        if m in ("plain", ""):
            return ""
        if m == "srun":
            return f"srun -n {self.n_core_task} "
        if m == "mpirun":
            return f"mpirun -np {self.n_core_task} "
        raise LauncherError(
            f"unknown launcher mode: {self.mode!r} "
            f"(use 'srun', 'mpirun', 'plain', or a callable)")

    def _full_command(self, cmd, i_slot: int = 0):
        pre = shlex.split(self.prefix(i_slot))
        if isinstance(cmd, str):
            cmd = shlex.split(cmd)
        return pre + list(cmd)

    # ------------------------------------------------------------------
    def run(self, commands: Sequence, directory: str, i_slot: int = 0,
            check: bool = True):
        """Run ``commands`` (each a list or string) sequentially in ``directory``.

        Returns the list of ``subprocess.CompletedProcess``. Raises
        :class:`LauncherError` on non-zero exit when ``check`` is True.
        """
        results = []
        for cmd in commands:
            full = self._full_command(cmd, i_slot)
            r = subprocess.run(full, cwd=str(directory),
                               capture_output=True, text=True)
            if check and r.returncode != 0:
                raise LauncherError(
                    f"command failed (exit {r.returncode}): {' '.join(full)}\n"
                    f"  stdout: {r.stdout[-2000:]}\n  stderr: {r.stderr[-2000:]}"
                )
            results.append(r)
        return results
