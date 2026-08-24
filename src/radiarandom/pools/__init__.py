"""Operating-system integrations that publish entropy to other software.

* :mod:`radiarandom.pools.linux` -- credits the kernel entropy pool through
  ``RNDADDENTROPY`` on ``/dev/random``. This is the real thing: entropy the
  detector produced becomes entropy every process on the machine benefits from,
  via ``getrandom(2)``.

* :mod:`radiarandom.pools.windows` -- Windows has no supported equivalent, so
  this module provides an opt-in named-pipe service instead. See its docstring
  for why, with citations.

* :mod:`radiarandom.pools.service` -- transport-agnostic serving (named pipe,
  FIFO, TCP) shared by both platforms.
"""

from __future__ import annotations

__all__ = ['linux', 'windows', 'service']
