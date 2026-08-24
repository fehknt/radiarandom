"""radiarandom -- a hardware random number generator built on a RadiaCode 103.

Entropy comes from gamma photons detected by the device's CsI(Tl) scintillator:
both the timing of a nuclear decay and the energy it deposits are quantum
indeterminate. We bank only the pulse-height channel of each photon, which is
the conservative half of that (see :mod:`radiarandom.entropy`).

Quick start::

    from radiarandom import open_generator

    with open_generator() as gen:
        gen.wait_for_startup()
        seed = gen.physical_block()      # 32 bytes, full entropy, slow
        bulk = gen.read(1024 * 1024)     # DRBG output, fast
"""

from __future__ import annotations

import contextlib
from typing import Iterator, Optional

from .conditioner import BLOCK_BYTES, EntropyPool, HmacDrbg
from .device import Batch, DeviceNotFound, RadiaCodeSource, SourceError
from .entropy import Assessment, default_assessment
from .generator import Generator, GeneratorError
from .health import HealthFailure, HealthMonitor

__version__ = '1.0.0'

__all__ = [
    'Assessment',
    'Batch',
    'BLOCK_BYTES',
    'DeviceNotFound',
    'EntropyPool',
    'Generator',
    'GeneratorError',
    'HealthFailure',
    'HealthMonitor',
    'HmacDrbg',
    'RadiaCodeSource',
    'SourceError',
    'default_assessment',
    'open_generator',
    '__version__',
]


@contextlib.contextmanager
def open_generator(
    serial_number: Optional[str] = None,
    poll_interval: Optional[float] = None,
    use_reference_spectrum: bool = True,
    **generator_kwargs,
) -> Iterator[Generator]:
    """Open the device and yield a ready-to-pump :class:`Generator`.

    The caller still has to call :meth:`Generator.wait_for_startup` before
    asking for output -- it can take minutes on background radiation, and
    blocking silently inside a context manager would be unkind.
    """
    source = RadiaCodeSource(serial_number=serial_number)
    if poll_interval is not None:
        source.poll_interval = poll_interval
    source.open()
    try:
        reference = source.reference_spectrum() if use_reference_spectrum else None
        generator = Generator(source, reference_spectrum=reference, **generator_kwargs)
        try:
            yield generator
        finally:
            generator.stop()
    finally:
        source.close()
