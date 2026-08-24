"""Contribute detector entropy to the Linux kernel random pool.

The kernel exposes ``RNDADDENTROPY`` on ``/dev/random``: it mixes a buffer into
the input pool *and* credits a caller-supplied number of entropy bits, so the
contribution shows up in ``/proc/sys/kernel/random/entropy_avail`` and benefits
every consumer of ``getrandom(2)``. This is the same interface ``rngd`` and
``haveged`` use.

Two things matter for doing this honestly:

* **The credit must be the assessed min-entropy, not the buffer length.**
  Over-crediting the kernel is worse than not contributing at all, because the
  kernel will hand out ``getrandom`` bytes believing it has entropy it does
  not. We feed 32-byte full-entropy blocks and credit 256 bits each -- blocks
  that already cost 320 bits of banked detector entropy to produce.

* **It needs ``CAP_SYS_ADMIN``.** Without it the ioctl fails with ``EPERM``.
  The module falls back to plain ``write()`` on ``/dev/random``, which still
  stirs the data into the pool but credits zero bits. That fallback is
  genuinely useful (it can only improve the pool's state) and is honest about
  contributing no counted entropy.

On kernels 5.6 and newer the pool is credited once at boot and stays
"initialised" forever, so ``entropy_avail`` is not the scarcity signal it was
on older kernels. The daemon therefore defaults to a steady low-rate trickle
rather than a watermark chase, with ``--watermark`` available for older
kernels or for operators who want it.
"""

from __future__ import annotations

import array
import logging
import os
import struct
import sys
import time
from typing import Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows and other non-POSIX hosts
    # Importing this module must not explode off-Linux: the CLI imports it in
    # order to *report* that kernel-pool contribution is unavailable, and a
    # bare ModuleNotFoundError would replace that explanation with a traceback.
    fcntl = None  # type: ignore[assignment]

_log = logging.getLogger(__name__)

#: ``_IOW('R', 0x03, int[2])`` -- add entropy and credit it.
RNDADDENTROPY = 0x40085203
#: ``_IOR('R', 0x00, int)`` -- read the current entropy estimate.
RNDGETENTCNT = 0x80045200

RANDOM_DEVICE = '/dev/random'
ENTROPY_AVAIL = '/proc/sys/kernel/random/entropy_avail'
POOL_SIZE = '/proc/sys/kernel/random/poolsize'


class NotLinux(RuntimeError):
    pass


def _require_linux() -> None:
    if fcntl is None or not sys.platform.startswith('linux'):
        raise NotLinux(
            'the kernel pool feeder is Linux-only; on Windows use '
            '"radiarandom serve" (see radiarandom.pools.windows for why)'
        )


def entropy_avail() -> Optional[int]:
    """Current kernel entropy estimate in bits, or None if unreadable."""
    try:
        with open(ENTROPY_AVAIL, 'r', encoding='ascii') as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


def pool_size() -> Optional[int]:
    try:
        with open(POOL_SIZE, 'r', encoding='ascii') as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


class KernelPool:
    """Handle on ``/dev/random`` for entropy contribution."""

    def __init__(self, device: str = RANDOM_DEVICE) -> None:
        _require_linux()
        self.device = device
        self._fd: Optional[int] = None
        self.can_credit = False
        self.bits_credited = 0
        self.bytes_written = 0

    def open(self) -> KernelPool:
        self._fd = os.open(self.device, os.O_WRONLY)
        self.can_credit = self._probe_credit()
        if not self.can_credit:
            _log.warning(
                'RNDADDENTROPY unavailable (need CAP_SYS_ADMIN); falling back '
                'to uncredited writes to %s', self.device,
            )
        return self

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> KernelPool:
        return self.open()

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _probe_credit(self) -> bool:
        """Try a zero-bit contribution to see whether the ioctl is permitted.

        Crediting zero bits changes nothing about the kernel's estimate, so
        this is a side-effect-free capability probe apart from stirring 4 bytes
        of our own data into the pool.
        """
        try:
            self._ioctl_add(b'\x00\x00\x00\x00', 0)
            return True
        except PermissionError:
            return False
        except OSError as exc:
            _log.warning('RNDADDENTROPY probe failed: %s', exc)
            return False

    def _ioctl_add(self, data: bytes, entropy_bits: int) -> None:
        if self._fd is None:
            raise RuntimeError('pool is not open')
        # struct rand_pool_info { int entropy_count; int buf_size; __u32 buf[]; }
        padded = data + b'\x00' * (-len(data) % 4)
        payload = array.array('B', struct.pack('ii', entropy_bits, len(padded)) + padded)
        fcntl.ioctl(self._fd, RNDADDENTROPY, payload, False)

    def add(self, data: bytes, entropy_bits: int) -> int:
        """Contribute ``data``, crediting ``entropy_bits``.

        Returns the number of bits actually credited (zero on the fallback
        path). Never credits more bits than ``data`` contains.
        """
        if self._fd is None:
            raise RuntimeError('pool is not open')
        credit = max(0, min(entropy_bits, len(data) * 8))
        if self.can_credit:
            self._ioctl_add(data, credit)
            self.bits_credited += credit
        else:
            os.write(self._fd, data)
            credit = 0
        self.bytes_written += len(data)
        return credit

    def stats(self) -> dict:
        return {
            'device': self.device,
            'can_credit': self.can_credit,
            'bits_credited': self.bits_credited,
            'bytes_written': self.bytes_written,
            'entropy_avail': entropy_avail(),
            'pool_size': pool_size(),
        }


def feed(
    generator,
    watermark: Optional[int] = None,
    interval: float = 1.0,
    max_bits_per_second: Optional[float] = None,
    device: str = RANDOM_DEVICE,
    on_status=None,
    status_interval: float = 60.0,
) -> None:
    """Run the kernel feeder until the generator stops or a health test fails.

    Args:
        generator: a started :class:`radiarandom.Generator` whose start-up test
            has already passed.
        watermark: if set, only contribute while ``entropy_avail`` is below
            this many bits. Useful on pre-5.6 kernels. If None, trickle
            continuously.
        interval: seconds between checks when idling.
        max_bits_per_second: optional cap on the credited rate. The detector is
            the real limit, so this is mostly a safety valve.
        device: the random device to feed.
        on_status: optional callable receiving a stats dict periodically.
    """
    from ..conditioner import BLOCK_BITS

    _require_linux()
    started = time.perf_counter()
    last_status = started
    credited_total = 0

    with KernelPool(device) as pool:
        _log.info(
            'feeding %s (credit=%s, entropy_avail=%s/%s)',
            device, pool.can_credit, entropy_avail(), pool_size(),
        )
        while not generator.stopped:
            if watermark is not None:
                current = entropy_avail()
                if current is not None and current >= watermark:
                    time.sleep(interval)
                    continue

            block = generator.physical_block()
            credited = pool.add(block, BLOCK_BITS)
            credited_total += credited

            if max_bits_per_second:
                elapsed = time.perf_counter() - started
                allowed = max_bits_per_second * elapsed
                if credited_total > allowed:
                    time.sleep((credited_total - allowed) / max_bits_per_second)

            now = time.perf_counter()
            if on_status is not None and now - last_status >= status_interval:
                last_status = now
                stats = pool.stats()
                stats['generator'] = generator.stats()
                stats['uptime_s'] = now - started
                on_status(stats)
