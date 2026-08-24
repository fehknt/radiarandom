"""Noise source: a RadiaCode 103 read as a channel-resolved event counter.

Physics
-------
Every count the detector reports is one gamma/x-ray photon depositing energy in
the CsI(Tl) crystal. Both *when* a decay happens and *how much* energy the
photon carries are quantum-mechanically indeterminate, so a count stream is a
genuine physical entropy source rather than a deterministic one.

Device interface
----------------
The RadiaCode exposes ``VS_SPEC_ACCUM``, a **monotonic, cumulative** 1024-bin
pulse-height histogram. Differencing two reads yields, exactly and without loss
or double counting, the multiset of pulse-height channels of every photon
detected in between. That is the highest-information observable the USB
protocol offers -- strictly more than the ``count_rate`` field in the data
buffer, which is a smoothed fixed-point average and must never be used as an
entropy source.

Measured on RC-103-013128 (firmware 4.14):

* the device refreshes the accumulated spectrum at **2 Hz**, so counts arrive
  in ~500 ms batches and arrival-time resolution is capped at 500 ms;
* a USB read of the accumulated spectrum costs ~3.6 ms, so polling faster than
  the refresh rate is cheap but buys no extra information;
* indoor background is ~4.4 counts/s.

We therefore poll a few times per refresh interval to pick up each batch
promptly, and treat each *batch* as the unit over which ordering information is
lost (see :mod:`radiarandom.entropy`).
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Iterator, Optional, Sequence

from . import _usbshim

_log = logging.getLogger(__name__)

#: Number of pulse-height channels reported by the RC-10x family.
N_CHANNELS = 1024

#: Interval at which the device refreshes VS_SPEC_ACCUM, measured empirically.
DEVICE_REFRESH_S = 0.5

#: Default host poll interval. Comfortably faster than DEVICE_REFRESH_S so no
#: batch waits long, but slow enough to leave the USB link almost idle.
DEFAULT_POLL_INTERVAL_S = 0.2


class SourceError(RuntimeError):
    """The noise source is not usable."""


class DeviceNotFound(SourceError):
    """No RadiaCode was found on USB."""


@dataclasses.dataclass(frozen=True)
class Batch:
    """One observation: every photon seen since the previous observation.

    Attributes:
        seq: Monotonic batch counter, starting at 0 for the first *usable*
            batch (the baseline read is not emitted).
        host_time: ``time.time()`` when the read completed, for logging only.
        host_monotonic: ``time.perf_counter()`` when the read completed.
        device_seconds: The device's own accumulation duration in seconds. An
            independent clock, used to detect a frozen or reset device.
        channels: Pulse-height channel of each new photon, ascending. Repeats
            appear once per photon, so ``len(channels) == count``.
        count: Number of photons in this batch.
        cumulative_total: Device lifetime total after this batch.
    """

    seq: int
    host_time: float
    host_monotonic: float
    device_seconds: int
    channels: tuple[int, ...]
    count: int
    cumulative_total: int

    @property
    def is_empty(self) -> bool:
        return self.count == 0


class RadiaCodeSource:
    """Turns a RadiaCode into a stream of :class:`Batch` observations.

    The source owns the USB connection and is *not* thread-safe; drive it from
    a single thread and fan out downstream if needed.
    """

    def __init__(
        self,
        serial_number: Optional[str] = None,
        poll_interval: float = DEFAULT_POLL_INTERVAL_S,
        bluetooth_mac: Optional[str] = None,
    ) -> None:
        self.serial_number = serial_number
        self.bluetooth_mac = bluetooth_mac
        self.poll_interval = poll_interval

        self._rc = None
        self._prev: Optional[list[int]] = None
        self._prev_total = 0
        self._prev_device_seconds = 0
        self._seq = 0
        self._resets = 0
        self._opened_monotonic = 0.0

    # ------------------------------------------------------------------ open

    def open(self) -> RadiaCodeSource:
        _usbshim.install()

        from radiacode import RadiaCode
        from radiacode.transports.usb import DeviceNotFound as _RCNotFound

        try:
            if self.bluetooth_mac:
                self._rc = RadiaCode(bluetooth_mac=self.bluetooth_mac)
            else:
                self._rc = RadiaCode(serial_number=self.serial_number)
        except _RCNotFound as exc:
            raise DeviceNotFound(
                'No RadiaCode found on USB. Check the cable, and on Linux make '
                'sure the udev rule from packaging/linux/ is installed.'
            ) from exc

        spectrum = self._rc.spectrum_accum()
        if len(spectrum.counts) != N_CHANNELS:
            raise SourceError(
                f'expected {N_CHANNELS} spectrum channels, device reported '
                f'{len(spectrum.counts)}'
            )
        self._prev = list(spectrum.counts)
        self._prev_total = sum(self._prev)
        self._prev_device_seconds = int(spectrum.duration.total_seconds())
        self._opened_monotonic = time.perf_counter()
        _log.info(
            'opened %s, lifetime counts=%d, accumulation=%ds',
            self.serial(),
            self._prev_total,
            self._prev_device_seconds,
        )
        return self

    def close(self) -> None:
        if self._rc is not None:
            self._rc.close()
            self._rc = None

    def __enter__(self) -> RadiaCodeSource:
        return self.open()

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ------------------------------------------------------------- metadata

    def _require(self):
        if self._rc is None:
            raise SourceError('source is not open')
        return self._rc

    def serial(self) -> str:
        return self._require().serial_number()

    def firmware(self) -> str:
        (_boot, (major, minor, date)) = self._require().fw_version()
        return f'{major}.{minor} ({date})'

    def energy_calibration(self) -> tuple[float, float, float]:
        spectrum = self._require().spectrum_accum()
        return (spectrum.a0, spectrum.a1, spectrum.a2)

    def reference_spectrum(self) -> list[int]:
        """The device lifetime accumulated spectrum.

        Useful as a high-statistics reference distribution for the spectral
        shape health test and for offline min-entropy estimation.
        """
        return list(self._require().spectrum_accum().counts)

    @property
    def resets_seen(self) -> int:
        return self._resets

    # ------------------------------------------------------------------ read

    def read_batch(self) -> Batch:
        """Read the device once and return everything new since the last read.

        An empty batch (``count == 0``) is normal and is returned as such: at
        5 Hz polling against a 2 Hz device refresh, most reads are empty.
        """
        rc = self._require()
        spectrum = rc.spectrum_accum()
        now_mono = time.perf_counter()
        now_wall = time.time()

        counts: Sequence[int] = spectrum.counts
        device_seconds = int(spectrum.duration.total_seconds())
        assert self._prev is not None

        if len(counts) != N_CHANNELS:
            raise SourceError(f'device returned {len(counts)} channels mid-stream')

        # A spectrum reset (user pressed reset, or the device rebooted) makes
        # the cumulative counter go backwards. Differencing across that point
        # would fabricate garbage, so re-baseline and report an empty batch.
        if device_seconds < self._prev_device_seconds or any(
            new < old for new, old in zip(counts, self._prev)
        ):
            self._resets += 1
            _log.warning('device spectrum reset detected; re-baselining')
            self._prev = list(counts)
            self._prev_total = sum(counts)
            self._prev_device_seconds = device_seconds
            batch = Batch(
                seq=self._seq,
                host_time=now_wall,
                host_monotonic=now_mono,
                device_seconds=device_seconds,
                channels=(),
                count=0,
                cumulative_total=self._prev_total,
            )
            self._seq += 1
            return batch

        channels: list[int] = []
        for channel, (new, old) in enumerate(zip(counts, self._prev)):
            delta = new - old
            if delta:
                channels.extend([channel] * delta)

        self._prev = list(counts)
        total = sum(counts)
        self._prev_total = total
        self._prev_device_seconds = device_seconds

        batch = Batch(
            seq=self._seq,
            host_time=now_wall,
            host_monotonic=now_mono,
            device_seconds=device_seconds,
            channels=tuple(channels),
            count=len(channels),
            cumulative_total=total,
        )
        self._seq += 1
        return batch

    def batches(self, poll_interval: Optional[float] = None) -> Iterator[Batch]:
        """Yield batches forever, pacing reads at ``poll_interval``.

        Empty batches are yielded too; consumers that only care about photons
        can filter on :attr:`Batch.is_empty`, but the health monitor wants to
        see the empty ones because a long run of them means a dead detector.
        """
        interval = self.poll_interval if poll_interval is None else poll_interval
        next_deadline = time.perf_counter()
        while True:
            yield self.read_batch()
            next_deadline += interval
            sleep_for = next_deadline - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # Fell behind (USB hiccup, host descheduled). Resynchronise
                # rather than spinning to catch up.
                next_deadline = time.perf_counter()

    def measure_count_rate(self, seconds: float = 20.0) -> float:
        """Measure counts per second over a window. Used by ``info``/``bench``."""
        start_total = self._prev_total
        start = time.perf_counter()
        deadline = start + seconds
        last_total = start_total
        while time.perf_counter() < deadline:
            batch = self.read_batch()
            last_total = batch.cumulative_total
            time.sleep(self.poll_interval)
        elapsed = time.perf_counter() - start
        return (last_total - start_total) / elapsed if elapsed > 0 else 0.0
