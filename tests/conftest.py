"""Shared fixtures. Nothing here touches real hardware.

The unit tests must run in CI, on a laptop with no detector plugged in, and
before anyone trusts the thing with a key. :class:`FakeSource` stands in for a
RadiaCode by emitting batches drawn from a configurable channel distribution,
which also lets tests simulate failure modes that are hard to arrange with real
hardware -- a detector that goes silent, a spectrum that collapses onto one
channel, a device that resets mid-stream.
"""

from __future__ import annotations

import itertools
import math
import random
import time
from typing import Optional, Sequence

import pytest

from radiarandom.device import Batch, N_CHANNELS, SourceError


def background_like_spectrum(n_channels: int = N_CHANNELS) -> list[float]:
    """A rough stand-in for a CsI(Tl) background spectrum.

    Falling exponential continuum plus a broad bump, which is close enough to
    the real shape (min-entropy around 5 bits) for tests that care about the
    entropy accounting behaving sensibly.
    """
    weights = []
    for channel in range(n_channels):
        continuum = math.exp(-channel / 90.0)
        bump = 0.35 * math.exp(-((channel - 35) ** 2) / (2 * 18.0 ** 2))
        weights.append(continuum + bump + 1e-4)
    total = sum(weights)
    return [weight / total for weight in weights]


class FakeSource:
    """A RadiaCodeSource work-alike driven by a seeded RNG."""

    def __init__(
        self,
        seed: int = 12345,
        count_rate: float = 4.4,
        refresh_s: float = 0.5,
        poll_interval: float = 0.0,
        weights: Optional[Sequence[float]] = None,
        n_channels: int = N_CHANNELS,
    ) -> None:
        self.rng = random.Random(seed)
        self.count_rate = count_rate
        self.refresh_s = refresh_s
        self.poll_interval = poll_interval
        self.n_channels = n_channels
        self.weights = list(weights) if weights is not None else background_like_spectrum(n_channels)
        self._population = list(range(n_channels))
        self._seq = 0
        self._clock = 0.0
        self._total = 0
        self._device_seconds = 1000
        self.closed = False

        # Failure-mode switches the tests flip.
        self.silent = False
        self.stuck_channel: Optional[int] = None
        self.reset_at: Optional[int] = None

    # -- RadiaCodeSource interface -----------------------------------------

    def read_batch(self) -> Batch:
        if self.closed:
            # The real transport raises once closed. Modelling that is what
            # exposes a pump thread that outlived its source.
            raise SourceError('read from a closed source')
        self._clock += max(self.refresh_s, 1e-6)
        self._seq += 1
        self._device_seconds += int(self.refresh_s)

        if self.reset_at is not None and self._seq >= self.reset_at:
            self._total = 0
            self._device_seconds = 0
            self.reset_at = None
            return Batch(self._seq, time.time(), self._clock, self._device_seconds,
                         (), 0, self._total)

        if self.silent:
            return Batch(self._seq, time.time(), self._clock, self._device_seconds,
                         (), 0, self._total)

        n = self._poisson(self.count_rate * self.refresh_s)
        if self.stuck_channel is not None:
            channels = tuple([self.stuck_channel] * n)
        else:
            channels = tuple(sorted(
                self.rng.choices(self._population, weights=self.weights, k=n)
            )) if n else ()
        self._total += n
        return Batch(self._seq, time.time(), self._clock, self._device_seconds,
                     channels, len(channels), self._total)

    def _poisson(self, lam: float) -> int:
        # Knuth's method; lambda here is small so the loop is short.
        limit = math.exp(-lam)
        k, p = 0, 1.0
        while True:
            p *= self.rng.random()
            if p <= limit:
                return k
            k += 1

    def reference_spectrum(self) -> list[int]:
        return [max(1, int(weight * 5_000_000)) for weight in self.weights]

    def serial(self) -> str:
        return 'FAKE-000000'

    def firmware(self) -> str:
        return '0.0 (fake)'

    def energy_calibration(self) -> tuple:
        return (4.0936, 2.3720, 0.000362)

    def measure_count_rate(self, seconds: float = 20.0) -> float:
        start = self._total
        for _ in range(max(1, int(seconds / max(self.refresh_s, 1e-6)))):
            self.read_batch()
        return (self._total - start) / seconds if seconds else 0.0

    def close(self) -> None:
        self.closed = True

    @property
    def resets_seen(self) -> int:
        return 0


@pytest.fixture
def fake_source() -> FakeSource:
    return FakeSource()


@pytest.fixture
def fast_source() -> FakeSource:
    """A hot source, so tests that need many photons finish quickly."""
    return FakeSource(count_rate=2000.0)
