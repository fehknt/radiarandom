"""The generator: source -> health tests -> pool -> (physical | DRBG) output.

Two output modes, kept strictly separate so a caller always knows what it got:

``physical``
    256-bit blocks released only as fast as the detector actually supplies
    min-entropy -- 320 banked bits per 256 emitted. "Full entropy" in the
    SP 800-90C sense, which rests on HMAC-SHA-512 being a good conditioning
    function rather than on an information-theoretic extraction argument; see
    DESIGN.md section 4. Roughly 0.7 bytes per second on indoor background.
    This is what you want for seeding.

``drbg``
    HMAC_DRBG(SHA-512) instantiated from a physical block and reseeded from
    fresh physical blocks whenever they are available. Unlimited rate,
    computational security. This is what you want for filling a file or
    feeding a statistical test battery.

Nothing is emitted in either mode until the start-up test has passed, and any
latched health failure stops output immediately.

Entropy is credited per unit of *confirmed live detector time*, using the
Poisson model in :mod:`radiarandom.entropy`. Credit only accrues while the
device demonstrably advanced -- new photons, or its own accumulation clock
ticking -- so a frozen or unplugged detector earns nothing even before the
stall test fires.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Iterator, Optional, Sequence

from . import entropy as entropy_mod
from .conditioner import BLOCK_BYTES, BLOCK_COST_BITS, EntropyPool, HmacDrbg, serialize_batch
from .device import Batch, N_CHANNELS, RadiaCodeSource
from .health import HealthFailure, HealthMonitor, STARTUP_SAMPLES

_log = logging.getLogger(__name__)

#: Reseed the DRBG at least this often when physical entropy is available.
RESEED_SECONDS = 60.0
#: ...and at least this often by volume.
RESEED_BYTES = 1 << 20
#: Batches between recomputations of the entropy assessment.
ASSESSMENT_REFRESH_BATCHES = 64


class GeneratorError(RuntimeError):
    pass


class Generator:
    """Drives a :class:`RadiaCodeSource` and hands out random bytes.

    Not implicitly threaded: call :meth:`run_background` if you want a pump
    thread (the daemons and DRBG mode do), otherwise every read pumps inline.
    """

    def __init__(
        self,
        source: RadiaCodeSource,
        assessment: Optional[entropy_mod.Assessment] = None,
        startup_samples: int = STARTUP_SAMPLES,
        use_live_estimate: bool = True,
        reference_spectrum: Optional[Sequence[float]] = None,
        safety_factor: float = entropy_mod.DEFAULT_SAFETY_FACTOR,
        h_channel: float = entropy_mod.DEFAULT_H_CHANNEL,
        personalization: bytes = b'radiarandom',
    ) -> None:
        self.source = source
        self.use_live_estimate = use_live_estimate
        self.personalization = personalization
        self.safety_factor = safety_factor
        self.reference_probs = (
            entropy_mod.normalise(reference_spectrum) if reference_spectrum else None
        )

        # The bootstrap assessment credits nothing: its rate is zero until the
        # detector has been measured. Entropy is never assumed, only observed.
        self.assessment = assessment or entropy_mod.Assessment(
            channel_probs=self.reference_probs or entropy_mod.uniform_channel_probs(),
            count_rate=0.0,
            safety_factor=safety_factor,
            h_channel=h_channel,
            origin='bootstrap (no rate measured yet)',
        )
        self._current = self.assessment

        self.pool = EntropyPool()
        self.rate = entropy_mod.RateEstimator()
        self.mcv = entropy_mod.MostCommonValue(N_CHANNELS)
        self.monitor = HealthMonitor(
            h_per_photon=h_channel,
            reference_spectrum=reference_spectrum,
            startup_samples=startup_samples,
        )

        self._drbg: Optional[HmacDrbg] = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._batches_seen = 0
        self._photons_seen = 0
        self._blocks_released = 0
        self._bits_credited = 0.0
        self._first_monotonic: Optional[float] = None
        self._last_monotonic: Optional[float] = None
        self._last_credit_monotonic: Optional[float] = None
        self._last_rate_monotonic: Optional[float] = None
        self._last_device_seconds: Optional[int] = None

    # ------------------------------------------------------------------ pump

    def _consume(self, batch: Batch) -> None:
        """Health-test, account, and absorb one batch. Caller holds the lock."""
        for event in self.monitor.observe(batch):
            if event.severity == 'failure':
                _log.error('%s', event)
            elif event.severity == 'warning':
                _log.warning('%s', event)
            else:
                _log.info('%s', event)
        if self.monitor.failed:
            raise HealthFailure(self.monitor.failure_reason or 'health test failed')

        now = batch.host_monotonic
        if self._first_monotonic is None:
            self._first_monotonic = now
            self._last_credit_monotonic = now
            self._last_rate_monotonic = now
        self._last_monotonic = now
        self._batches_seen += 1

        # Did the device actually advance? New photons, or its own accumulation
        # clock ticking, both prove it is alive. Anything else earns no credit.
        device_advanced = (
            batch.count > 0
            or (self._last_device_seconds is not None
                and batch.device_seconds != self._last_device_seconds)
        )
        self._last_device_seconds = batch.device_seconds

        # Two separate clocks, and conflating them is a bug.
        #
        # The *rate* clock must advance on every batch: wall time passes
        # whether or not this particular poll caught the device mid-refresh.
        # Anchoring it to the credit clock instead double-counts the interval
        # every time a poll finds nothing new, which inflates the denominator
        # and under-reports the count rate -- measured at 1.8x on real
        # hardware (8.97/s reported against 16.21/s actual).
        rate_elapsed = now - (self._last_rate_monotonic or now)
        self._last_rate_monotonic = now
        self.rate.update(batch.count, rate_elapsed)

        if batch.count:
            self._photons_seen += batch.count
            self.mcv.update(batch.channels)

        if self._batches_seen % ASSESSMENT_REFRESH_BATCHES == 0:
            self._refresh_assessment()

        # The *credit* clock only advances when the device proved it is alive,
        # so a frozen detector banks nothing.
        bits = 0.0
        if device_advanced:
            credit_elapsed = now - (self._last_credit_monotonic or now)
            bits = self._current.credit(credit_elapsed)
            self._last_credit_monotonic = now
            self._bits_credited += bits

        self.pool.absorb(serialize_batch(batch), bits)

    def _refresh_assessment(self) -> None:
        """Recompute the budget from the measured rate and spectrum.

        Uses a lower confidence bound on the count rate (over-stating the rate
        would over-credit) and, when the live spectrum is available, whichever
        of the live and reference distributions yields the *smaller* budget.
        """
        rate = self.rate.lower_bound()
        if rate is None:
            return

        candidates: list[entropy_mod.Assessment] = []
        if self.reference_probs:
            candidates.append(entropy_mod.Assessment(
                channel_probs=self.reference_probs,
                count_rate=rate,
                safety_factor=self.safety_factor,
                h_channel=self.assessment.h_channel,
                origin='reference spectrum + measured rate',
            ))
        if self.use_live_estimate:
            live_probs = self.mcv.probabilities()
            if live_probs is not None:
                candidates.append(entropy_mod.Assessment(
                    channel_probs=live_probs,
                    count_rate=rate,
                    safety_factor=self.safety_factor,
                    h_channel=self.assessment.h_channel,
                    origin='live spectrum + measured rate',
                ))
        if not candidates:
            candidates.append(self.assessment.with_rate(rate, 'measured rate'))

        self._current = min(candidates, key=lambda a: a.bits_per_second())

        # Give the rate-excursion health test a baseline to compare against.
        # Without this it has no expected rate and silently never fires. The
        # baseline is latched on the first measurement rather than tracking the
        # current rate, because a test that follows the thing it is watching
        # cannot detect that thing changing.
        if self.monitor.expected_count_rate is None and self.rate.seconds >= 60.0:
            # The point estimate, not `rate` -- that one is a deliberately low
            # confidence bound for entropy accounting, and using it as a health
            # baseline would guarantee a permanent positive excursion.
            #
            # Latched only after a minute of data: baselining on the first few
            # seconds bakes in the ramp-up noise and the rate test then warns
            # about the detector failing to be as slow as it briefly looked.
            self.monitor.expected_count_rate = self.rate.point_estimate

    def pump_once(self) -> Batch:
        """Read one batch from the device and fold it in."""
        batch = self.source.read_batch()
        with self._lock:
            self._consume(batch)
        return batch

    def pump_until(self, predicate, timeout: Optional[float] = None) -> bool:
        deadline = None if timeout is None else time.perf_counter() + timeout
        while not predicate():
            if self._stop.is_set():
                return False
            if deadline is not None and time.perf_counter() > deadline:
                return False
            self.pump_once()
            time.sleep(self.source.poll_interval)
        return True

    def run_background(self) -> None:
        """Start a thread that keeps the pool topped up forever."""
        if self._thread is not None:
            return

        def loop() -> None:
            interval = self.source.poll_interval
            while not self._stop.is_set():
                try:
                    self.pump_once()
                except HealthFailure:
                    _log.error('health failure in pump thread; stopping')
                    self._stop.set()
                    return
                except Exception:  # pragma: no cover - transport errors
                    _log.exception('pump thread error; stopping')
                    self._stop.set()
                    return
                time.sleep(interval)

        self._thread = threading.Thread(target=loop, name='radiarandom-pump', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    # ------------------------------------------------------------- start-up

    def wait_for_startup(self, progress=None, timeout: Optional[float] = None) -> None:
        """Block until the start-up test has passed."""
        deadline = None if timeout is None else time.perf_counter() + timeout
        last_report = 0.0
        while not self.monitor.started:
            if self.monitor.failed:
                raise HealthFailure(self.monitor.failure_reason or 'health test failed')
            if deadline is not None and time.perf_counter() > deadline:
                passed, needed = self.monitor.startup_progress
                raise GeneratorError(
                    f'start-up test did not complete within the timeout '
                    f'({passed}/{needed} photons)')
            self.pump_once()
            if progress is not None and time.perf_counter() - last_report > 1.0:
                last_report = time.perf_counter()
                progress(*self.monitor.startup_progress)
            time.sleep(self.source.poll_interval)
        if progress is not None:
            progress(*self.monitor.startup_progress)

    def _require_healthy(self) -> None:
        if self.monitor.failed:
            raise HealthFailure(self.monitor.failure_reason or 'health test failed')
        if not self.monitor.started:
            raise GeneratorError('start-up test has not completed; call wait_for_startup()')

    # ------------------------------------------------------- physical output

    def _try_extract(self) -> Optional[bytes]:
        """Take a block if one is ready. Atomic against the pump thread."""
        with self._lock:
            if not self.pool.ready():
                return None
            self._blocks_released += 1
            return self.pool.extract_block()

    def physical_block(self, timeout: Optional[float] = None) -> bytes:
        """Return one 256-bit full-entropy block, waiting for the detector.

        Safe whether or not a pump thread is running: the readiness check and
        the extraction happen together under the lock, so two callers can never
        both believe the same block is theirs.
        """
        self._require_healthy()
        deadline = None if timeout is None else time.perf_counter() + timeout
        while True:
            block = self._try_extract()
            if block is not None:
                return block
            if self._stop.is_set():
                raise GeneratorError('generator stopped while waiting for entropy')
            if deadline is not None and time.perf_counter() > deadline:
                raise TimeoutError(
                    f'timed out waiting for entropy; pool holds '
                    f'{self.pool.entropy_bits:.0f} of {BLOCK_COST_BITS} bits')
            if self._thread is None:
                self.pump_once()
            time.sleep(self.source.poll_interval)
            if self.monitor.failed:
                raise HealthFailure(self.monitor.failure_reason or 'health test failed')

    def physical_bytes(self, n: int, timeout: Optional[float] = None) -> bytes:
        """Exactly ``n`` bytes of full-entropy output. Rate-limited by decay."""
        out = bytearray()
        while len(out) < n:
            out += self.physical_block(timeout=timeout)
        return bytes(out[:n])

    def physical_stream(self, timeout: Optional[float] = None) -> Iterator[bytes]:
        while not self._stop.is_set():
            yield self.physical_block(timeout=timeout)

    # ----------------------------------------------------------- DRBG output

    def _ensure_drbg(self, timeout: Optional[float] = None) -> HmacDrbg:
        if self._drbg is None:
            seed = self.physical_block(timeout=timeout)
            nonce = self.physical_block(timeout=timeout)
            self._drbg = HmacDrbg(seed, nonce=nonce, personalization=self.personalization)
            _log.info('DRBG instantiated from %d bits of physical entropy',
                      len(seed + nonce) * 8)
        return self._drbg

    def _maybe_reseed(self) -> None:
        """Fold fresh detector entropy into the DRBG when any is available.

        Opportunistic by default: if the detector has not finished another
        block we carry on generating rather than stalling, because the DRBG is
        still secure. Only the hard SP 800-90A reseed interval forces a wait.
        """
        drbg = self._drbg
        if drbg is None:
            return
        if drbg.needs_reseed:
            drbg.reseed(self.physical_block())
            drbg.bytes_generated = 0
            return
        due = (
            drbg.bytes_generated >= RESEED_BYTES
            or (time.perf_counter() - drbg.last_reseed_monotonic) >= RESEED_SECONDS
        )
        if not due:
            return
        block = self._try_extract()
        if block is not None:
            drbg.reseed(block)
            drbg.bytes_generated = 0

    def read(self, n: int, timeout: Optional[float] = None) -> bytes:
        """Return ``n`` bytes of DRBG output, reseeded from the detector."""
        self._require_healthy()
        drbg = self._ensure_drbg(timeout=timeout)
        out = bytearray()
        while len(out) < n:
            self._maybe_reseed()
            if self.monitor.failed:
                raise HealthFailure(self.monitor.failure_reason or 'health test failed')
            want = min(n - len(out), RESEED_BYTES)
            out += drbg.generate(want)
        return bytes(out[:n])

    def stream(self, chunk: int = 65536) -> Iterator[bytes]:
        while not self._stop.is_set():
            yield self.read(chunk)

    # ---------------------------------------------------------------- status

    @property
    def count_rate(self) -> Optional[float]:
        return self.rate.point_estimate

    @property
    def entropy_rate_bits_per_s(self) -> float:
        return self._current.bits_per_second()

    def stats(self) -> dict:
        return {
            'batches': self._batches_seen,
            'photons': self._photons_seen,
            'count_rate': self.count_rate,
            'count_rate_lower_bound': self.rate.lower_bound(),
            'assessment': self._current.describe(),
            'live_channel_min_entropy': self.mcv.min_entropy(),
            'entropy_rate_bits_per_s': self.entropy_rate_bits_per_s,
            'bits_credited': self._bits_credited,
            'blocks_released': self._blocks_released,
            'pool': self.pool.stats(),
            'drbg': self._drbg.stats() if self._drbg else None,
            'health': self.monitor.status(),
        }
