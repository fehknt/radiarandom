"""Continuous health tests for the noise source.

Why not the textbook RCT and APT
--------------------------------
NIST SP 800-90B section 4.4 specifies two continuous tests -- the Repetition
Count Test and the Adaptive Proportion Test -- and both are defined on an
*ordered* sequence of samples. The RadiaCode does not give us one. It reports a
cumulative per-channel histogram, so differencing two reads yields the
**multiset** of pulse-height channels detected in between, with the order
irretrievably gone.

Expanding that multiset into a sorted list and feeding it to the textbook tests
is wrong, and wrong in a way that bites hardest exactly when the detector is
working best:

* Sorting manufactures runs. Two photons in the same channel become adjacent
  by construction, so at high count rates the RCT fires constantly on a
  perfectly healthy detector. With a check source raising the rate to a few
  hundred counts/s this is not a corner case, it is every batch.
* Sorting biases the APT. The test keys on the *first* sample in each window;
  in ascending order that is systematically a low channel, which is where most
  of the probability mass sits, so match counts are inflated.

So this module uses order-free tests with the same false-positive discipline
(``alpha = 2^-20``) that catch the same failures:

**Proportion test** -- over a sliding window of 512 photons, no single channel
may account for more than ``cutoff`` of them. This is the APT's statistic with
the window's arbitrary first sample replaced by the *most common* channel,
which is strictly more sensitive. Taking a maximum over ``K`` channels needs a
multiple-comparison correction, so the cutoff is computed against ``alpha / K``
via a union bound -- conservative and exact enough.

**Repetition test** -- consecutive batches with identical, non-empty channel
multisets. A live detector essentially never repeats itself; a wedged device
replaying a buffer does nothing else.

Plus three detector-specific tests, because a physically dead detector emits
*no* samples rather than bad ones and would sail through any test that only
looks at the samples it does emit:

* **stall** -- silence far longer than the Poisson model permits;
* **rate excursion** -- the count rate collapsing or saturating;
* **spectral collapse** -- counts piling into a narrow band, the signature of a
  bias-voltage or gain fault, which would gut the real entropy while the
  proportion test still passed.

A failure latches. Once the source is unhealthy the generator stops rather than
degrading silently, which is the only safe behaviour for something that may be
seeding keys.
"""

from __future__ import annotations

import collections
import dataclasses
import math
import time
from typing import Optional, Sequence

from .device import Batch, N_CHANNELS

#: SP 800-90B false-positive target for the continuous health tests.
ALPHA_EXPONENT = 20
ALPHA = 2.0 ** -ALPHA_EXPONENT

#: Sliding window, in photons, for the proportion test.
PROPORTION_WINDOW = 512

#: Consecutive identical batches tolerated before calling the device wedged.
REPEAT_CUTOFF = 3

#: Photons that must pass before output is permitted (SP 800-90B 4.3).
STARTUP_SAMPLES = 1024

#: No single channel may ever exceed this fraction of the proportion window,
#: however peaky the calibrated spectrum is. This is the floor that still
#: catches a genuinely stuck ADC when the baseline itself looks peaky.
PROPORTION_HARD_CEILING = 0.5

#: Length of the sliding window used by the rate-excursion test, in seconds.
RATE_WINDOW_S = 60.0

#: Sigma threshold for the rate-excursion warning. Generous, because we are
#: looking for a detector that has broken, not for weather.
RATE_SIGMA = 8.0

#: Weight given to the newest window when updating the rolling baselines.
#:
#: The baselines have to track the environment or they become the "warns
#: forever" bug again the first time anything legitimately changes -- a source
#: is added or removed, the detector is moved, the room warms up. But they must
#: not track so eagerly that a slow degradation is followed all the way down
#: and never reported. Two timescales resolve it: a *fast* window is compared
#: against a *slowly* adapting reference, so an abrupt change raises a warning
#: for as long as the transition lasts and then settles.
#:
#: What deliberately does NOT adapt is the boiling-frog guard: the proportion
#: hard ceiling, and the entropy budget, which is recomputed from the live
#: spectrum and simply credits less when the detector gets worse.
BASELINE_BLEND = 0.25

#: Photons between re-derivations of the proportion cutoff.
RECALIBRATE_PHOTONS = 4096

#: Blend used for the window that actually triggered a shape warning.
#:
#: Once the transition has been reported there is no value in reporting it
#: again for the next half hour, so the baseline jumps most of the way to the
#: new shape. The effect is: warn while it changes, then settle. Slow drift
#: still only gets BASELINE_BLEND, so it is tracked without being announced --
#: and the entropy budget, which reprices from the live spectrum, is what
#: protects against being boiled slowly.
BASELINE_BLEND_ON_CHANGE = 0.6

#: Coarse bins for the spectral-shape test. Fine enough to notice a collapse,
#: coarse enough that ordinary gain drift does not trip it.
SHAPE_BINS = 32


class HealthFailure(RuntimeError):
    """A health test failed; the source must not be used."""


@dataclasses.dataclass(frozen=True)
class HealthEvent:
    kind: str
    severity: str  # 'info' | 'warning' | 'failure'
    message: str
    at: float = dataclasses.field(default_factory=time.time)

    def __str__(self) -> str:
        return f'[{self.severity}] {self.kind}: {self.message}'


def _binomial_cdf(k: int, n: int, p: float) -> float:
    """``P(X <= k)`` for ``X ~ Binomial(n, p)``, summed in log space."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    log_p = math.log(p)
    log_q = math.log1p(-p)
    total = 0.0
    for i in range(0, k + 1):
        log_term = (
            math.lgamma(n + 1)
            - math.lgamma(i + 1)
            - math.lgamma(n - i + 1)
            + i * log_p
            + (n - i) * log_q
        )
        total += math.exp(log_term)
    return min(1.0, total)


def repetition_count_cutoff(h_per_sample: float, alpha_exponent: int = ALPHA_EXPONENT) -> int:
    """SP 800-90B 4.4.1 cutoff ``C = 1 + ceil(-log2(alpha) / H)``.

    Applied here to whole batches rather than to individual photons.
    """
    if h_per_sample <= 0:
        return 1
    return 1 + math.ceil(alpha_exponent / h_per_sample)


def proportion_cutoff(
    h_per_photon: float,
    window: int = PROPORTION_WINDOW,
    alpha: float = ALPHA,
    n_channels: int = N_CHANNELS,
) -> int:
    """Smallest ``C`` with ``P(any channel appears >= C times) <= alpha``.

    Bounds the maximum over ``n_channels`` by a union bound, so each individual
    channel is tested at ``alpha / n_channels``.
    """
    p_max = min(1.0, 2.0 ** -h_per_photon) if h_per_photon > 0 else 1.0
    per_channel_alpha = alpha / max(1, n_channels)
    target = 1.0 - per_channel_alpha
    for c in range(0, window + 1):
        if _binomial_cdf(c, window, p_max) >= target:
            return c + 1
    return window + 1


class HealthMonitor:
    """Runs every health test over the batch stream.

    Feed it every batch, including empty ones -- the stall detector needs the
    gaps. All timing is taken from :attr:`Batch.host_monotonic` rather than
    from the wall clock, so replayed or simulated streams behave the same as
    live ones.
    """

    def __init__(
        self,
        h_per_photon: float,
        expected_count_rate: Optional[float] = None,
        reference_spectrum: Optional[Sequence[float]] = None,
        startup_samples: int = STARTUP_SAMPLES,
        stall_grace_s: float = 30.0,
        n_symbols: int = N_CHANNELS,
        calibrate_photons: Optional[int] = None,
    ) -> None:
        self.h_per_photon = h_per_photon
        self.n_symbols = n_symbols
        self.startup_samples = startup_samples

        self.assumed_proportion_cutoff = proportion_cutoff(h_per_photon, n_channels=n_symbols)
        #: Non-adaptive failure threshold. Nothing re-calibration does can move
        #: this, so a stuck ADC is caught however peaky the baseline becomes.
        self.proportion_ceiling = int(PROPORTION_HARD_CEILING * PROPORTION_WINDOW)
        self._proportion_warned = False
        # While calibrating we do not yet know the detector's real pulse-height
        # distribution, so the tight assumed cutoff would fire on any legitimate
        # peak -- a check source, say. During this phase only the hard ceiling
        # applies, which still catches a grossly stuck channel. No output is
        # released in the meantime: calibration finishes no later than the
        # start-up test, which gates every output path.
        self.proportion_cutoff = int(PROPORTION_HARD_CEILING * PROPORTION_WINDOW)
        self.proportion_cutoff_origin = 'calibrating (hard ceiling only)'
        self.repeat_cutoff = REPEAT_CUTOFF
        # Kept for reporting: what a per-photon RCT cutoff would have been.
        self.rct_cutoff = repetition_count_cutoff(h_per_photon)

        # Proportion-test state: a sliding window of the last N photons.
        self._window: collections.deque = collections.deque(maxlen=PROPORTION_WINDOW)
        self._window_counts: collections.Counter = collections.Counter()

        # Repetition-test state.
        self._last_fingerprint: Optional[tuple] = None
        self._repeat_run = 0

        # Start-up state.
        self._samples_passed = 0
        self._started = startup_samples <= 0

        # Timing / liveness state.
        self.expected_count_rate = expected_count_rate
        self.stall_grace_s = stall_grace_s
        self._first_monotonic: Optional[float] = None
        self._last_monotonic: Optional[float] = None
        self._last_photon_monotonic: Optional[float] = None
        self._total_photons = 0
        self._last_device_seconds: Optional[int] = None
        # Sliding window for the rate-excursion test.
        self._window_photons = 0
        self._window_start: Optional[float] = None

        # Spectral shape state.
        #
        # The reference is learned from *this session*, not taken from the
        # device's lifetime spectrum. The lifetime spectrum covers months of
        # whatever the detector happened to be near, in whatever location, at
        # whatever temperature; comparing the current background against it
        # produces a chi-square in the thousands on a perfectly healthy device
        # and warns forever. A warning that always fires trains the operator to
        # ignore warnings, which is worse than having no test at all.
        #
        # Observed on RC-103-013128: chi2 approximately 2980 against dof 18,
        # every four minutes, with nothing whatsoever wrong.
        #
        # So we calibrate against the first `calibrate_photons` of the run and
        # then watch for drift away from *that*, which is what actually
        # indicates a gain or bias-voltage fault. The lifetime spectrum is kept
        # only for a one-shot informational comparison once calibration ends.
        # Calibration must complete no later than the start-up test, so that
        # the permissive calibration phase never overlaps with output.
        self.calibrate_photons = (
            calibrate_photons if calibrate_photons is not None
            else (startup_samples if startup_samples > 0 else 2048)
        )
        self._prior_shape = self._coarse(reference_spectrum) if reference_spectrum else None
        self._reference_shape: Optional[list[float]] = None
        self._calibration = [0] * SHAPE_BINS
        self._calibration_total = 0
        self._calibration_channels: collections.Counter = collections.Counter()
        # Decaying channel histogram driving periodic cutoff re-derivation.
        self._rolling_channels: dict = {}
        self._rolling_total = 0.0
        self._photons_since_recalibration = 0
        self.recalibrations = 0
        self._recent = [0] * SHAPE_BINS
        self._recent_total = 0

        self.failed = False
        self.failure_reason: Optional[str] = None
        self.events: list[HealthEvent] = []

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _coarse(counts: Sequence[float]) -> list[float]:
        per_bin = max(1, len(counts) // SHAPE_BINS)
        binned = [0.0] * SHAPE_BINS
        for channel, value in enumerate(counts):
            binned[min(SHAPE_BINS - 1, channel // per_bin)] += value
        total = sum(binned)
        if total <= 0:
            return binned
        return [value / total for value in binned]

    def _fail(self, kind: str, message: str) -> HealthEvent:
        self.failed = True
        if self.failure_reason is None:
            self.failure_reason = f'{kind}: {message}'
        event = HealthEvent(kind=kind, severity='failure', message=message)
        self.events.append(event)
        return event

    def _warn(self, kind: str, message: str) -> HealthEvent:
        event = HealthEvent(kind=kind, severity='warning', message=message)
        self.events.append(event)
        return event

    @property
    def started(self) -> bool:
        return self._started and not self.failed

    @property
    def startup_progress(self) -> tuple[int, int]:
        return (min(self._samples_passed, self.startup_samples), self.startup_samples)

    @property
    def observed_count_rate(self) -> Optional[float]:
        """Photons per second, measured up to the *last photon*.

        Deliberately excludes any trailing silence. Measuring up to "now"
        instead would let a stall deflate the very rate estimate that sets the
        stall threshold, so the longer the detector stayed dead the more
        patient the test would become -- and it would never fire.
        """
        if self._first_monotonic is None or self._last_photon_monotonic is None:
            return None
        elapsed = self._last_photon_monotonic - self._first_monotonic
        if elapsed < 5.0 or self._total_photons <= 0:
            return None
        return self._total_photons / elapsed

    # ------------------------------------------------------------------ tests

    def _proportion_test(self, channels: Sequence[int]) -> list[HealthEvent]:
        """Order-free APT: no channel may dominate a window of photons."""
        events: list[HealthEvent] = []
        for channel in channels:
            if len(self._window) == self._window.maxlen:
                evicted = self._window[0]
                self._window_counts[evicted] -= 1
                if self._window_counts[evicted] <= 0:
                    del self._window_counts[evicted]
            self._window.append(channel)
            self._window_counts[channel] += 1

            if len(self._window) < self._window.maxlen:
                continue
            channel_count = self._window_counts[channel]

            # Two thresholds, and the distinction matters. Exceeding the hard
            # ceiling means one channel is taking half of everything, which no
            # legitimate source does -- that is a stuck ADC and it is fatal.
            # Exceeding the *adaptive* cutoff just means the spectrum is
            # peakier than the current baseline expects, which is exactly what
            # happens when someone puts a check source next to the detector.
            # Failing there would reject the configuration the documentation
            # recommends, so it warns and re-derives instead. The entropy
            # budget handles the peak quantitatively: a narrower spectrum is
            # priced lower, so output slows rather than becoming overstated.
            if channel_count >= self.proportion_ceiling:
                events.append(self._fail(
                    'proportion',
                    f'channel {channel} accounts for {channel_count} of the last '
                    f'{PROPORTION_WINDOW} photons, past the hard ceiling of '
                    f'{self.proportion_ceiling}; detector appears stuck',
                ))
                return events

            if channel_count >= self.proportion_cutoff and not self._proportion_warned:
                self._proportion_warned = True
                events.append(self._warn(
                    'proportion',
                    f'channel {channel} accounts for {channel_count} of the last '
                    f'{PROPORTION_WINDOW} photons (cutoff {self.proportion_cutoff}); '
                    f're-deriving against the current spectrum'))
                events.extend(self._derive_proportion_cutoff(
                    self._rolling_channels or self._window_counts,
                    self._rolling_total or float(len(self._window)),
                    'excursion'))
        return events

    def _repetition_test(self, batch: Batch) -> list[HealthEvent]:
        """Identical consecutive batches mean the device is replaying, not measuring."""
        if not batch.count:
            return []
        fingerprint = batch.channels
        if fingerprint == self._last_fingerprint:
            self._repeat_run += 1
            if self._repeat_run >= self.repeat_cutoff:
                return [self._fail(
                    'repetition',
                    f'{self._repeat_run + 1} consecutive batches with an identical '
                    f'{batch.count}-photon spectrum; device appears wedged',
                )]
        else:
            self._last_fingerprint = fingerprint
            self._repeat_run = 0
        return []

    @staticmethod
    def _chi_square(observed: Sequence[int], expected_probs: Sequence[float],
                    total: int) -> tuple[float, int]:
        chi2 = 0.0
        dof = 0
        for expected_p, count in zip(expected_probs, observed):
            expected = expected_p * total
            if expected < 5.0:
                continue
            chi2 += (count - expected) ** 2 / expected
            dof += 1
        return chi2, dof

    def _shape_accumulate(self, channels: Sequence[int]) -> list[HealthEvent]:
        """Build the session baseline first, then watch for drift from it."""
        per_bin = max(1, self.n_symbols // SHAPE_BINS)

        if self._reference_shape is None:
            for channel in channels:
                self._calibration[min(SHAPE_BINS - 1, channel // per_bin)] += 1
                self._calibration_channels[channel] += 1
            self._calibration_total += len(channels)
            if self._calibration_total < self.calibrate_photons:
                return []
            total = self._calibration_total
            self._reference_shape = [count / total for count in self._calibration]
            events = [HealthEvent(
                'shape', 'info',
                f'spectral baseline calibrated on {total} photons')]
            self.events.append(events[0])
            events.extend(self._recalibrate_proportion())
            # One-shot sanity check against the device's lifetime spectrum.
            # Informational only: a mismatch usually means the detector has
            # simply moved, not that anything is wrong.
            if self._prior_shape is not None:
                chi2, dof = self._chi_square(self._calibration, self._prior_shape, total)
                if dof >= 4 and chi2 > 20.0 * dof:
                    events.append(self._warn(
                        'shape',
                        f'session baseline differs from the device lifetime '
                        f'spectrum (chi2={chi2:.0f}, dof={dof}). Expected if the '
                        f'detector has moved or a source is present; '
                        f'monitoring drift from the session baseline instead'))
            return events

        for channel in channels:
            self._recent[min(SHAPE_BINS - 1, channel // per_bin)] += 1
        self._recent_total += len(channels)
        return self._track_rolling_spectrum(channels)

    def _recalibrate_proportion(self) -> list[HealthEvent]:
        """Retune the proportion cutoff to the spectrum actually observed.

        The cutoff starts from an assumed per-photon min-entropy, which is only
        a guess about the detector's pulse-height distribution. A check source
        -- the standard, recommended way to raise the entropy rate -- puts a
        sharp peak in the spectrum and pushes one channel's share well above
        that guess. Refusing to operate in that case would be exactly backwards:
        the source *increases* the entropy rate, and the Poisson budget in
        :mod:`radiarandom.entropy` already accounts for the narrower spectrum by
        crediting less per photon.

        Observed on RC-103-013128 with an Am-241 source present: channel 25
        (~64 keV) held 71 of 512 photons against a cutoff of 71, and the
        capture halted after 100 seconds.

        So the cutoff is widened to fit the calibrated baseline, but never past
        :data:`PROPORTION_HARD_CEILING` of the window -- a channel taking half
        of everything is a stuck ADC no matter what the baseline looked like.
        The cutoff is only ever loosened, never tightened, so a quiet spectrum
        cannot make the test hair-trigger.
        """
        return self._derive_proportion_cutoff(
            self._calibration_channels, self._calibration_total, 'calibration')

    def _derive_proportion_cutoff(self, counts, total: float,
                                  reason: str) -> list:
        if total < 64 or not counts:
            return []
        p_hat = max(counts.values()) / total
        # 99% upper bound on the busiest channel's share.
        margin = 2.576 * math.sqrt(p_hat * (1.0 - p_hat) / max(1, total - 1))
        p_upper = min(1.0, p_hat + margin)
        if p_upper <= 0.0:
            return []
        h_observed = -math.log2(p_upper)
        fitted = proportion_cutoff(h_observed, n_channels=self.n_symbols)
        ceiling = int(PROPORTION_HARD_CEILING * PROPORTION_WINDOW)
        # Take the looser of the assumed and fitted cutoffs, then clamp. Never
        # tighter than assumed, so a quiet spectrum cannot make this
        # hair-trigger; never looser than the ceiling, so a stuck ADC is caught
        # whatever the baseline looked like.
        chosen = min(max(self.assumed_proportion_cutoff, fitted), ceiling)
        previous = self.proportion_cutoff
        self.proportion_cutoff = chosen
        self.proportion_cutoff_origin = (
            f'{reason} on {total:.0f} photons (busiest channel {p_hat:.1%})')
        if chosen == previous:
            return []
        self.recalibrations += 1
        self._proportion_warned = False
        event = HealthEvent(
            'proportion', 'info',
            f'proportion cutoff {previous} -> {chosen} of {PROPORTION_WINDOW} '
            f'after {reason} (assumed {self.assumed_proportion_cutoff}, '
            f'fitted {fitted}, ceiling {ceiling}; busiest channel {p_hat:.1%})')
        self.events.append(event)
        return [event]

    def _track_rolling_spectrum(self, channels) -> list:
        """Keep a decaying channel histogram and re-derive the cutoff from it.

        Without this the cutoff is fixed at start-up forever: add a check
        source an hour in and the proportion test rejects a detector that is
        working better than when it was calibrated.
        """
        for channel in channels:
            self._rolling_channels[channel] = self._rolling_channels.get(channel, 0.0) + 1.0
        self._rolling_total += len(channels)
        self._photons_since_recalibration += len(channels)
        if self._photons_since_recalibration < RECALIBRATE_PHOTONS:
            return []
        self._photons_since_recalibration = 0
        events = self._derive_proportion_cutoff(
            self._rolling_channels, self._rolling_total, 're-calibration')
        # Forget slowly so the histogram keeps following the environment.
        factor = 1.0 - BASELINE_BLEND
        self._rolling_channels = {c: v * factor
                                  for c, v in self._rolling_channels.items() if v * factor > 1e-6}
        self._rolling_total *= factor
        return events

    def _check_shape(self) -> list[HealthEvent]:
        if self._reference_shape is None or self._recent_total < 4096:
            return []
        chi2, dof = self._chi_square(self._recent, self._reference_shape,
                                     self._recent_total)
        observed = [count / self._recent_total for count in self._recent]
        self._recent = [0] * SHAPE_BINS
        self._recent_total = 0

        drifted = dof >= 4 and chi2 > 20.0 * dof
        # Track the environment. A step change -- a source arriving, the
        # detector moving -- is reported once and then absorbed, rather than
        # nagging for the rest of the session.
        blend = BASELINE_BLEND_ON_CHANGE if drifted else BASELINE_BLEND
        self._reference_shape = [
            (1.0 - blend) * ref + blend * obs
            for ref, obs in zip(self._reference_shape, observed)]

        if dof < 4:
            return []
        # Deliberately loose: we are looking for a collapsed spectrum, not for
        # mild gain drift. Twenty times the degrees of freedom is far into the
        # tail for any healthy detector.
        if chi2 > 20.0 * dof:
            return [self._warn(
                'shape',
                f'spectrum shape has drifted from this session baseline '
                f'(chi2={chi2:.0f}, dof={dof}); check for gain drift, a '
                f'temperature excursion, or a source that has moved',
            )]
        return []

    def _stall_limit(self) -> float:
        """How long a silence is allowed before the detector is called dead.

        Under Poisson, the chance of seeing nothing for ``t`` seconds is
        ``exp(-rate*t)``; pick ``t`` so that is about ``2^-20``, then floor it
        with a grace period so a USB hiccup or a genuinely quiet location does
        not trip the alarm.
        """
        rate = self.observed_count_rate or self.expected_count_rate
        if not rate or rate <= 0:
            return max(self.stall_grace_s, 120.0)
        poisson_limit = ALPHA_EXPONENT * math.log(2.0) / rate
        return max(self.stall_grace_s, 3.0 * poisson_limit)

    def _check_rate(self) -> list[HealthEvent]:
        """Compare a *recent window* against the latched baseline.

        Comparing the cumulative average against a baseline derived from that
        same cumulative average is circular: the sigma shrinks as the run gets
        longer, so any systematic difference -- including the deliberate
        pessimism of a lower-confidence-bound baseline -- eventually reads as
        many sigma and the warning fires forever. Measured on a simulated
        1485/s source: a 0.45% offset reported as +12.9 sigma, every batch.

        So the window is bounded and independent, and the baseline is a fixed
        point estimate latched once.
        """
        if self._window_start is None or self._last_monotonic is None:
            return []
        if not self.expected_count_rate or self.expected_count_rate <= 0:
            return []
        elapsed = self._last_monotonic - self._window_start
        if elapsed < RATE_WINDOW_S:
            return []

        observed = self._window_photons / elapsed
        expected = self.expected_count_rate
        self._window_start = self._last_monotonic
        self._window_photons = 0

        # Poisson standard deviation of a rate measured over `elapsed`.
        sigma = math.sqrt(expected / elapsed)
        if sigma <= 0:
            return []
        z = (observed - expected) / sigma

        # Adapt toward what the detector is actually doing. Moving it, or
        # adding a source, legitimately changes the rate; a baseline frozen at
        # start-up would warn about that for the rest of the run. Excursions
        # adapt more slowly, so a genuine collapse is reported for several
        # windows before the baseline concedes.
        excursion = abs(z) > RATE_SIGMA
        blend = BASELINE_BLEND * (0.25 if excursion else 1.0)
        self.expected_count_rate = (1.0 - blend) * expected + blend * observed

        if excursion:
            return [self._warn(
                'rate',
                f'count rate over the last {elapsed:.0f}s was {observed:.2f}/s '
                f'against a {expected:.2f}/s baseline ({z:+.1f} sigma)',
            )]
        return []

    # ------------------------------------------------------------------- feed

    def observe(self, batch: Batch) -> list[HealthEvent]:
        """Run every test against one batch. Returns any new events."""
        if self.failed:
            return []

        events: list[HealthEvent] = []
        now = batch.host_monotonic
        if self._first_monotonic is None:
            self._first_monotonic = now
            self._last_photon_monotonic = now
        self._last_monotonic = now
        self._last_device_seconds = batch.device_seconds

        if batch.count:
            for channel in batch.channels:
                if not (0 <= channel < self.n_symbols):
                    return [self._fail(
                        'range', f'channel {channel} outside 0..{self.n_symbols - 1}')]

            self._last_photon_monotonic = now
            self._total_photons += batch.count
            if self._window_start is None:
                self._window_start = now
            self._window_photons += batch.count

            events.extend(self._repetition_test(batch))
            if self.failed:
                return events
            events.extend(self._proportion_test(batch.channels))
            if self.failed:
                return events

            events.extend(self._shape_accumulate(batch.channels))
            events.extend(self._check_shape())

            self._samples_passed += batch.count
            if not self._started and self._samples_passed >= self.startup_samples:
                self._started = True
                events.append(HealthEvent(
                    'startup', 'info',
                    f'start-up test passed after {self._samples_passed} photons'))
                self.events.append(events[-1])

        # Stall detection runs on every batch, including empty ones.
        assert self._last_photon_monotonic is not None
        silence = now - self._last_photon_monotonic
        limit = self._stall_limit()
        if silence > limit:
            events.append(self._fail(
                'stall',
                f'no photons for {silence:.1f}s (limit {limit:.1f}s); detector '
                f'appears dead or disconnected'))
            return events

        events.extend(self._check_rate())
        return events

    # ----------------------------------------------------------------- status

    def status(self) -> dict:
        passed, needed = self.startup_progress
        return {
            'healthy': not self.failed,
            'started': self.started,
            'failure_reason': self.failure_reason,
            'startup_passed': passed,
            'startup_needed': needed,
            'photons_seen': self._total_photons,
            'observed_count_rate': self.observed_count_rate,
            'proportion_cutoff': self.proportion_cutoff,
            'proportion_cutoff_origin': self.proportion_cutoff_origin,
            'proportion_window': PROPORTION_WINDOW,
            'repeat_cutoff': self.repeat_cutoff,
            'recalibrations': self.recalibrations,
            'expected_count_rate': self.expected_count_rate,
            'rct_cutoff': self.rct_cutoff,
            'events': [str(event) for event in self.events[-20:]],
        }
