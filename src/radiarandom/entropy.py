"""Min-entropy accounting for the RadiaCode noise source.

The model
--------
Radioactive decay is a Poisson process, and each detected photon lands in
pulse-height channel ``i`` independently with probability ``p_i``. By the
Poisson splitting theorem the per-channel counts over a fixed time window are
therefore **independent** Poisson variables:

    N_i ~ Poisson(mu_i),    mu_i = rate * p_i * window

That is the whole model, and it is exact rather than an approximation. Because
the ``N_i`` are independent, their min-entropies add, so the min-entropy of one
observation window is

    H_inf(window) = sum_i H_inf(Poisson(mu_i))

with ``H_inf(Poisson(mu)) = -log2(P(N = floor(mu)))``, the probability of the
distribution's mode.

Why not "bits per photon"
-------------------------
An earlier version of this module conditioned on the photon count ``n`` and
banked ``n * H_channel - log2(n!)`` -- the per-photon channel entropy, less the
ordering information the device throws away when it hands us a histogram
instead of a sequence. That is wrong in both directions and was caught by the
test suite:

* **It over-credits at low rates.** Min-entropy is set by the single most
  likely outcome. At 4.4 counts/s over a 0.5 s window the most likely outcome
  is *no photons at all*, with probability ``exp(-2.2) = 0.111``, so the window
  is worth 3.17 bits -- not the 9.6 bits the per-photon formula claimed. You
  may not condition on ``n`` and then decline to pay for it.

* **It collapses at high rates.** With a check source at 200 counts/s a window
  holds ~100 photons, and ``100 * 3.6 - log2(100!)`` is negative, i.e. the
  formula credits *zero* entropy exactly when the detector is supplying the
  most.

The Poisson model has neither defect. At low rates it reduces to
``log2(e) * rate * window`` = 1.443 bits per photon; at high rates the busy
channels saturate individually and the total keeps growing, sub-linearly and
correctly.

Measured baseline
-----------------
On RC-103-013128 the lifetime spectrum (49.4 M counts over 88 days) has
``p_max = 0.0274`` at channel 1, i.e. 5.19 bits/photon of channel min-entropy
and 7.14 bits of Shannon entropy.

At the observed 4.4 counts/s the Poisson model yields 4.4 x 1.443 = 6.35 bits/s
before the safety factor, and **5.7 bits/s (0.71 bytes/s) after it** -- which is
the figure actually banked. A 256-bit block costs 320 banked bits, so roughly
56 seconds each. Adding a check source raises this close to proportionally
until the busiest channels start to saturate.

An independent cross-check: the SP 800-90B estimators in ``validation/`` put
the *raw channel stream* at 3.69 bits/photon (on a short 7 k-photon capture,
where the compression estimator binds and is known to read low). We bank
1.443 x 0.9 = 1.30 bits/photon, comfortably below that.

The per-photon channel figure is still used, but only to parameterise the
*health tests* (which need a ``p_max`` to set their cutoffs). It no longer
feeds the entropy budget.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Iterable, Optional, Sequence

LOG2_E = 1.4426950408889634

#: Conservative per-photon channel min-entropy, in bits, used to size the
#: health-test cutoffs. The measured value is 5.19; we assume 4.0 so that
#: ordinary gain drift does not make the cutoffs too tight.
DEFAULT_H_CHANNEL = 4.0

#: Multiplied into the banked entropy. Covers model error -- detector dead
#: time, afterpulsing, and any correlation the independence assumption misses.
DEFAULT_SAFETY_FACTOR = 0.9

#: Photons the live estimator wants before its numbers are trusted.
MCV_MIN_SAMPLES = 2000

#: Effective memory of the live spectrum estimate, in photons.
#:
#: Without forgetting, the estimate is a cumulative average over the whole run:
#: after an hour a change in the environment barely moves it, so the entropy
#: budget would keep pricing the detector as it was rather than as it is.
#: Exponential decay keeps it tracking current conditions. Roughly 20 minutes
#: at 16 counts/s.
MCV_HALF_LIFE_SAMPLES = 20000

#: Longest gap we will pay for in one go. Guards against crediting a stall.
MAX_CREDIT_GAP_S = 2.0


def log2_factorial(n: int) -> float:
    """``log2(n!)`` via lgamma, so large ``n`` stays cheap."""
    if n < 2:
        return 0.0
    return math.lgamma(n + 1) / math.log(2.0)


def poisson_min_entropy(mu: float) -> float:
    """``-log2(P(N = mode))`` for ``N ~ Poisson(mu)``, in bits.

    The mode of a Poisson distribution is ``floor(mu)``, so this is the
    min-entropy of a single channel's count over one observation window.

    For small ``mu`` the mode is 0 and this reduces to ``mu * log2(e)``; for
    large ``mu`` it grows like ``0.5 * log2(2*pi*e*mu)``, which is why piling
    more photons into one channel yields diminishing returns.
    """
    if mu <= 0.0:
        return 0.0
    if mu < 1e-9:
        return mu * LOG2_E
    mode = math.floor(mu)
    # log P(N = mode) = -mu + mode*ln(mu) - ln(mode!)
    log_p = -mu + mode * math.log(mu) - math.lgamma(mode + 1.0)
    return max(0.0, -log_p * LOG2_E)


def normalise(counts: Sequence[float]) -> list[float]:
    total = float(sum(counts))
    if total <= 0:
        return [0.0] * len(counts)
    return [value / total for value in counts]


def window_min_entropy(
    count_rate: float,
    channel_probs: Sequence[float],
    window_seconds: float,
) -> float:
    """Min-entropy of one observation window, in bits.

    Sums the independent per-channel Poisson min-entropies. Exact under the
    model; no approximation beyond the model itself.
    """
    if count_rate <= 0 or window_seconds <= 0:
        return 0.0
    scale = count_rate * window_seconds
    return sum(poisson_min_entropy(scale * p) for p in channel_probs if p > 0.0)


@dataclasses.dataclass
class Assessment:
    """How much min-entropy the detector is supplying, and on what basis.

    Attributes:
        channel_probs: the pulse-height distribution used for the model.
        count_rate: photons per second, ideally a lower confidence bound.
        safety_factor: multiplied into every credited amount.
        h_channel: per-photon channel min-entropy, for health-test cutoffs.
        origin: human-readable provenance, shown by ``radiarandom info``.
    """

    channel_probs: Sequence[float]
    count_rate: float
    safety_factor: float = DEFAULT_SAFETY_FACTOR
    h_channel: float = DEFAULT_H_CHANNEL
    origin: str = 'default'
    _bits_per_second: Optional[float] = dataclasses.field(
        default=None, repr=False, init=False, compare=False)

    def bits_per_second(self) -> float:
        """Banked bits per second at the current rate, after the safety factor.

        Computed over a one-second window and prorated linearly for shorter
        intervals. Since ``H_inf`` is concave in the window length, prorating
        down from one second under-credits rather than over-credits.
        """
        if self._bits_per_second is None:
            raw = window_min_entropy(self.count_rate, self.channel_probs, 1.0)
            self._bits_per_second = max(0.0, raw * self.safety_factor)
        return self._bits_per_second

    def credit(self, elapsed_seconds: float) -> float:
        """Bits to bank for ``elapsed_seconds`` of confirmed live detector."""
        if elapsed_seconds <= 0:
            return 0.0
        return self.bits_per_second() * min(elapsed_seconds, MAX_CREDIT_GAP_S)

    def bytes_per_second(self) -> float:
        return self.bits_per_second() / 8.0

    def describe(self) -> str:
        return (
            f'{self.bits_per_second():.2f} bits/s at {self.count_rate:.2f} '
            f'photons/s (Poisson model over {len(self.channel_probs)} channels, '
            f'safety {self.safety_factor:.2f}, source: {self.origin})'
        )

    def with_rate(self, count_rate: float, origin: Optional[str] = None) -> Assessment:
        return Assessment(
            channel_probs=self.channel_probs,
            count_rate=count_rate,
            safety_factor=self.safety_factor,
            h_channel=self.h_channel,
            origin=origin or self.origin,
        )

    def with_spectrum(self, counts: Sequence[float], origin: Optional[str] = None) -> Assessment:
        probs = normalise(counts)
        return Assessment(
            channel_probs=probs,
            count_rate=self.count_rate,
            safety_factor=self.safety_factor,
            h_channel=min_entropy_of_histogram(counts) if sum(counts) else self.h_channel,
            origin=origin or self.origin,
        )


def uniform_channel_probs(n_channels: int = 1024) -> list[float]:
    return [1.0 / n_channels] * n_channels


def default_assessment(
    channel_probs: Optional[Sequence[float]] = None,
    count_rate: float = 4.4,
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
    h_channel: float = DEFAULT_H_CHANNEL,
) -> Assessment:
    """A starting assessment, before the device has been measured."""
    return Assessment(
        channel_probs=list(channel_probs) if channel_probs else uniform_channel_probs(),
        count_rate=count_rate,
        safety_factor=safety_factor,
        h_channel=h_channel,
        origin='default',
    )


class RateEstimator:
    """Running count-rate estimate with a Poisson lower confidence bound.

    The budget scales with the rate, so an over-estimate would over-credit.
    We therefore bank against the *lower* bound: with ``k`` photons observed
    over ``t`` seconds the rate is at least about
    ``(k - z*sqrt(k)) / t`` at the chosen confidence.
    """

    Z = 2.576  # one-sided 99%

    def __init__(self) -> None:
        self.photons = 0
        self.seconds = 0.0

    def update(self, photons: int, elapsed: float) -> None:
        self.photons += photons
        self.seconds += max(0.0, elapsed)

    @property
    def point_estimate(self) -> Optional[float]:
        if self.seconds < 1.0:
            return None
        return self.photons / self.seconds

    def lower_bound(self) -> Optional[float]:
        """Conservative rate, or None until there is enough data."""
        if self.seconds < 5.0 or self.photons < 20:
            return None
        k = float(self.photons)
        lower = (k - self.Z * math.sqrt(k)) / self.seconds
        return max(0.0, lower)


class MostCommonValue:
    """NIST SP 800-90B section 6.3.1 estimator, applied to channel values.

    Used two ways: as a live check that the detector's pulse-height
    distribution has not degraded, and as the source of the ``p_max`` that
    sizes the health-test cutoffs.

    Counts decay exponentially so the estimate follows the environment. Decay
    shrinks the effective sample size, which widens the confidence margin and
    therefore *lowers* the reported min-entropy -- the conservative direction.
    """

    Z = 2.576

    def __init__(self, n_symbols: int,
                 half_life: Optional[float] = MCV_HALF_LIFE_SAMPLES) -> None:
        self.n_symbols = n_symbols
        self.counts = [0.0] * n_symbols
        self.total = 0.0
        self.half_life = half_life
        self._since_decay = 0.0

    def update(self, symbols: Iterable[int]) -> None:
        symbols = list(symbols)
        n = len(symbols)
        if not n:
            return

        if self.half_life:
            # Decay the *existing* counts before the new ones are added --
            # doing it afterwards ages the newest samples as if they were old,
            # and a single large update would decay itself into irrelevance.
            # Applied in chunks of a tenth of a half-life: 1024 multiplications
            # that often is cheap, and the resulting over-ageing of the newest
            # chunk is bounded at 10%.
            self._since_decay += n
            step = self.half_life / 10.0
            if self._since_decay >= step:
                self.decay(0.5 ** (self._since_decay / self.half_life))
                self._since_decay = 0.0

        counts = self.counts
        for symbol in symbols:
            counts[symbol] += 1.0
        self.total += n

    def decay(self, factor: float) -> None:
        """Scale every count, forgetting the past at the given rate."""
        if factor >= 1.0 or self.total <= 0:
            return
        self.counts = [c * factor for c in self.counts]
        self.total *= factor

    def p_upper(self) -> Optional[float]:
        if self.total < MCV_MIN_SAMPLES:
            return None
        p_hat = max(self.counts) / self.total
        margin = self.Z * math.sqrt(p_hat * (1.0 - p_hat) / (self.total - 1))
        return min(1.0, p_hat + margin)

    def min_entropy(self) -> Optional[float]:
        p_upper = self.p_upper()
        if p_upper is None:
            return None
        return -math.log2(p_upper)

    def probabilities(self) -> Optional[list]:
        """The observed channel distribution, once there is enough of it."""
        if self.total < MCV_MIN_SAMPLES:
            return None
        return [count / self.total for count in self.counts]


def min_entropy_of_histogram(counts: Sequence[float]) -> float:
    """Plain min-entropy of an observed histogram, in bits."""
    total = sum(counts)
    if total <= 0:
        return 0.0
    return -math.log2(max(counts) / total)


def shannon_entropy_of_histogram(counts: Sequence[float]) -> float:
    """Shannon entropy, for reporting only.

    Never used for the budget: Shannon entropy overstates what an adversary
    must guess, which is exactly why min-entropy is the right measure here.
    """
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for count in counts:
        if count:
            p = count / total
            h -= p * math.log2(p)
    return h


def projected_bit_rate(
    count_rate: float,
    assessment: Assessment,
    window_seconds: float = 1.0,
) -> float:
    """Bits per second the detector would supply at ``count_rate``."""
    if count_rate <= 0:
        return 0.0
    raw = window_min_entropy(count_rate, assessment.channel_probs, window_seconds)
    return max(0.0, raw * assessment.safety_factor / window_seconds)
