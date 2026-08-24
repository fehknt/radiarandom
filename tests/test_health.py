"""Health tests: they must fire on real faults and stay quiet on good data.

The important cases here are the ones that motivated replacing the textbook
SP 800-90B continuous tests: a healthy detector under a check source produces
big sorted batches, and any order-sensitive test would fail on those constantly.
"""

from __future__ import annotations

import math
import random

import pytest

from radiarandom import health
from radiarandom.device import Batch


def batch(seq, channels, monotonic=None, device_seconds=None) -> Batch:
    return Batch(
        seq=seq,
        host_time=1000.0 + seq,
        host_monotonic=0.5 * seq if monotonic is None else monotonic,
        device_seconds=seq if device_seconds is None else device_seconds,
        channels=tuple(channels),
        count=len(channels),
        cumulative_total=seq * 3,
    )


def healthy_batches(n_batches, photons_per_batch, seed=1, n_channels=1024):
    """Sorted batches drawn from a broad distribution, like the real device."""
    rng = random.Random(seed)
    weights = [math.exp(-c / 90.0) + 1e-4 for c in range(n_channels)]
    population = list(range(n_channels))
    for seq in range(1, n_batches + 1):
        channels = sorted(rng.choices(population, weights=weights, k=photons_per_batch))
        yield batch(seq, channels)


# --------------------------------------------------------------- cutoffs


def test_repetition_cutoff_matches_the_standards_formula():
    assert health.repetition_count_cutoff(1.0) == 21
    assert health.repetition_count_cutoff(4.0) == 6
    assert health.repetition_count_cutoff(20.0) == 2


def test_proportion_cutoff_is_in_range_and_moves_the_right_way():
    low = health.proportion_cutoff(2.0)
    high = health.proportion_cutoff(6.0)
    assert 1 < high < low <= health.PROPORTION_WINDOW + 1


def test_proportion_cutoff_respects_the_union_bound():
    """Per-channel false-positive rate must be at most alpha / n_channels."""
    n_channels = 1024
    for h in (2.0, 4.0, 6.0):
        cutoff = health.proportion_cutoff(h, n_channels=n_channels)
        p = 2.0 ** -h
        tail = 1.0 - health._binomial_cdf(cutoff - 1, health.PROPORTION_WINDOW, p)
        assert tail <= health.ALPHA / n_channels


def test_binomial_cdf_edges():
    assert health._binomial_cdf(-1, 10, 0.5) == 0.0
    assert health._binomial_cdf(10, 10, 0.5) == 1.0
    assert health._binomial_cdf(5, 10, 0.5) == pytest.approx(0.623046875, rel=1e-9)


# ------------------------------------------------- healthy data passes


def test_healthy_low_rate_stream_passes():
    monitor = health.HealthMonitor(h_per_photon=4.0, startup_samples=64)
    for b in healthy_batches(400, 2):
        monitor.observe(b)
    assert not monitor.failed
    assert monitor.started


def test_healthy_high_rate_stream_passes():
    """The regression that motivated the redesign.

    With a check source a batch holds hundreds of photons, arriving as a sorted
    run. An order-sensitive repetition test fires immediately on this; the
    order-free proportion test must not.
    """
    monitor = health.HealthMonitor(h_per_photon=4.0, startup_samples=0)
    for b in healthy_batches(200, 500, seed=7):
        monitor.observe(b)
        assert not monitor.failed, monitor.failure_reason


def test_sorted_duplicates_within_one_batch_are_not_a_failure():
    """Two photons in the same channel are adjacent purely because of sorting."""
    monitor = health.HealthMonitor(h_per_photon=4.0, startup_samples=0)
    monitor.observe(batch(1, [5, 5, 5, 5, 5, 5, 5, 5] + list(range(100, 900))))
    assert not monitor.failed


# ------------------------------------------------------- faults detected


def test_proportion_test_fires_on_a_stuck_channel():
    """Batch sizes vary so the repetition test cannot mask the result."""
    monitor = health.HealthMonitor(h_per_photon=4.0, startup_samples=0)
    rng = random.Random(17)
    seq = 0
    while not monitor.failed and seq < 200:
        seq += 1
        monitor.observe(batch(seq, [512] * rng.randint(30, 70)))
    assert monitor.failed
    assert 'proportion' in (monitor.failure_reason or ''), monitor.failure_reason


def test_proportion_test_fires_on_a_dominant_but_not_exclusive_channel():
    """Half the photons in one channel is not a stuck ADC, but is still fatal."""
    monitor = health.HealthMonitor(h_per_photon=4.0, startup_samples=0)
    rng = random.Random(3)
    seq = 0
    while not monitor.failed and seq < 200:
        seq += 1
        channels = sorted([100] * 32 + [rng.randrange(1024) for _ in range(32)])
        monitor.observe(batch(seq, channels))
    assert monitor.failed
    assert 'proportion' in (monitor.failure_reason or '')


def test_repetition_test_catches_a_replaying_device():
    monitor = health.HealthMonitor(h_per_photon=4.0, startup_samples=0)
    frozen = tuple(sorted(random.Random(5).sample(range(1024), 40)))
    for seq in range(1, health.REPEAT_CUTOFF + 3):
        monitor.observe(batch(seq, frozen))
    assert monitor.failed
    assert 'repetition' in (monitor.failure_reason or '')


def test_repetition_test_ignores_empty_batches():
    """Empty batches are normal at 5 Hz polling against a 2 Hz device."""
    monitor = health.HealthMonitor(h_per_photon=4.0, startup_samples=0,
                                   expected_count_rate=4.4)
    monitor.observe(batch(1, [10, 20], monotonic=0.0))
    for seq in range(2, 20):
        monitor.observe(batch(seq, [], monotonic=seq * 0.2))
    assert not monitor.failed


def test_stall_detected_when_the_detector_goes_silent():
    monitor = health.HealthMonitor(h_per_photon=4.0, expected_count_rate=4.4,
                                   startup_samples=0, stall_grace_s=5.0)
    monitor.observe(batch(1, [10, 20, 30], monotonic=0.0))
    monitor.observe(batch(2, [11, 21], monotonic=1.0))
    assert not monitor.failed
    monitor.observe(batch(3, [], monotonic=500.0))
    assert monitor.failed
    assert 'stall' in (monitor.failure_reason or '')


def test_brief_silence_is_tolerated():
    monitor = health.HealthMonitor(h_per_photon=4.0, expected_count_rate=4.4,
                                   startup_samples=0, stall_grace_s=30.0)
    monitor.observe(batch(1, [10, 20], monotonic=0.0))
    monitor.observe(batch(2, [], monotonic=10.0))
    assert not monitor.failed


def test_out_of_range_channel_is_a_failure():
    monitor = health.HealthMonitor(h_per_photon=4.0, startup_samples=0, n_symbols=1024)
    monitor.observe(batch(1, [5000]))
    assert monitor.failed
    assert 'range' in (monitor.failure_reason or '')


def test_shape_test_calibrates_on_the_session_not_the_lifetime_spectrum():
    """A device whose lifetime spectrum differs from today must not warn forever.

    This is the regression from the first live run: the reference was the
    device's 88-day accumulated spectrum, today's background looked nothing
    like it, and the monitor warned every four minutes with nothing wrong.
    """
    lifetime = [1000] * 1024          # nothing like today's conditions
    monitor = health.HealthMonitor(h_per_photon=4.0, startup_samples=0,
                                   reference_spectrum=lifetime,
                                   calibrate_photons=2048)
    rng = random.Random(31)
    warnings = []
    for seq in range(1, 1500):
        channels = sorted(int(rng.expovariate(1 / 90.0)) % 1024
                          for _ in range(rng.randint(20, 40)))
        for event in monitor.observe(batch(seq, channels)):
            if event.kind == 'shape' and event.severity == 'warning':
                warnings.append(event)
    assert not monitor.failed
    # At most the single one-shot note about differing from the lifetime
    # spectrum; never a repeating drift warning on a steady detector.
    assert len(warnings) <= 1, [str(w) for w in warnings]


def test_shape_test_warns_when_the_spectrum_drifts_mid_run():
    """Calibrate on a healthy spectrum, then move it, and expect a warning."""
    monitor = health.HealthMonitor(h_per_photon=4.0, startup_samples=0,
                                   calibrate_photons=2048)
    rng = random.Random(37)
    for seq in range(1, 200):
        channels = sorted(int(rng.expovariate(1 / 90.0)) % 1024
                          for _ in range(30))
        monitor.observe(batch(seq, channels))
    assert monitor._reference_shape is not None

    warned = False
    seq = 200
    while seq < 2000 and not warned and not monitor.failed:
        seq += 1
        # Gain has shifted: the same distribution, moved 300 channels up.
        # A shift rather than a collapse, so the proportion test stays quiet
        # and the shape test is the one under examination.
        channels = sorted(300 + int(rng.expovariate(1 / 90.0)) % 700
                          for _ in range(30))
        events = monitor.observe(batch(seq, channels))
        warned = any(event.kind == 'shape' and event.severity == 'warning'
                     for event in events)
    assert warned, monitor.failure_reason


# ------------------------------------------------------------ mechanics


def test_startup_gate_counts_photons():
    monitor = health.HealthMonitor(h_per_photon=4.0, startup_samples=100)
    assert not monitor.started
    for b in healthy_batches(10, 5, seed=11):
        monitor.observe(b)
    assert not monitor.started
    for b in healthy_batches(40, 5, seed=12):
        monitor.observe(b)
    assert monitor.started


def test_startup_disabled_when_zero():
    monitor = health.HealthMonitor(h_per_photon=4.0, startup_samples=0)
    assert monitor.started


def test_failure_latches_and_keeps_the_first_reason():
    monitor = health.HealthMonitor(h_per_photon=4.0, startup_samples=0)
    monitor.observe(batch(1, [5000]))
    assert monitor.failed
    reason = monitor.failure_reason
    monitor.observe(batch(2, [100, 200, 300]))
    assert monitor.failure_reason == reason


def test_observe_is_a_no_op_after_failure():
    monitor = health.HealthMonitor(h_per_photon=4.0, startup_samples=0)
    monitor.observe(batch(1, [5000]))
    assert monitor.observe(batch(2, [1, 2, 3])) == []


def test_status_reports_the_configured_cutoffs():
    monitor = health.HealthMonitor(h_per_photon=4.0, startup_samples=10)
    status = monitor.status()
    assert status['proportion_cutoff'] == monitor.proportion_cutoff
    assert status['proportion_window'] == health.PROPORTION_WINDOW
    assert status['repeat_cutoff'] == health.REPEAT_CUTOFF
    assert status['healthy'] is True


def test_rate_test_does_not_cry_wolf_on_a_steady_source():
    """A steady detector must not accumulate rate warnings over a long run.

    Regression: comparing the cumulative average against a baseline drawn from
    that same average made sigma shrink without bound, so a 0.45% offset was
    reported as +12.9 sigma on every batch. The window is now bounded.
    """
    rng = random.Random(41)
    rate = 100.0
    monitor = health.HealthMonitor(h_per_photon=4.0, startup_samples=0,
                                   expected_count_rate=rate)
    warnings = []
    now = 0.0
    for seq in range(1, 4000):
        now += 0.5
        n = 50
        channels = sorted(int(rng.expovariate(1 / 90.0)) % 1024 for _ in range(n))
        for event in monitor.observe(batch(seq, channels, monotonic=now)):
            if event.kind == 'rate':
                warnings.append(str(event))
    assert not monitor.failed
    assert not warnings, warnings[:3]


def test_rate_test_flags_a_genuine_collapse():
    """Half the expected rate over a full window must warn."""
    monitor = health.HealthMonitor(h_per_photon=4.0, startup_samples=0,
                                   expected_count_rate=100.0)
    rng = random.Random(43)
    warned = False
    now = 0.0
    for seq in range(1, 1000):
        now += 0.5
        channels = sorted(int(rng.expovariate(1 / 90.0)) % 1024 for _ in range(10))
        for event in monitor.observe(batch(seq, channels, monotonic=now)):
            if event.kind == 'rate':
                warned = True
        if warned:
            break
    assert warned


def test_a_check_source_peak_does_not_halt_the_generator():
    """A source putting 14% of counts in one channel must be tolerated.

    Regression from live hardware: an Am-241 source parked on the detector put
    71 of every 512 photons into channel 25, against an assumed-spectrum cutoff
    of exactly 71, and the capture halted after 100 seconds. Adding a check
    source is the recommended way to raise the entropy rate, so refusing to run
    with one is backwards -- the Poisson budget already credits less per photon
    for the narrower spectrum.
    """
    rng = random.Random(51)
    monitor = health.HealthMonitor(h_per_photon=4.0, startup_samples=1024)

    def source_like_batch(seq):
        channels = []
        for _ in range(40):
            if rng.random() < 0.14:
                channels.append(25)                       # the source peak
            else:
                channels.append(int(rng.expovariate(1 / 90.0)) % 1024)
        return batch(seq, sorted(channels))

    for seq in range(1, 400):
        monitor.observe(source_like_batch(seq))
        assert not monitor.failed, monitor.failure_reason
    assert monitor.started
    assert 'calibrated' in monitor.proportion_cutoff_origin


def test_a_stuck_channel_is_still_caught_after_calibration():
    """Widening the cutoff must not disarm the test."""
    rng = random.Random(53)
    monitor = health.HealthMonitor(h_per_photon=4.0, startup_samples=256)
    for seq in range(1, 40):
        channels = sorted(
            25 if rng.random() < 0.14 else int(rng.expovariate(1 / 90.0)) % 1024
            for _ in range(40))
        monitor.observe(batch(seq, channels))
    assert monitor.started and not monitor.failed

    seq = 100
    while not monitor.failed and seq < 400:
        seq += 1
        monitor.observe(batch(seq, [900] * rng.randint(30, 70)))
    assert monitor.failed
    assert 'proportion' in (monitor.failure_reason or ''), monitor.failure_reason


def test_calibration_phase_still_catches_a_totally_stuck_detector():
    """Even before calibration, one channel taking everything must fail."""
    rng = random.Random(57)
    monitor = health.HealthMonitor(h_per_photon=4.0, startup_samples=4096)
    seq = 0
    while not monitor.failed and seq < 400:
        seq += 1
        monitor.observe(batch(seq, [77] * rng.randint(30, 70)))
    assert monitor.failed
    assert 'proportion' in (monitor.failure_reason or ''), monitor.failure_reason
