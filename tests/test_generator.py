"""End-to-end generator behaviour against a simulated detector."""

from __future__ import annotations

import pytest

from radiarandom import entropy
from radiarandom.conditioner import BLOCK_BYTES
from radiarandom.generator import Generator, GeneratorError
from radiarandom.health import HealthFailure

from conftest import FakeSource


def build(source: FakeSource, **kwargs) -> Generator:
    kwargs.setdefault('startup_samples', 64)
    kwargs.setdefault('reference_spectrum', source.reference_spectrum())
    return Generator(source, **kwargs)


# ------------------------------------------------------------- gating


def test_startup_must_complete_before_output(fast_source):
    generator = build(fast_source)
    with pytest.raises(GeneratorError, match='start-up'):
        generator.physical_block()


def test_no_entropy_is_credited_before_the_rate_is_measured(fast_source):
    """The bootstrap assessment must hand out nothing."""
    generator = build(fast_source)
    generator.pump_once()
    assert generator.pool.entropy_bits == 0.0
    assert generator._current.count_rate == 0.0


# ------------------------------------------------------ physical output


def test_physical_block_is_full_size_and_varies(fast_source):
    generator = build(fast_source)
    generator.wait_for_startup()
    blocks = [generator.physical_block() for _ in range(6)]
    assert all(len(block) == BLOCK_BYTES for block in blocks)
    assert len(set(blocks)) == len(blocks)


def test_physical_bytes_returns_exact_length(fast_source):
    generator = build(fast_source)
    generator.wait_for_startup()
    for n in (1, 31, 32, 33, 100):
        assert len(generator.physical_bytes(n)) == n


def test_entropy_is_actually_consumed(fast_source):
    generator = build(fast_source)
    generator.wait_for_startup()
    before = generator.pool.stats()['blocks_released']
    generator.physical_block()
    assert generator.pool.stats()['blocks_released'] == before + 1


def test_pool_never_goes_negative(fast_source):
    generator = build(fast_source)
    generator.wait_for_startup()
    for _ in range(300):
        generator.pump_once()
        while generator.pool.ready():
            generator.physical_block()
        assert generator.pool.entropy_bits >= 0


# ---------------------------------------------------------- DRBG output


def test_drbg_mode_produces_requested_length(fast_source):
    generator = build(fast_source)
    generator.wait_for_startup()
    for n in (1, 1000, 100000):
        assert len(generator.read(n)) == n


def test_drbg_output_is_not_constant(fast_source):
    generator = build(fast_source)
    generator.wait_for_startup()
    data = generator.read(1 << 16)
    assert len(set(data)) == 256
    ones = sum(bin(byte).count('1') for byte in data)
    assert abs(ones / (len(data) * 8) - 0.5) < 0.02


def test_two_generators_do_not_produce_the_same_stream():
    a = build(FakeSource(seed=1, count_rate=2000.0))
    b = build(FakeSource(seed=2, count_rate=2000.0))
    a.wait_for_startup()
    b.wait_for_startup()
    assert a.read(256) != b.read(256)


def test_drbg_does_not_stall_waiting_for_a_slow_detector():
    """A megabyte must come out without blocking on a 4 counts/s source.

    The DRBG's mandatory reseed interval is deliberately loose precisely so
    this works; a tight interval would stall for ~30 s every few tens of MB.
    """
    source = FakeSource(count_rate=4.4)
    generator = build(source, startup_samples=8)
    generator.wait_for_startup()
    assert len(generator.read(1 << 20)) == 1 << 20


# ----------------------------------------------------- entropy tracking


def test_assessment_tracks_the_measured_rate(fast_source):
    generator = build(fast_source)
    generator.wait_for_startup()
    for _ in range(200):
        generator.pump_once()
    assert generator._current.count_rate > 0
    assert generator.entropy_rate_bits_per_s > 0


def test_faster_source_yields_more_entropy_per_second():
    slow = build(FakeSource(seed=4, count_rate=20.0), startup_samples=8)
    fast = build(FakeSource(seed=4, count_rate=2000.0), startup_samples=8)
    for generator in (slow, fast):
        generator.wait_for_startup()
        for _ in range(200):
            generator.pump_once()
    assert fast.entropy_rate_bits_per_s > slow.entropy_rate_bits_per_s


def test_credited_rate_never_exceeds_the_reference_based_estimate():
    """The live spectrum may only ever lower the claim, never raise it.

    This is the invariant that makes a degrading detector safe: whichever of
    the reference and live distributions is more pessimistic is the one that
    gets banked, so a spectrum that narrows earns less without anyone having
    to notice.
    """
    source = FakeSource(count_rate=2000.0)
    generator = build(source, startup_samples=8)
    generator.wait_for_startup()
    for _ in range(200):
        generator.pump_once()

    reference_only = entropy.Assessment(
        channel_probs=generator.reference_probs,
        count_rate=generator._current.count_rate,
        safety_factor=generator.safety_factor,
        origin='reference',
    )
    assert (generator._current.bits_per_second()
            <= reference_only.bits_per_second() + 1e-9)


def test_a_narrower_spectrum_is_worth_less():
    """Same photon rate, fewer channels in play, less banked entropy."""
    broad = entropy.uniform_channel_probs(1024)
    narrow = entropy.normalise([1.0] * 200 + [0.0] * 824)
    rate = 2000.0
    assert (entropy.window_min_entropy(rate, narrow, 1.0)
            < entropy.window_min_entropy(rate, broad, 1.0))


def test_disabling_live_estimate_uses_only_the_reference_spectrum():
    source = FakeSource(count_rate=2000.0)
    generator = build(source, use_live_estimate=False, startup_samples=8)
    generator.wait_for_startup()
    for _ in range(200):
        generator.pump_once()
    assert 'reference' in generator._current.origin


# ------------------------------------------------------------- failures


def test_health_failure_stops_output(fast_source):
    generator = build(fast_source)
    generator.wait_for_startup()
    generator.read(64)
    fast_source.stuck_channel = 512
    with pytest.raises(HealthFailure):
        for _ in range(500):
            generator.pump_once()
            generator.read(64)


def test_silent_detector_is_caught(fast_source):
    generator = build(fast_source)
    generator.wait_for_startup()
    fast_source.silent = True
    with pytest.raises(HealthFailure, match='stall'):
        for _ in range(100000):
            generator.pump_once()


def test_silent_detector_earns_no_entropy(fast_source):
    generator = build(fast_source)
    generator.wait_for_startup()
    for _ in range(200):
        generator.pump_once()
    while generator.pool.ready():
        generator.physical_block()
    fast_source.silent = True
    before = generator.pool.entropy_bits
    try:
        for _ in range(20):
            generator.pump_once()
    except HealthFailure:
        pass
    # The device clock still ticks, so a little credit is possible, but a dead
    # detector must never fill a block on its own.
    assert generator.pool.entropy_bits - before < 320


def test_device_reset_does_not_fabricate_photons(fast_source):
    generator = build(fast_source)
    generator.wait_for_startup()
    before = generator._photons_seen
    fast_source.reset_at = fast_source._seq + 1
    generator.pump_once()
    assert generator._photons_seen == before


# ---------------------------------------------------------------- misc


def test_stats_are_coherent(fast_source):
    generator = build(fast_source)
    generator.wait_for_startup()
    generator.read(4096)
    stats = generator.stats()
    assert stats['photons'] > 0
    assert stats['batches'] > 0
    assert stats['health']['healthy'] is True
    assert stats['drbg']['bytes_generated'] >= 0
    assert stats['bits_credited'] >= 0
    assert 'bits/s' in stats['assessment']


def test_background_pump_fills_the_pool(fast_source):
    generator = build(fast_source)
    generator.wait_for_startup()
    generator.run_background()
    try:
        assert len(generator.read(1 << 20)) == 1 << 20
    finally:
        generator.stop()


def test_stop_is_idempotent(fast_source):
    generator = build(fast_source)
    generator.stop()
    generator.stop()
    assert generator.stopped


def test_rate_estimate_matches_the_source_when_polls_outpace_the_device():
    """Polling faster than the device refreshes must not deflate the rate.

    Regression from live hardware: the rate estimator shared its clock with the
    credit accounting, which only advances when the device demonstrably moved.
    Every poll that found nothing new re-counted the same interval, inflating
    the denominator. The generator reported 8.97 photons/s while the device was
    actually delivering 16.21.
    """
    true_rate = 40.0
    source = FakeSource(seed=8, count_rate=true_rate, refresh_s=0.5)
    generator = build(source, startup_samples=8)
    generator.wait_for_startup()
    for _ in range(2000):
        generator.pump_once()

    measured = generator.count_rate
    assert measured is not None
    assert 0.85 * true_rate < measured < 1.15 * true_rate, measured


def test_rate_baseline_is_not_latched_from_the_first_few_seconds():
    """A baseline taken during ramp-up makes the rate test warn about nothing."""
    source = FakeSource(seed=9, count_rate=40.0, refresh_s=0.5)
    generator = build(source, startup_samples=8)
    generator.wait_for_startup()
    for _ in range(10):
        generator.pump_once()
    # Ten batches is five simulated seconds: far too little to baseline on.
    assert generator.monitor.expected_count_rate is None
