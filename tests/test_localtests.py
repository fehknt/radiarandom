"""The portable battery, and the statistics of judging its own self-test.

These run on synthetic data only -- no detector, no dieharder.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'validation'))

import localtests as lt  # noqa: E402


# ------------------------------------------------------- special functions


def test_chi2_p_value_matches_known_values():
    # chi2 = dof is the median-ish region; these are standard table values.
    assert lt.chi2_p_value(0.0, 4) == pytest.approx(1.0)
    assert lt.chi2_p_value(9.488, 4) == pytest.approx(0.05, abs=1e-3)
    assert lt.chi2_p_value(13.277, 4) == pytest.approx(0.01, abs=1e-3)


def test_chi2_p_value_is_monotonic():
    values = [lt.chi2_p_value(x, 8) for x in (1, 4, 8, 16, 32)]
    assert values == sorted(values, reverse=True)


def test_to_bits_is_msb_first():
    assert lt.to_bits(b'\x80') == [1, 0, 0, 0, 0, 0, 0, 0]
    assert lt.to_bits(b'\x01') == [0, 0, 0, 0, 0, 0, 0, 1]


# ------------------------------------------------------------- the battery


def test_good_data_mostly_passes():
    report = lt.run_battery(os.urandom(1 << 17), quick=True)
    assert report['tests_run'] >= 8
    # A single failure is expected noise at alpha=0.01 across ~11 tests.
    assert len(report['failures']) <= lt.SELF_TEST_TOLERATED_FAILURES, report['failures']


def test_a_counter_is_rejected():
    report = lt.run_battery(bytes(i & 0xFF for i in range(1 << 17)), quick=True)
    assert len(report['failures']) >= lt.SELF_TEST_MIN_BAD_FAILURES
    assert not report['passed']


def test_a_stuck_bit_is_caught_by_the_per_position_test():
    stuck = bytes(b | 1 for b in os.urandom(1 << 17))
    report = lt.run_battery(stuck, quick=True)
    assert 'bit_position_bias' in report['failures']


def test_a_constant_stream_is_rejected():
    report = lt.run_battery(b'\x00' * (1 << 17), quick=True)
    assert not report['passed']


def test_short_input_skips_rather_than_crashing():
    report = lt.run_battery(os.urandom(256), quick=True)
    assert report['tests_skipped'], 'undersized tests should skip, not fail'


def test_one_sided_tests_are_not_flagged_for_high_p_values():
    """bit_position_bias is a Bonferroni minimum; it saturates at 1.0."""
    assert 'bit_position_bias' in lt.ONE_SIDED_TESTS
    report = lt.run_battery(os.urandom(1 << 17), quick=True)
    detail = report['results']['bit_position_bias']
    if detail['p_value'] is not None and detail['p_value'] > 0.99:
        assert 'bit_position_bias' not in report['weak']


# ------------------------------------------- the self-test's own statistics


def test_self_test_tolerates_the_noise_it_will_actually_see():
    """Requiring a clean sweep is a multiple-comparisons bug.

    With `n` graded tests at alpha, the chance of at least one false failure on
    genuinely random data is 1 - (1-alpha)^n. At n=13 that is 12.2%, which made
    roughly one CI run in eight fail on good entropy. The tolerance must be
    high enough that the false-alarm rate is small, and the retry squares it.
    """
    alpha, n = lt.ALPHA, 13
    p_at_least_one = 1 - (1 - alpha) ** n
    p_above_tolerance = 1 - sum(
        math.comb(n, k) * alpha ** k * (1 - alpha) ** (n - k)
        for k in range(lt.SELF_TEST_TOLERATED_FAILURES + 1))

    assert p_at_least_one > 0.10, 'the bug this guards against was real'
    assert p_above_tolerance < 0.01, 'single-run false alarms must be rare'
    assert p_above_tolerance ** 2 < 1e-4, 'and vanishing after the retry'


def test_bad_streams_clear_the_rejection_threshold_by_a_margin():
    """The counter fails more tests than the threshold demands.

    Quick mode skips binary_matrix_rank and approximate_entropy, which are two
    of the strongest detectors here, so the margin is smaller than the full
    battery gives -- 4 failures against a threshold of 3. The full battery
    reaches 6.
    """
    report = lt.run_battery(bytes(i & 0xFF for i in range(1 << 17)), quick=True)
    assert len(report['failures']) > lt.SELF_TEST_MIN_BAD_FAILURES


def test_tolerance_does_not_swallow_a_genuinely_bad_stream():
    """Whatever tolerance we allow must stay well below what bad data produces."""
    assert lt.SELF_TEST_TOLERATED_FAILURES < lt.SELF_TEST_MIN_BAD_FAILURES
