"""The entropy accounting is the part that must not be wrong.

Over-crediting is the failure mode that matters: it produces keys that look
fine and are not. These tests pin down the arithmetic and, more importantly,
the direction of every approximation.
"""

from __future__ import annotations

import math

import pytest

from radiarandom import entropy


# ------------------------------------------------- Poisson min-entropy


def test_poisson_min_entropy_of_a_dead_channel_is_zero():
    assert entropy.poisson_min_entropy(0.0) == 0.0
    assert entropy.poisson_min_entropy(-1.0) == 0.0


def test_poisson_min_entropy_small_mu_matches_the_closed_form():
    """For mu << 1 the mode is zero, so H = -log2(exp(-mu)) = mu*log2(e)."""
    for mu in (1e-6, 1e-4, 0.01, 0.1):
        assert entropy.poisson_min_entropy(mu) == pytest.approx(mu * entropy.LOG2_E, rel=1e-6)


def test_poisson_min_entropy_matches_a_brute_force_mode_search():
    """Cross-check the analytic mode against an explicit maximisation."""
    for mu in (0.5, 1.0, 2.2, 5.0, 17.3, 100.0):
        best = 0.0
        for k in range(0, int(mu) + 200):
            log_p = -mu + k * math.log(mu) - math.lgamma(k + 1.0)
            best = max(best, math.exp(log_p))
        expected = -math.log2(best)
        assert entropy.poisson_min_entropy(mu) == pytest.approx(expected, rel=1e-9)


def test_poisson_min_entropy_grows_with_mu_but_sublinearly():
    values = [entropy.poisson_min_entropy(mu) for mu in (1, 2, 4, 8, 16, 32)]
    assert values == sorted(values)
    # Doubling mu must not double the entropy once the mode moves off zero.
    assert values[-1] < 2 * values[-2]


# ------------------------------------------------------- window entropy


def test_window_entropy_is_zero_for_a_dead_detector():
    probs = entropy.uniform_channel_probs(64)
    assert entropy.window_min_entropy(0.0, probs, 1.0) == 0.0
    assert entropy.window_min_entropy(10.0, probs, 0.0) == 0.0


def test_window_entropy_at_low_rate_is_log2e_per_photon():
    """With every channel far below mu=1 the total is just log2(e)*rate*t."""
    probs = entropy.uniform_channel_probs(1024)
    rate, window = 4.4, 0.5
    expected = entropy.LOG2_E * rate * window
    assert entropy.window_min_entropy(rate, probs, window) == pytest.approx(expected, rel=1e-6)


def test_window_entropy_beats_the_old_broken_formula_at_high_rates():
    """The superseded n*H - log2(n!) rule credited zero under a check source.

    That is the bug this model exists to fix, so assert the new model keeps
    delivering when the count rate is high.
    """
    probs = entropy.uniform_channel_probs(1024)
    high_rate = 400.0
    n = high_rate * 0.5
    old_formula = n * 3.6 - entropy.log2_factorial(int(n))
    assert old_formula < 0  # the old rule really did collapse
    assert entropy.window_min_entropy(high_rate, probs, 0.5) > 100.0


def test_window_entropy_increases_with_rate():
    probs = entropy.uniform_channel_probs(1024)
    values = [entropy.window_min_entropy(rate, probs, 1.0)
              for rate in (1, 10, 100, 1000)]
    assert values == sorted(values)


def test_concentrated_spectrum_yields_less_entropy_than_a_spread_one():
    """A collapsed spectrum must be worth less. This is the whole safety story."""
    spread = entropy.uniform_channel_probs(1024)
    concentrated = [0.0] * 1024
    concentrated[0] = 1.0
    rate = 100.0
    assert (entropy.window_min_entropy(rate, concentrated, 1.0)
            < entropy.window_min_entropy(rate, spread, 1.0))


def test_a_single_channel_detector_saturates():
    """All counts in one channel: entropy grows only logarithmically."""
    single = [1.0]
    low = entropy.window_min_entropy(10.0, single, 1.0)
    high = entropy.window_min_entropy(10000.0, single, 1.0)
    assert high < 3 * low


# ------------------------------------------------------------ Assessment


def make_assessment(rate=4.4, n_channels=1024, safety=1.0):
    return entropy.Assessment(
        channel_probs=entropy.uniform_channel_probs(n_channels),
        count_rate=rate,
        safety_factor=safety,
        origin='test',
    )


def test_assessment_applies_the_safety_factor():
    full = make_assessment(safety=1.0).bits_per_second()
    half = make_assessment(safety=0.5).bits_per_second()
    assert half == pytest.approx(0.5 * full)


def test_assessment_credit_is_proportional_to_elapsed_time():
    assessment = make_assessment()
    per_second = assessment.bits_per_second()
    assert assessment.credit(1.0) == pytest.approx(per_second)
    assert assessment.credit(0.5) == pytest.approx(0.5 * per_second)
    assert assessment.credit(0.0) == 0.0
    assert assessment.credit(-5.0) == 0.0


def test_assessment_credit_is_capped_so_a_stall_cannot_be_banked():
    """A long gap must not be paid for in full; the detector may have been dead."""
    assessment = make_assessment()
    capped = assessment.credit(1000.0)
    assert capped == pytest.approx(assessment.bits_per_second() * entropy.MAX_CREDIT_GAP_S)


def test_assessment_with_zero_rate_credits_nothing():
    """The bootstrap state must never hand out free entropy."""
    assessment = make_assessment(rate=0.0)
    assert assessment.bits_per_second() == 0.0
    assert assessment.credit(10.0) == 0.0


def test_with_rate_and_with_spectrum_produce_new_assessments():
    base = make_assessment()
    faster = base.with_rate(44.0)
    assert faster.bits_per_second() > base.bits_per_second()
    narrowed = base.with_spectrum([1] + [0] * 1023)
    assert narrowed.bits_per_second() < base.bits_per_second()


def test_describe_mentions_the_rate():
    text = make_assessment(rate=4.4).describe()
    assert '4.4' in text and 'bits/s' in text


# --------------------------------------------------------- rate estimator


def test_rate_estimator_withholds_a_bound_until_it_has_data():
    estimator = entropy.RateEstimator()
    estimator.update(3, 1.0)
    assert estimator.lower_bound() is None


def test_rate_estimator_lower_bound_is_below_the_truth():
    estimator = entropy.RateEstimator()
    true_rate = 10.0
    estimator.update(int(true_rate * 600), 600.0)
    bound = estimator.lower_bound()
    assert bound is not None
    assert 0 < bound < true_rate
    assert bound > true_rate * 0.85  # but not absurdly pessimistic


def test_rate_estimator_point_estimate():
    estimator = entropy.RateEstimator()
    estimator.update(100, 10.0)
    assert estimator.point_estimate == pytest.approx(10.0)


# --------------------------------------------------------------- helpers


def test_log2_factorial_matches_exact_values():
    for n in range(0, 20):
        assert entropy.log2_factorial(n) == pytest.approx(math.log2(math.factorial(n)), rel=1e-12)


def test_min_entropy_of_histogram():
    assert entropy.min_entropy_of_histogram([1, 1, 1, 1]) == pytest.approx(2.0)
    assert entropy.min_entropy_of_histogram([4, 0, 0, 0]) == pytest.approx(0.0)
    assert entropy.min_entropy_of_histogram([]) == 0.0


def test_shannon_is_at_least_min_entropy():
    counts = [100, 50, 25, 10, 5, 1]
    assert (entropy.shannon_entropy_of_histogram(counts)
            >= entropy.min_entropy_of_histogram(counts))


def test_normalise_sums_to_one():
    probs = entropy.normalise([3, 1, 4, 1, 5])
    assert sum(probs) == pytest.approx(1.0)
    assert entropy.normalise([0, 0]) == [0.0, 0.0]


# ---------------------------------------------------- MCV cross-check


def test_mcv_needs_enough_samples_before_it_reports():
    estimator = entropy.MostCommonValue(1024)
    estimator.update(range(100))
    assert estimator.min_entropy() is None
    assert estimator.probabilities() is None


def test_mcv_recovers_a_known_uniform_distribution():
    estimator = entropy.MostCommonValue(256)
    for _ in range(200):
        estimator.update(range(256))
    h = estimator.min_entropy()
    assert h is not None and 7.0 < h <= 8.0


def test_mcv_detects_a_degenerate_source():
    estimator = entropy.MostCommonValue(1024)
    estimator.update([7] * 5000)
    assert estimator.min_entropy() == pytest.approx(0.0, abs=1e-9)


def test_mcv_bound_never_exceeds_the_truth():
    import random
    rng = random.Random(99)
    p_max = 0.25
    symbols = [0 if rng.random() < p_max else rng.randrange(1, 16) for _ in range(200000)]
    estimator = entropy.MostCommonValue(16)
    estimator.update(symbols)
    assert estimator.min_entropy() <= -math.log2(p_max) + 1e-6


def test_projected_rate_scales_with_count_rate():
    assessment = make_assessment()
    assert (entropy.projected_bit_rate(44.0, assessment)
            > entropy.projected_bit_rate(4.4, assessment))
    assert entropy.projected_bit_rate(0.0, assessment) == 0.0
