"""NIST SP 800-90B min-entropy estimators for the RadiaCode noise source.

Why this and not just Dieharder
-------------------------------
Dieharder answers "does this byte stream look uniform?". For a generator whose
output passes through a DRBG, the answer is yes no matter how bad the physics
is -- a DRBG launders any seed into a uniform-looking stream. The question that
actually matters for a hardware RNG is "how much unpredictability does the
physical source supply per sample?", and that is what SP 800-90B estimators
measure. They work on the *raw* channel stream, before any conditioning, and
they need on the order of a million samples rather than gigabytes.

This module implements the non-IID track of SP 800-90B section 6.3 and reports
the minimum across estimators, which is the standard's own rule. Implemented
here:

    6.3.1  Most Common Value
    6.3.2  Collision                  (on the binarised stream)
    6.3.3  Markov                     (on the binarised stream)
    6.3.4  Compression                (on the binarised stream)
    6.3.5  t-Tuple
    6.3.6  Longest Repeated Substring
    6.3.7  MultiMCW prediction
    6.3.8  Lag prediction
    6.3.9  MultiMMC prediction
    6.3.10 LZ78Y prediction

For a formal, citable assessment use NIST's own reference implementation --
https://github.com/usnistgov/SP800-90B_EntropyAssessment -- against the 8-bit
export this script can write (``--export-nist``). This module is an
independent cross-check with the same formulas, not a replacement for the
reference tool.

Usage::

    python validation/sp800_90b.py data/soak.channels.u16
    python validation/sp800_90b.py data/soak.channels.u16 --export-nist data/nist8.bin
    python validation/sp800_90b.py data/soak.batches.jsonl --json
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
from typing import Iterable, Optional, Sequence

Z_ALPHA = 2.576  # one-sided 99% bound, per SP 800-90B


# ------------------------------------------------------------------- loading


def load_u16(path: str, limit: Optional[int] = None) -> list[int]:
    """Load a flat little-endian uint16 file of channel values."""
    with open(path, 'rb') as handle:
        raw = handle.read()
    values = [int.from_bytes(raw[i:i + 2], 'little') for i in range(0, len(raw) - 1, 2)]
    return values[:limit] if limit else values


def load_jsonl(path: str, limit: Optional[int] = None) -> list[int]:
    """Load channel values from a ``.batches.jsonl`` capture."""
    values: list[int] = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            values.extend(json.loads(line)['c'])
            if limit and len(values) >= limit:
                return values[:limit]
    return values


def load_bytes(path: str, limit: Optional[int] = None) -> list[int]:
    with open(path, 'rb') as handle:
        raw = handle.read()
    return list(raw[:limit] if limit else raw)


def load_any(path: str, limit: Optional[int] = None) -> list[int]:
    if path.endswith('.u16'):
        return load_u16(path, limit)
    if path.endswith('.jsonl'):
        return load_jsonl(path, limit)
    return load_bytes(path, limit)


def binarise(samples: Sequence[int], bits: Optional[int] = None) -> list[int]:
    """Map each sample to its bit expansion, as SP 800-90B section 6.4 requires.

    The collision, Markov and compression estimators are defined for binary
    data only; the standard's remedy for a larger alphabet is to expand each
    sample into its bits and run those estimators on the result.
    """
    if not samples:
        return []
    width = bits or max(1, max(samples).bit_length())
    out: list[int] = []
    for value in samples:
        for shift in range(width - 1, -1, -1):
            out.append((value >> shift) & 1)
    return out


# --------------------------------------------------------- helper statistics


def _upper_bound(p_hat: float, n: int) -> float:
    if n < 2:
        return 1.0
    return min(1.0, p_hat + Z_ALPHA * math.sqrt(p_hat * (1.0 - p_hat) / (n - 1)))


def _entropy(p: float) -> float:
    p = min(1.0, max(p, 1e-300))
    return -math.log2(p)


def _p_local(longest_run: int, n_predictions: int, alpha: float = 0.99) -> float:
    """Solve for the per-trial success probability implied by the longest run.

    Uses Feller's run-length approximation: for Bernoulli(p) trials, the
    probability of seeing no run of ``r`` successes in ``n`` trials is

        (1 - p*x) / ((r + 1 - r*x) * q) * x^-(n+1)

    where ``x`` is the smallest root greater than 1 of
    ``1 - x + q*p^r*x^(r+1) = 0``. We binary-search ``p`` until that
    probability equals ``alpha``.
    """
    r = longest_run + 1
    if r <= 1 or n_predictions <= 0:
        return 1.0
    # A very long correct-prediction run means the predictor is essentially
    # always right; the root-finding below overflows in that regime and the
    # answer is unambiguous anyway.
    if longest_run >= 0.5 * n_predictions or r > 1000:
        return 1.0

    def no_run_probability(p: float) -> float:
        if p <= 0.0:
            return 1.0
        if p >= 1.0:
            return 0.0
        q = 1.0 - p

        def poly(x: float) -> float:
            try:
                return 1.0 - x + q * (p ** r) * (x ** (r + 1))
            except OverflowError:
                # Overflow only happens on the positive branch, which is what
                # the bracketing loop treats as "still too high".
                return math.inf

        # Root just above 1; poly(1) = q*p^r > 0 and poly grows negative first.
        low, high = 1.0, 1.0 / p
        # Walk high down until poly(high) < 0 so we bracket a sign change.
        for _ in range(200):
            if poly(high) < 0:
                break
            high = 1.0 + (high - 1.0) * 0.5
            if high - 1.0 < 1e-15:
                return 1.0
        for _ in range(200):
            mid = 0.5 * (low + high)
            if poly(mid) > 0:
                low = mid
            else:
                high = mid
        x = 0.5 * (low + high)
        denominator = (r + 1 - r * x) * q
        if abs(denominator) < 1e-300:
            return 1.0
        try:
            return (1.0 - p * x) / denominator * (x ** -(n_predictions + 1))
        except (OverflowError, ZeroDivisionError):
            return 0.0

    low, high = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        if no_run_probability(mid) > alpha:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _predictor_entropy(correct: list[bool]) -> tuple[float, dict]:
    """SP 800-90B section 6.3.7 scoring shared by all four predictors."""
    n = len(correct)
    if n == 0:
        return 0.0, {}
    hits = sum(correct)
    p_global_hat = hits / n
    p_global = _upper_bound(p_global_hat, n)

    longest = run = 0
    for value in correct:
        run = run + 1 if value else 0
        longest = max(longest, run)
    p_local = _p_local(longest, n)

    # The standard takes the larger of the two bounds: whichever way the
    # predictor did better is the one an attacker would exploit.
    p_max = max(p_global, p_local)
    return _entropy(p_max), {
        'predictions': n,
        'correct': hits,
        'p_global_hat': p_global_hat,
        'p_global_upper': p_global,
        'longest_run': longest,
        'p_local': p_local,
        'p_max': p_max,
    }


# ------------------------------------------------------------- 6.3.1 MCV


def most_common_value(samples: Sequence[int]) -> tuple[float, dict]:
    n = len(samples)
    if n < 2:
        return 0.0, {'reason': 'too few samples'}
    counts = collections.Counter(samples)
    mode_value, mode_count = counts.most_common(1)[0]
    p_hat = mode_count / n
    p_upper = _upper_bound(p_hat, n)
    return _entropy(p_upper), {
        'samples': n,
        'distinct': len(counts),
        'mode': mode_value,
        'mode_count': mode_count,
        'p_hat': p_hat,
        'p_upper': p_upper,
    }


# ------------------------------------------------------- 6.3.2 collision


def collision(binary: Sequence[int]) -> tuple[float, dict]:
    """Mean time to a repeated value, inverted to a bound on ``p_max``.

    For a binary source with symbol probabilities ``p`` and ``q = 1-p``, the
    time to the first collision is 2 with probability ``p^2 + q^2`` and 3
    otherwise, so

        E[T] = 2(p^2 + q^2) + 3(2pq) = 2 + 2pq

    which inverts in closed form to ``p = (1 + sqrt(5 - 2*E[T])) / 2``. A
    perfectly unbiased bit gives E[T] = 2.5 and hence p = 0.5, i.e. one bit of
    entropy; a fully biased one gives E[T] = 2 and zero entropy.
    """
    n = len(binary)
    if n < 100:
        return 1.0, {'reason': 'too few samples'}
    times: list[int] = []
    index = 0
    while index + 2 < n:
        # With a two-symbol alphabet any three consecutive samples must repeat.
        if binary[index] == binary[index + 1]:
            times.append(2)
            index += 2
        else:
            times.append(3)
            index += 3
    v = len(times)
    if v < 2:
        return 1.0, {'reason': 'no collisions observed'}
    mean = sum(times) / v
    variance = sum((t - mean) ** 2 for t in times) / (v - 1)
    mean_lower = mean - Z_ALPHA * math.sqrt(variance / v)

    # Clamp into the achievable range [2, 2.5] before inverting.
    clamped = min(2.5, max(2.0, mean_lower))
    discriminant = max(0.0, 5.0 - 2.0 * clamped)
    p = min(1.0, max(0.5, 0.5 * (1.0 + math.sqrt(discriminant))))
    return _entropy(p), {
        'collisions': v,
        'mean_time': mean,
        'mean_lower': mean_lower,
        'p_max': p,
        'note': 'binarised stream',
    }


# ---------------------------------------------------------- 6.3.3 Markov


def markov(binary: Sequence[int], chain_length: int = 128) -> tuple[float, dict]:
    """First-order Markov bound on the most probable length-128 path.

    Estimates the initial and transition probabilities with 99% upper bounds,
    then finds the highest-probability path of ``chain_length`` states by
    dynamic programming (a greedy walk is not enough -- the best path can
    require an initially worse step). Entropy is that path's probability
    spread over the chain, capped at one bit per sample.
    """
    n = len(binary)
    if n < 1000:
        return 1.0, {'reason': 'too few samples'}
    counts = [0, 0]
    transitions = [[0, 0], [0, 0]]
    for i in range(n - 1):
        counts[binary[i]] += 1
        transitions[binary[i]][binary[i + 1]] += 1
    total = counts[0] + counts[1]
    if total == 0 or counts[0] == 0 or counts[1] == 0:
        return 0.0, {'reason': 'degenerate binary stream', 'note': 'binarised stream'}

    def upper(numerator: int, denominator: int) -> float:
        p = numerator / denominator
        return min(1.0, p + Z_ALPHA * math.sqrt(p * (1 - p) / denominator))

    p_initial = [
        min(1.0, counts[0] / total + Z_ALPHA * math.sqrt(
            (counts[0] / total) * (1 - counts[0] / total) / total)),
        min(1.0, counts[1] / total + Z_ALPHA * math.sqrt(
            (counts[1] / total) * (1 - counts[1] / total) / total)),
    ]
    p_transition = [
        [upper(transitions[0][0], counts[0]), upper(transitions[0][1], counts[0])],
        [upper(transitions[1][0], counts[1]), upper(transitions[1][1], counts[1])],
    ]

    # DP in log space so a 128-step product does not underflow.
    log_best = [math.log2(max(p_initial[s], 1e-300)) for s in (0, 1)]
    for _ in range(chain_length - 1):
        log_next = [-math.inf, -math.inf]
        for destination in (0, 1):
            for origin in (0, 1):
                candidate = log_best[origin] + math.log2(
                    max(p_transition[origin][destination], 1e-300))
                if candidate > log_next[destination]:
                    log_next[destination] = candidate
        log_best = log_next

    log_p_max = max(log_best)
    h_per_sample = min(1.0, -log_p_max / chain_length)
    return max(0.0, h_per_sample), {
        'p_initial_upper': p_initial,
        'p_transition_upper': p_transition,
        'chain_length': chain_length,
        'note': 'binarised stream',
    }


# ------------------------------------------------------ 6.3.4 compression


def _compression_g(z: float, d: int, total_blocks: int, v: int) -> float:
    """The ``G(z)`` function from SP 800-90B section 6.3.4.

        G(z) = (1/v) * sum_{t=d+1..L} sum_{u=1..t} log2(u) * F(z, t, u)
        F(z, t, u) = z^2 (1-z)^(u-1)   for u < t
                     z   (1-z)^(t-1)   for u = t

    Evaluated in O(L) by carrying the inner sum forward instead of recomputing
    it, and short-circuiting once ``(1-z)^(t-1)`` underflows -- past that point
    every remaining term contributes the same constant ``z^2 * S``.
    """
    if z <= 0.0 or z >= 1.0 or v <= 0:
        return 0.0
    one_minus = 1.0 - z
    inner_sum = 0.0          # S(t) = sum_{u=1..t-1} log2(u) (1-z)^(u-1)
    power = 1.0              # (1-z)^(t-1)
    total = 0.0
    for t in range(1, total_blocks + 1):
        if t > d:
            total += z * z * inner_sum + math.log2(t) * z * power
        # Advance to S(t+1) and (1-z)^t.
        inner_sum += math.log2(t) * power
        power *= one_minus
        if power < 1e-300:
            # Remaining terms: the geometric factor is gone, so only the
            # z^2 * S piece survives and S no longer changes.
            remaining = total_blocks - max(t, d)
            if remaining > 0:
                total += z * z * inner_sum * remaining
            break
    return total / v


def compression(binary: Sequence[int], block_bits: int = 6, d: int = 1000) -> tuple[float, dict]:
    """Maurer-style universal statistic, per SP 800-90B section 6.3.4.

    Measures the mean ``log2`` distance back to the previous occurrence of each
    6-bit block, lower-bounds it at 99% confidence, then solves
    ``G(p) + (2^b - 1) G(q) = mean`` for the most likely symbol probability
    ``p``. Returns entropy per *bit* of the binarised stream.
    """
    n_blocks = (len(binary)) // block_bits
    if n_blocks < d + 100:
        return float(block_bits), {'reason': 'too few blocks'}

    blocks = [0] * n_blocks
    for i in range(n_blocks):
        value = 0
        base = i * block_bits
        for j in range(block_bits):
            value = (value << 1) | binary[base + j]
        blocks[i] = value

    last_seen: dict[int, int] = {}
    for i in range(d):
        last_seen[blocks[i]] = i + 1

    distances: list[float] = []
    for i in range(d, n_blocks):
        position = i + 1
        previous = last_seen.get(blocks[i])
        distance = position - previous if previous is not None else position
        distances.append(math.log2(distance))
        last_seen[blocks[i]] = position

    v = len(distances)
    if v < 2:
        return float(block_bits), {'reason': 'no test blocks'}
    mean = sum(distances) / v
    variance = sum((value - mean) ** 2 for value in distances) / (v - 1)
    mean_lower = mean - Z_ALPHA * math.sqrt(variance / v)

    alphabet = 1 << block_bits

    def expected(p: float) -> float:
        q = (1.0 - p) / (alphabet - 1)
        return _compression_g(p, d, n_blocks, v) + (alphabet - 1) * _compression_g(q, d, n_blocks, v)

    uniform_p = 1.0 / alphabet
    if mean_lower >= expected(uniform_p):
        # The source looks at least as unpredictable as a uniform one.
        return 1.0, {
            'blocks': n_blocks, 'mean': mean, 'mean_lower': mean_lower,
            'p_max': uniform_p, 'block_bits': block_bits, 'note': 'binarised stream',
        }

    # expected() decreases as p rises, so bisect on p in [1/2^b, 1].
    low, high = uniform_p, 1.0
    for _ in range(40):
        mid = 0.5 * (low + high)
        if expected(mid) > mean_lower:
            low = mid
        else:
            high = mid
    p = 0.5 * (low + high)
    h_per_bit = min(1.0, max(0.0, -math.log2(p) / block_bits))
    return h_per_bit, {
        'blocks': n_blocks,
        'mean': mean,
        'mean_lower': mean_lower,
        'p_max': p,
        'block_bits': block_bits,
        'note': 'binarised stream',
    }


# ----------------------------------------------------------- 6.3.5 t-Tuple


def t_tuple(samples: Sequence[int], max_t: int = 32) -> tuple[float, dict, int]:
    n = len(samples)
    if n < 100:
        return 0.0, {'reason': 'too few samples'}, 1
    best_p = 0.0
    t_found = 1
    for t in range(1, max_t + 1):
        counts: dict[tuple, int] = collections.defaultdict(int)
        limit = n - t + 1
        if limit <= 0:
            break
        for i in range(limit):
            counts[tuple(samples[i:i + t])] += 1
        max_count = max(counts.values())
        if max_count < 35:
            break
        t_found = t
        p = (max_count / limit) ** (1.0 / t)
        best_p = max(best_p, p)
    if best_p <= 0:
        return 0.0, {'reason': 'no tuple reached the count threshold'}, 1
    p_upper = _upper_bound(best_p, n)
    return _entropy(p_upper), {
        't': t_found,
        'p_hat': best_p,
        'p_upper': p_upper,
    }, t_found


# --------------------------------------------------------------- 6.3.6 LRS


def longest_repeated_substring(samples: Sequence[int], u: int, max_v: int = 64) -> tuple[float, dict]:
    n = len(samples)
    if n < 100:
        return 0.0, {'reason': 'too few samples'}
    best_p = 0.0
    v_found = u
    for length in range(u, max_v + 1):
        limit = n - length + 1
        if limit < 2:
            break
        counts: dict[tuple, int] = collections.defaultdict(int)
        for i in range(limit):
            counts[tuple(samples[i:i + length])] += 1
        repeated = sum(c * (c - 1) // 2 for c in counts.values())
        if repeated == 0:
            break
        v_found = length
        pairs = limit * (limit - 1) // 2
        p_w = repeated / pairs
        best_p = max(best_p, p_w ** (1.0 / length))
    if best_p <= 0:
        return 0.0, {'reason': 'no repeated substrings'}
    return _entropy(min(1.0, best_p)), {'u': u, 'v': v_found, 'p_max': best_p}


# --------------------------------------------------------- 6.3.7 MultiMCW


def multi_mcw(samples: Sequence[int], windows: Sequence[int] = (63, 255, 1023, 4095)) -> tuple[float, dict]:
    """Predict the most common value in each of several trailing windows."""
    n = len(samples)
    usable = [w for w in windows if w < n]
    if not usable:
        return 0.0, {'reason': 'too few samples'}
    scoreboard = [0] * len(usable)
    correct: list[bool] = []
    counters = [collections.Counter() for _ in usable]
    start = max(usable)
    for i, w in enumerate(usable):
        counters[i].update(samples[start - w:start])
    best = 0
    for index in range(start, n):
        prediction = counters[best].most_common(1)[0][0] if counters[best] else samples[index - 1]
        actual = samples[index]
        correct.append(prediction == actual)
        for i, w in enumerate(usable):
            guess = counters[i].most_common(1)[0][0] if counters[i] else None
            if guess == actual:
                scoreboard[i] += 1
            counters[i][samples[index - w]] -= 1
            if counters[i][samples[index - w]] <= 0:
                del counters[i][samples[index - w]]
            counters[i][actual] += 1
        best = max(range(len(usable)), key=lambda i: scoreboard[i])
    h, detail = _predictor_entropy(correct)
    detail['windows'] = list(usable)
    return h, detail


# -------------------------------------------------------------- 6.3.8 Lag


def lag_predictor(samples: Sequence[int], max_lag: int = 128) -> tuple[float, dict]:
    """Predict ``samples[i-lag]`` for the currently best-scoring lag."""
    n = len(samples)
    if n <= max_lag + 2:
        return 0.0, {'reason': 'too few samples'}
    scoreboard = [0] * max_lag
    correct: list[bool] = []
    best = 0
    for index in range(max_lag, n):
        actual = samples[index]
        correct.append(samples[index - best - 1] == actual)
        for lag in range(max_lag):
            if samples[index - lag - 1] == actual:
                scoreboard[lag] += 1
        best = max(range(max_lag), key=lambda i: scoreboard[i])
    h, detail = _predictor_entropy(correct)
    detail['best_lag'] = best + 1
    detail['max_lag'] = max_lag
    return h, detail


# ---------------------------------------------------------- 6.3.9 MultiMMC


def multi_mmc(samples: Sequence[int], depths: Sequence[int] = (1, 2, 3, 4, 8, 16)) -> tuple[float, dict]:
    """Multi Most Common in Window: order-d Markov models voting."""
    n = len(samples)
    if n < 64:
        return 0.0, {'reason': 'too few samples'}
    models: list[dict[tuple, collections.Counter]] = [dict() for _ in depths]
    scoreboard = [0] * len(depths)
    correct: list[bool] = []
    start = max(depths)
    best = 0
    for index in range(start, n):
        actual = samples[index]
        predictions = []
        for i, depth in enumerate(depths):
            context = tuple(samples[index - depth:index])
            counter = models[i].get(context)
            predictions.append(counter.most_common(1)[0][0] if counter else None)
        correct.append(predictions[best] == actual)
        for i, depth in enumerate(depths):
            if predictions[i] == actual:
                scoreboard[i] += 1
            context = tuple(samples[index - depth:index])
            models[i].setdefault(context, collections.Counter())[actual] += 1
        best = max(range(len(depths)), key=lambda i: scoreboard[i])
    h, detail = _predictor_entropy(correct)
    detail['depths'] = list(depths)
    detail['best_depth'] = depths[best]
    return h, detail


# ----------------------------------------------------------- 6.3.10 LZ78Y


def lz78y(samples: Sequence[int], max_dictionary: int = 65536, b: int = 16) -> tuple[float, dict]:
    """Dictionary predictor in the spirit of LZ78."""
    n = len(samples)
    if n < b + 2:
        return 0.0, {'reason': 'too few samples'}
    dictionary: dict[tuple, collections.Counter] = {}
    correct: list[bool] = []
    for index in range(b, n):
        actual = samples[index]
        prediction = None
        best_count = 0
        for length in range(1, b + 1):
            context = tuple(samples[index - length:index])
            counter = dictionary.get(context)
            if counter:
                value, count = counter.most_common(1)[0]
                if count > best_count:
                    best_count = count
                    prediction = value
        correct.append(prediction == actual)
        for length in range(1, b + 1):
            context = tuple(samples[index - length:index])
            if context in dictionary or len(dictionary) < max_dictionary:
                dictionary.setdefault(context, collections.Counter())[actual] += 1
    h, detail = _predictor_entropy(correct)
    detail['dictionary_size'] = len(dictionary)
    return h, detail


# ----------------------------------------------------------------- driver


def assess(samples: Sequence[int], quick: bool = False) -> dict:
    """Run every estimator and return the minimum, per SP 800-90B section 6.3."""
    results: dict[str, dict] = {}
    n = len(samples)
    alphabet = len(set(samples))

    h, detail = most_common_value(samples)
    results['most_common_value'] = {'h_min': h, **detail}

    h_tt, detail_tt, t_found = t_tuple(samples, max_t=8 if quick else 16)
    results['t_tuple'] = {'h_min': h_tt, **detail_tt}

    h_lrs, detail_lrs = longest_repeated_substring(
        samples, u=t_found + 1, max_v=16 if quick else 32)
    results['lrs'] = {'h_min': h_lrs, **detail_lrs}

    h_mcw, detail_mcw = multi_mcw(samples)
    results['multi_mcw'] = {'h_min': h_mcw, **detail_mcw}

    h_lag, detail_lag = lag_predictor(samples, max_lag=32 if quick else 128)
    results['lag'] = {'h_min': h_lag, **detail_lag}

    h_mmc, detail_mmc = multi_mmc(samples, depths=(1, 2, 3) if quick else (1, 2, 3, 4, 8, 16))
    results['multi_mmc'] = {'h_min': h_mmc, **detail_mmc}

    h_lz, detail_lz = lz78y(samples, b=8 if quick else 16)
    results['lz78y'] = {'h_min': h_lz, **detail_lz}

    # Binary-only estimators, run on the bit expansion and scaled back up.
    bit_width = max(1, max(samples).bit_length()) if samples else 1
    bits = binarise(samples, bit_width)
    h_col, detail_col = collision(bits)
    results['collision_per_bit'] = {'h_min': h_col, **detail_col}
    h_mk, detail_mk = markov(bits)
    results['markov_per_bit'] = {'h_min': h_mk, **detail_mk}
    h_cmp, detail_cmp = compression(bits)
    results['compression_per_bit'] = {'h_min': h_cmp, **detail_cmp}

    # An estimator that reported a 'reason' could not run and must be excluded.
    # An estimator that legitimately returned 0.0 must NOT be -- a zero means
    # the source is fully predictable, which is the single most important
    # result the battery can produce.
    def scaled(name: str) -> float:
        value = max(0.0, results[name]['h_min'])
        return value * bit_width if name.endswith('_per_bit') else value

    usable = {name: scaled(name) for name, detail in results.items() if 'reason' not in detail}
    skipped = {name: detail['reason'] for name, detail in results.items() if 'reason' in detail}

    if usable:
        binding = min(usable, key=lambda name: usable[name])
        h_final = usable[binding]
    else:
        binding, h_final = None, 0.0

    return {
        'samples': n,
        'alphabet_size': alphabet,
        'bit_width': bit_width,
        'estimators': results,
        'min_entropy_per_sample': h_final,
        'binding_estimator': binding,
        'skipped_estimators': skipped,
        'sufficient_samples': n >= 1_000_000,
    }


def export_nist_8bit(samples: Sequence[int], path: str, shift: int = 2) -> dict:
    """Write an 8-bit-per-sample file for NIST's ``ea_non_iid`` reference tool.

    That tool accepts 1-8 bits per symbol, while our channels are 10 bits, so
    the top 8 bits are kept (``channel >> 2``). That discards information and
    therefore *understates* the true entropy -- which is the safe direction for
    a validation artefact.

    Run it as::

        ea_non_iid -i -a -v <path> 8
    """
    with open(path, 'wb') as handle:
        handle.write(bytes((value >> shift) & 0xFF for value in samples))
    return {
        'path': path,
        'samples': len(samples),
        'bits_per_symbol': 8,
        'transform': f'channel >> {shift}',
        'command': f'ea_non_iid -i -a -v {path} 8',
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('path', help='.channels.u16, .batches.jsonl, or a raw byte file')
    parser.add_argument('--limit', type=int, default=None, help='use at most N samples')
    parser.add_argument('--quick', action='store_true', help='cheaper settings, coarser bounds')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--export-nist', metavar='PATH',
                        help="also write an 8-bit file for NIST's ea_non_iid")
    parser.add_argument('--compare', type=float, default=None, metavar='BITS',
                        help='compare against the value radiarandom assesses (e.g. 3.6)')
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f'no such file: {args.path}', file=sys.stderr)
        return 1

    samples = load_any(args.path, args.limit)
    if len(samples) < 100:
        print(f'only {len(samples)} samples; collect more before assessing', file=sys.stderr)
        return 1

    print(f'assessing {len(samples)} samples from {args.path}...', file=sys.stderr, flush=True)
    result = assess(samples, quick=args.quick)

    if args.export_nist:
        result['nist_export'] = export_nist_8bit(samples, args.export_nist)

    if args.compare is not None:
        result['assessed_by_radiarandom'] = args.compare
        result['assessment_is_conservative'] = result['min_entropy_per_sample'] >= args.compare

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print()
    print(f'samples          {result["samples"]:,}  '
          f'(alphabet {result["alphabet_size"]}, {result["bit_width"]} bits wide)')
    if not result['sufficient_samples']:
        print(f'                 NOTE: SP 800-90B asks for 1,000,000 samples for the')
        print(f'                 non-IID track; bounds here are wider than they would be.')
    print()
    print(f'{"estimator":<24} {"H_min (bits/sample)":>20}')
    print('-' * 46)
    width = result['bit_width']
    for name, detail in result['estimators'].items():
        value = detail['h_min']
        shown = value * width if name.endswith('_per_bit') else value
        suffix = f'   ({value:.4f}/bit x {width})' if name.endswith('_per_bit') else ''
        note = f'   [{detail["reason"]}]' if 'reason' in detail else ''
        print(f'{name:<24} {shown:>20.4f}{suffix}{note}')
    print('-' * 46)
    print(f'{"MINIMUM (assessed)":<24} {result["min_entropy_per_sample"]:>20.4f}')
    print(f'binding estimator: {result["binding_estimator"]}')

    if args.compare is not None:
        print()
        verdict = 'CONSERVATIVE' if result['assessment_is_conservative'] else 'OPTIMISTIC'
        print(f'radiarandom banks {args.compare:.3f} bits/photon; '
              f'measurement says {result["min_entropy_per_sample"]:.3f}  -> {verdict}')
        if not result['assessment_is_conservative']:
            print('  Lower --h-per-photon until the banked value sits below the measurement.')

    if args.export_nist:
        print()
        print(f'NIST export: {result["nist_export"]["path"]}')
        print(f'  run: {result["nist_export"]["command"]}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
