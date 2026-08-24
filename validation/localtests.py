"""A portable statistical battery, for when Dieharder is not available.

Dieharder is the reference and this does not replace it. What this does give
you is a fast, dependency-free pre-flight check that runs anywhere Python does
-- including Windows, where dieharder has no native build -- so a broken
pipeline is caught in seconds rather than after an eight-hour battery.

The tests are the classical ones, with real p-values:

    monobit                 NIST SP 800-22 2.1
    block frequency         NIST SP 800-22 2.2
    runs                    NIST SP 800-22 2.3
    longest run of ones     NIST SP 800-22 2.4
    binary matrix rank      NIST SP 800-22 2.5  (32x32)
    cumulative sums         NIST SP 800-22 2.13
    approximate entropy     NIST SP 800-22 2.12
    serial correlation      lag-1 autocorrelation of the byte stream
    byte chi-square         uniformity over 256 values
    poker (4-bit)           FIPS 140-2 style
    bit-position bias       per-bit-position monobit, catches lane defects
    birthday spacings       Marsaglia, catches lattice structure

Usage::

    python validation/localtests.py data/output.bin
    python validation/localtests.py data/output.bin --json
    python validation/localtests.py --self-test        # sanity-check the tests
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
from typing import Optional, Sequence

ALPHA = 0.01  # a test is called a failure below this p-value

#: Tests whose p-value is only meaningful in the lower tail. bit_position_bias
#: is a Bonferroni-corrected minimum, so it saturates at 1.0 on healthy data;
#: flagging that as "suspiciously good" would be nonsense.
ONE_SIDED_TESTS = frozenset({'bit_position_bias'})


# ------------------------------------------------------- special functions


def _gammainc_upper(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a, x), Numerical Recipes style."""
    if x < 0 or a <= 0:
        return float('nan')
    if x == 0:
        return 1.0
    if x < a + 1.0:
        # Series expansion for P(a, x), then Q = 1 - P.
        ap = a
        total = 1.0 / a
        delta = total
        for _ in range(1000):
            ap += 1.0
            delta *= x / ap
            total += delta
            if abs(delta) < abs(total) * 1e-15:
                break
        return 1.0 - total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    # Continued fraction for Q(a, x).
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def chi2_p_value(chi2: float, dof: int) -> float:
    """Upper-tail probability of a chi-square statistic."""
    if dof <= 0:
        return float('nan')
    return _gammainc_upper(dof / 2.0, chi2 / 2.0)


def erfc(x: float) -> float:
    return math.erfc(x)


# ------------------------------------------------------------- bit helpers


def to_bits(data: bytes) -> list[int]:
    bits: list[int] = []
    for byte in data:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits


# ------------------------------------------------------------------- tests


def monobit(bits: Sequence[int]) -> dict:
    n = len(bits)
    if n < 100:
        return {'p_value': None, 'note': 'need at least 100 bits'}
    s = sum(1 if bit else -1 for bit in bits)
    s_obs = abs(s) / math.sqrt(n)
    return {'p_value': erfc(s_obs / math.sqrt(2)), 'ones': sum(bits), 'n': n}


def block_frequency(bits: Sequence[int], block_size: int = 128) -> dict:
    n = len(bits)
    blocks = n // block_size
    if blocks < 1:
        return {'p_value': None, 'note': 'not enough bits for one block'}
    total = 0.0
    for i in range(blocks):
        chunk = bits[i * block_size:(i + 1) * block_size]
        pi = sum(chunk) / block_size
        total += (pi - 0.5) ** 2
    chi2 = 4.0 * block_size * total
    return {'p_value': chi2_p_value(chi2, blocks), 'blocks': blocks, 'chi2': chi2}


def runs(bits: Sequence[int]) -> dict:
    n = len(bits)
    if n < 100:
        return {'p_value': None, 'note': 'need at least 100 bits'}
    pi = sum(bits) / n
    if abs(pi - 0.5) >= 2.0 / math.sqrt(n):
        return {'p_value': 0.0, 'note': 'failed the monobit precondition', 'pi': pi}
    v = 1 + sum(1 for i in range(n - 1) if bits[i] != bits[i + 1])
    numerator = abs(v - 2.0 * n * pi * (1 - pi))
    denominator = 2.0 * math.sqrt(2.0 * n) * pi * (1 - pi)
    return {'p_value': erfc(numerator / denominator), 'runs': v, 'pi': pi}


_LONGEST_RUN_PARAMS = {
    # M: (K, N, thresholds, probabilities)
    8: (3, 16, [1, 2, 3, 4], [0.2148, 0.3672, 0.2305, 0.1875]),
    128: (5, 49, [4, 5, 6, 7, 8, 9],
          [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]),
    10000: (6, 75, [10, 11, 12, 13, 14, 15, 16],
            [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]),
}


def longest_run_of_ones(bits: Sequence[int]) -> dict:
    n = len(bits)
    if n >= 750000:
        m = 10000
    elif n >= 6272:
        m = 128
    elif n >= 128:
        m = 8
    else:
        return {'p_value': None, 'note': 'need at least 128 bits'}
    k, n_blocks, thresholds, probabilities = _LONGEST_RUN_PARAMS[m]
    available = n // m
    if available < n_blocks:
        n_blocks = available
    if n_blocks < 1:
        return {'p_value': None, 'note': 'not enough blocks'}

    counts = [0] * (k + 1)
    for i in range(n_blocks):
        chunk = bits[i * m:(i + 1) * m]
        longest = current = 0
        for bit in chunk:
            current = current + 1 if bit else 0
            longest = max(longest, current)
        if longest <= thresholds[0]:
            counts[0] += 1
        elif longest >= thresholds[-1]:
            counts[k] += 1
        else:
            counts[longest - thresholds[0]] += 1
    chi2 = sum(
        (counts[i] - n_blocks * probabilities[i]) ** 2 / (n_blocks * probabilities[i])
        for i in range(k + 1)
    )
    return {'p_value': chi2_p_value(chi2, k), 'blocks': n_blocks, 'M': m, 'chi2': chi2}


def binary_matrix_rank(bits: Sequence[int], size: int = 32) -> dict:
    """Rank distribution of 32x32 GF(2) matrices -- catches linear structure."""
    per_matrix = size * size
    matrices = len(bits) // per_matrix
    if matrices < 38:
        return {'p_value': None, 'note': f'need {38 * per_matrix} bits, have {len(bits)}'}

    full = deficient1 = other = 0
    for index in range(matrices):
        base = index * per_matrix
        rows = []
        for r in range(size):
            value = 0
            offset = base + r * size
            for c in range(size):
                value = (value << 1) | bits[offset + c]
            rows.append(value)
        # Gaussian elimination over GF(2).
        rank = 0
        pivot_bit = 1 << (size - 1)
        row_index = 0
        for _ in range(size):
            pivot = None
            for r in range(row_index, size):
                if rows[r] & pivot_bit:
                    pivot = r
                    break
            if pivot is not None:
                rows[row_index], rows[pivot] = rows[pivot], rows[row_index]
                for r in range(size):
                    if r != row_index and (rows[r] & pivot_bit):
                        rows[r] ^= rows[row_index]
                row_index += 1
                rank += 1
            pivot_bit >>= 1
        if rank == size:
            full += 1
        elif rank == size - 1:
            deficient1 += 1
        else:
            other += 1

    # Asymptotic probabilities for a random 32x32 GF(2) matrix.
    p_full, p_deficient1 = 0.2888, 0.5776
    p_other = 1.0 - p_full - p_deficient1
    expected = [matrices * p_full, matrices * p_deficient1, matrices * p_other]
    observed = [full, deficient1, other]
    chi2 = sum((o - e) ** 2 / e for o, e in zip(observed, expected) if e > 0)
    return {
        'p_value': chi2_p_value(chi2, 2),
        'matrices': matrices,
        'full_rank': full,
        'rank_minus_1': deficient1,
        'lower': other,
    }


def cumulative_sums(bits: Sequence[int], forward: bool = True) -> dict:
    n = len(bits)
    if n < 100:
        return {'p_value': None, 'note': 'need at least 100 bits'}
    sequence = bits if forward else list(reversed(bits))
    total = 0
    z = 0
    for bit in sequence:
        total += 1 if bit else -1
        z = max(z, abs(total))
    if z == 0:
        return {'p_value': 1.0, 'z': 0}
    root_n = math.sqrt(n)
    p = 1.0
    start = int((-n / z + 1) // 4)
    end = int((n / z - 1) // 4)
    for k in range(start, end + 1):
        p -= _phi((4 * k + 1) * z / root_n) - _phi((4 * k - 1) * z / root_n)
    start = int((-n / z - 3) // 4)
    for k in range(start, end + 1):
        p += _phi((4 * k + 3) * z / root_n) - _phi((4 * k + 1) * z / root_n)
    return {'p_value': max(0.0, min(1.0, p)), 'z': z, 'direction': 'forward' if forward else 'reverse'}


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def approximate_entropy(bits: Sequence[int], m: int = 8) -> dict:
    n = len(bits)
    if n < 1000:
        return {'p_value': None, 'note': 'need at least 1000 bits'}

    def phi(block: int) -> float:
        if block == 0:
            return 0.0
        counts: dict[int, int] = collections.defaultdict(int)
        extended = list(bits) + list(bits[:block - 1])
        for i in range(n):
            value = 0
            for j in range(block):
                value = (value << 1) | extended[i + j]
            counts[value] += 1
        return sum((c / n) * math.log(c / n) for c in counts.values())

    ap_en = phi(m) - phi(m + 1)
    chi2 = 2.0 * n * (math.log(2) - ap_en)
    # SP 800-22 2.12 gives p = igamc(2^(m-1), chi2/2); chi2_p_value halves its
    # dof argument internally, so the degrees of freedom to pass is 2^m.
    return {'p_value': chi2_p_value(chi2, 1 << m), 'apen': ap_en, 'm': m}


def serial_correlation(data: bytes) -> dict:
    """Lag-1 autocorrelation of the byte stream, as a z-test."""
    n = len(data)
    if n < 1000:
        return {'p_value': None, 'note': 'need at least 1000 bytes'}
    mean = sum(data) / n
    numerator = sum((data[i] - mean) * (data[i + 1] - mean) for i in range(n - 1))
    denominator = sum((value - mean) ** 2 for value in data)
    if denominator == 0:
        return {'p_value': 0.0, 'note': 'constant stream'}
    r = numerator / denominator
    z = r * math.sqrt(n - 1)
    return {'p_value': erfc(abs(z) / math.sqrt(2)), 'r': r}


def byte_chi_square(data: bytes) -> dict:
    n = len(data)
    if n < 2560:
        return {'p_value': None, 'note': 'need at least 2560 bytes'}
    counts = collections.Counter(data)
    expected = n / 256.0
    chi2 = sum((counts.get(value, 0) - expected) ** 2 / expected for value in range(256))
    return {'p_value': chi2_p_value(chi2, 255), 'chi2': chi2, 'distinct': len(counts)}


def poker_4bit(data: bytes) -> dict:
    nibbles: list[int] = []
    for byte in data:
        nibbles.append(byte >> 4)
        nibbles.append(byte & 0xF)
    n = len(nibbles)
    if n < 5000:
        return {'p_value': None, 'note': 'need at least 2500 bytes'}
    counts = collections.Counter(nibbles)
    expected = n / 16.0
    chi2 = sum((counts.get(value, 0) - expected) ** 2 / expected for value in range(16))
    return {'p_value': chi2_p_value(chi2, 15), 'chi2': chi2}


def bit_position_bias(data: bytes) -> dict:
    """Monobit per bit position -- catches a single stuck or biased lane."""
    n = len(data)
    if n < 1000:
        return {'p_value': None, 'note': 'need at least 1000 bytes'}
    worst_p = 1.0
    worst_position = None
    positions = {}
    for position in range(8):
        ones = sum((byte >> position) & 1 for byte in data)
        s = abs(2 * ones - n) / math.sqrt(n)
        p = erfc(s / math.sqrt(2))
        positions[position] = {'ones': ones, 'p_value': p}
        if p < worst_p:
            worst_p, worst_position = p, position
    # Bonferroni across the eight positions.
    return {
        'p_value': min(1.0, worst_p * 8),
        'worst_position': worst_position,
        'positions': positions,
    }


def birthday_spacings(data: bytes, n_points: int = 512, bits: int = 24) -> dict:
    """Marsaglia's birthday spacings -- sensitive to lattice structure."""
    bytes_per_point = (bits + 7) // 8
    per_round = n_points * bytes_per_point
    rounds = len(data) // per_round
    if rounds < 20:
        return {'p_value': None,
                'note': f'need {20 * per_round} bytes, have {len(data)}'}
    modulus = 1 << bits
    expected_lambda = (n_points ** 3) / (4.0 * modulus)
    observed = []
    for r in range(rounds):
        base = r * per_round
        points = sorted(
            int.from_bytes(data[base + i * bytes_per_point:
                                base + (i + 1) * bytes_per_point], 'little') % modulus
            for i in range(n_points)
        )
        spacings = sorted(points[i + 1] - points[i] for i in range(n_points - 1))
        duplicates = sum(1 for i in range(len(spacings) - 1) if spacings[i] == spacings[i + 1])
        observed.append(duplicates)
    # Compare the observed duplicate counts to Poisson(lambda). Buckets are
    # built up from k = 0 while the expected count stays usable, and everything
    # above lands in a single tail bucket -- which is where a degenerate
    # stream (identical spacings every round) shows up.
    counts = collections.Counter(observed)

    def poisson_pmf(k: int) -> float:
        # In log space: a counter-like stream produces k in the hundreds, where
        # lambda**k and k! both overflow.
        if k < 0:
            return 0.0
        log_p = -expected_lambda + k * math.log(expected_lambda) - math.lgamma(k + 1)
        return math.exp(log_p) if log_p > -700 else 0.0

    buckets: list[tuple[float, int]] = []  # (expected_count, observed_count)
    covered = 0.0
    k = 0
    while k < 1000:
        expected_count = rounds * poisson_pmf(k)
        if expected_count < 5 and buckets:
            break
        buckets.append((expected_count, counts.get(k, 0)))
        covered += expected_count
        k += 1
    tail_expected = max(0.0, rounds - covered)
    tail_observed = sum(count for value, count in counts.items() if value >= k)
    if tail_expected >= 5 or tail_observed:
        buckets.append((max(tail_expected, 1e-9), tail_observed))

    if len(buckets) < 2:
        return {'p_value': None, 'note': 'too few rounds for a chi-square'}
    chi2 = sum((observed_count - expected_count) ** 2 / expected_count
               for expected_count, observed_count in buckets)
    return {'p_value': chi2_p_value(chi2, len(buckets) - 1),
            'rounds': rounds, 'lambda': expected_lambda, 'chi2': chi2,
            'buckets': len(buckets)}


# ---------------------------------------------------------------- driver


def run_battery(data: bytes, quick: bool = False) -> dict:
    bits = to_bits(data)
    results: dict[str, dict] = {
        'monobit': monobit(bits),
        'block_frequency': block_frequency(bits),
        'runs': runs(bits),
        'longest_run_of_ones': longest_run_of_ones(bits),
        'cumulative_sums_forward': cumulative_sums(bits, True),
        'cumulative_sums_reverse': cumulative_sums(bits, False),
        'serial_correlation': serial_correlation(data),
        'byte_chi_square': byte_chi_square(data),
        'poker_4bit': poker_4bit(data),
        'bit_position_bias': bit_position_bias(data),
        'birthday_spacings': birthday_spacings(data),
    }
    if not quick:
        results['binary_matrix_rank'] = binary_matrix_rank(bits)
        results['approximate_entropy'] = approximate_entropy(bits, m=8 if len(bits) > 100000 else 4)

    graded = {name: detail for name, detail in results.items()
              if detail.get('p_value') is not None}
    skipped = {name: detail.get('note', 'skipped') for name, detail in results.items()
               if detail.get('p_value') is None}
    failures = [name for name, detail in graded.items() if detail['p_value'] < ALPHA]
    weak = [name for name, detail in graded.items()
            if ALPHA <= detail['p_value'] < 0.05
            or (detail['p_value'] > 0.99 and name not in ONE_SIDED_TESTS)]

    return {
        'bytes': len(data),
        'tests_run': len(graded),
        'tests_skipped': skipped,
        'failures': failures,
        'weak': weak,
        'passed': not failures,
        'results': results,
    }


def _print_report(report: dict) -> None:
    print()
    print(f'{report["bytes"]:,} bytes, {report["tests_run"]} tests graded '
          f'(alpha = {ALPHA})')
    print()
    print(f'{"test":<28} {"p-value":>10}   verdict')
    print('-' * 56)
    for name, detail in report['results'].items():
        p = detail.get('p_value')
        if p is None:
            print(f'{name:<28} {"--":>10}   skipped: {detail.get("note", "")}')
            continue
        if p < ALPHA:
            verdict = 'FAIL'
        elif p < 0.05 or (p > 0.99 and name not in ONE_SIDED_TESTS):
            verdict = 'weak'
        else:
            verdict = 'pass'
        print(f'{name:<28} {p:>10.6f}   {verdict}')
    print('-' * 56)
    if report['failures']:
        print(f'FAILED: {", ".join(report["failures"])}')
    elif report['weak']:
        print(f'passed, with weak results in: {", ".join(report["weak"])}')
        print('A weak p-value or two across a dozen tests is expected noise.')
    else:
        print('all tests passed')


def self_test() -> int:
    """Check the battery itself: a good stream should pass, a bad one fail."""
    print('self-test: os.urandom (expected: pass)')
    good = run_battery(os.urandom(1 << 20))
    _print_report(good)

    print()
    print('self-test: a counter (expected: fail loudly)')
    bad = bytes(i & 0xFF for i in range(1 << 20))
    bad_report = run_battery(bad)
    _print_report(bad_report)

    print()
    print('self-test: LSB-stuck stream (expected: fail bit_position_bias)')
    stuck = bytes(b | 1 for b in os.urandom(1 << 19))
    stuck_report = run_battery(stuck)
    _print_report(stuck_report)

    ok = good['passed'] and not bad_report['passed'] and not stuck_report['passed']
    print()
    print('SELF-TEST PASSED' if ok else 'SELF-TEST FAILED')
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('path', nargs='?', help='binary file to test')
    parser.add_argument('--limit', type=int, default=None, help='read at most N bytes')
    parser.add_argument('--quick', action='store_true', help='skip the slow tests')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--self-test', action='store_true',
                        help='verify the battery against known-good and known-bad streams')
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if not args.path:
        parser.error('a file path is required (or use --self-test)')
    if not os.path.exists(args.path):
        print(f'no such file: {args.path}', file=sys.stderr)
        return 1

    with open(args.path, 'rb') as handle:
        data = handle.read(args.limit) if args.limit else handle.read()
    if len(data) < 1024:
        print(f'only {len(data)} bytes; collect more before testing', file=sys.stderr)
        return 1

    report = run_battery(data, quick=args.quick)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
