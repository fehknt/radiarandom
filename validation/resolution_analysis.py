"""Would finer arrival timing buy anything, and what are the channels worth?

Compares count-only, channel-resolved, and time-subdivided min-entropy
against the linear ceiling, using the measured spectra.

Run from the repository root:
    python validation/resolution_analysis.py
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'src'))
os.chdir(_ROOT)
import collections, math
from radiarandom.entropy import window_min_entropy, poisson_min_entropy, LOG2_E, normalise

def load(p):
    raw = open(p, 'rb').read()
    return [int.from_bytes(raw[i:i+2], 'little') for i in range(0, len(raw)-1, 2)]

def spectrum(path, n=1024):
    c = collections.Counter(load(path))
    return normalise([c.get(i, 0) for i in range(n)])

def count_only_min_entropy(rate, T):
    """H_inf if we learned ONLY how many photons arrived, no channel."""
    mu = rate * T
    return poisson_min_entropy(mu)

for name, path, rate in (('background', 'data/soak.channels.u16', 4.4),
                         ('with Am-241', 'data/soak2.channels.u16', 16.2)):
    p = spectrum(path)
    T = 0.5
    print(f'=== {name} @ {rate} counts/s, {T}s device window ===')
    co = count_only_min_entropy(rate, T)
    ch = window_min_entropy(rate, p, T)
    ceiling = rate * T * LOG2_E
    print(f'  count only (no channel resolution) : {co:6.3f} bits/window')
    print(f'  channel-resolved (what we use)     : {ch:6.3f} bits/window   ({ch/co:.2f}x better)')
    print(f'  linear ceiling  rate*T*log2(e)     : {ceiling:6.3f} bits/window')
    print(f'  we are at {100*ch/ceiling:.1f}% of the ceiling')
    print(f'  finer timing: split the window into N sub-windows')
    for N in (1, 2, 10, 100, 1000):
        sub = N * window_min_entropy(rate, p, T/N)
        print(f'    N={N:5d} (dt={1000*T/N:8.3f} ms): {sub:6.3f} bits/window   '
              f'gain vs N=1: {100*(sub/ch-1):+.2f}%')
    print()

print('=== where would finer timing actually pay? ===')
p = spectrum('data/soak2.channels.u16')
print(f'{"rate":>10} {"N=1":>9} {"N=1000":>9} {"gain":>8}   {"% of ceiling at N=1":>20}')
for rate in (16, 100, 1000, 10000, 100000, 1000000):
    a = window_min_entropy(rate, p, 0.5)
    b = 1000 * window_min_entropy(rate, p, 0.0005)
    ceil_ = rate * 0.5 * LOG2_E
    print(f'{rate:>10} {a:>9.1f} {b:>9.1f} {100*(b/a-1):>7.1f}%   {100*a/ceil_:>19.1f}%')