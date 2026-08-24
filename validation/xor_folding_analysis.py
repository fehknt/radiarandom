"""Exact information-theoretic accounting for XOR-folding the channel stream.

Shows that H_inf(X xor Y) = H_2(X) for i.i.d. draws, that the operation is
lossy, and that correlated (small-lag) pairs make it lose catastrophically.

Run from the repository root:
    python validation/xor_folding_analysis.py
"""
import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'src'))
os.chdir(_ROOT)
import collections, math
import collections, math

def load(p):
    raw = open(p, 'rb').read()
    return [int.from_bytes(raw[i:i+2], 'little') for i in range(0, len(raw)-1, 2)]

def entropies(counts, n=1024):
    tot = sum(counts.values())
    p = [counts.get(i, 0)/tot for i in range(n)]
    hmin = -math.log2(max(p))
    h2   = -math.log2(sum(x*x for x in p))          # Renyi-2 / collision entropy
    hsh  = -sum(x*math.log2(x) for x in p if x > 0)  # Shannon
    return hmin, h2, hsh, p

def xor_dist(p, n=1024):
    """Exact distribution of X xor Y for X,Y iid ~ p."""
    q = [0.0]*n
    for x in range(n):
        px = p[x]
        if px == 0.0: continue
        for y in range(n):
            if p[y]: q[x ^ y] += px * p[y]
    return q

for name, path in (('background', 'data/soak.channels.u16'),
                   ('with Am-241 source', 'data/soak2.channels.u16')):
    v = load(path)
    c = collections.Counter(v)
    hmin, h2, hsh, p = entropies(c)
    q = xor_dist(p)
    q_hmin = -math.log2(max(q))
    q_h2   = -math.log2(sum(x*x for x in q))
    print(f'=== {name}  ({len(v)} photons) ===')
    print(f'  single photon : H_inf={hmin:.3f}  H_2={h2:.3f}  H_shannon={hsh:.3f}  bits')
    print(f'  X xor Y (iid) : H_inf={q_hmin:.3f}  H_2={q_h2:.3f}  bits')
    print(f'    theory says H_inf(X xor Y) ~ H_2(X) = {h2:.3f}   (predicted {h2:.3f}, got {q_hmin:.3f})')
    print(f'  ENTROPY BUDGET per output symbol:')
    print(f'    consumed : 2 x {hmin:.3f} = {2*hmin:.3f} bits of source min-entropy')
    print(f'    produced : {q_hmin:.3f} bits')
    print(f'    efficiency: {q_hmin/(2*hmin):.1%}   (loss {2*hmin-q_hmin:.3f} bits/pair)')
    # empirical adjacent-lag XOR: what real correlation does
    print(f'  empirical XOR at lag k (real data, includes any correlation):')
    for k in (1, 2, 8, 64, 512):
        if len(v) <= k: continue
        xc = collections.Counter(a ^ b for a, b in zip(v, v[k:]))
        t = sum(xc.values())
        print(f'    lag {k:4d}: H_inf={-math.log2(max(xc.values())/t):.3f} bits')
    print()