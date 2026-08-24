"""Stream DRBG output seeded from previously captured detector entropy.

Used by the validation harness when the detector is busy or when a cross-OS
pipe is the throughput bottleneck. The seed is real physical entropy read from
a `radiarandom raw` capture; only the expansion happens here.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from radiarandom.conditioner import HmacDrbg  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else 'data/soak.physical.bin'
with open(path, 'rb') as handle:
    material = handle.read()
if len(material) < 64:
    raise SystemExit(f'{path} holds only {len(material)} bytes; need at least 64')

drbg = HmacDrbg(material[:32], material[32:64], b'radiarandom/validation')
out = sys.stdout.buffer
try:
    while True:
        out.write(drbg.generate(1 << 20))
except (BrokenPipeError, OSError):
    pass
