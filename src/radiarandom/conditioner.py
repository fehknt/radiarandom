"""Entropy pool, conditioning function, and DRBG.

Pipeline::

    photons -> EntropyPool (HMAC-SHA-512 vetted conditioner)
                    |
                    +-- full-entropy 256-bit blocks  --> "physical" output
                    |
                    +-- seed / reseed --> HmacDrbg(SHA-512) --> "drbg" output

:class:`EntropyPool` absorbs serialised observations and tracks how much
assessed min-entropy they carried. It only produces output once the banked
entropy exceeds the output length by a margin: SP 800-90C allows a vetted
conditioning function to be treated as producing full-entropy output when the
input min-entropy is at least ``output_bits + 64``, so a 256-bit block costs
320 bits of banked entropy. That 1.25x overhead is the price of being able to
call the output full entropy rather than "probably fine".

:class:`HmacDrbg` is SP 800-90A HMAC_DRBG instantiated with SHA-512 at a
256-bit security strength. It exists because the physical source yields on the
order of a couple of bytes per second and almost every real use -- filling a
file, feeding a test suite, answering a ``gen`` request for a megabyte --
needs more than that. Its output is computationally, not information
theoretically, random; the CLI keeps the two modes clearly separated so callers
always know which one they got.
"""

from __future__ import annotations

import hashlib
import hmac
import struct
import time
from typing import Optional

from .device import Batch

#: Key for the HMAC-SHA-512 conditioning function. A conditioning function key
#: is a domain separator, not a secret (SP 800-90B B.2), so it is published.
CONDITIONER_KEY = b'radiarandom/v1 entropy conditioner'

#: Bits of assessed min-entropy required per bit of full-entropy output.
FULL_ENTROPY_MARGIN_BITS = 64

#: Size of one full-entropy output block.
BLOCK_BITS = 256
BLOCK_BYTES = BLOCK_BITS // 8

#: Banked entropy needed to release one block.
BLOCK_COST_BITS = BLOCK_BITS + FULL_ENTROPY_MARGIN_BITS

#: Most min-entropy the pool can actually hold, in bits.
#:
#: The pool is a running HMAC-SHA-512, so its accumulated state *is* a 512-bit
#: chaining value. It cannot carry more entropy than that no matter how much is
#: absorbed, and an earlier version of this class counted credited bits without
#: any limit -- it was observed claiming 2240 banked bits in a 512-bit state,
#: which is simply not true. Anything absorbed beyond the cap still stirs the
#: state but earns no credit.
#:
#: Banking more than this is the reservoir's job (see
#: :class:`radiarandom.generator.Generator`): blocks are *extracted* into a
#: buffer, where each one is genuinely 256 independent bits.
STATE_CAPACITY_BITS = 512


def serialize_batch(batch: Batch) -> bytes:
    """Canonical byte encoding of one observation.

    Everything the host knows about the batch goes in, including the fields we
    assign no entropy to (timestamps, sequence numbers, the device clock).
    Mixing them in can only increase the unpredictability of the pool; the
    accounting in :mod:`radiarandom.entropy` simply refuses to bank anything
    for them.
    """
    header = struct.pack(
        '<QdQIH',
        batch.seq,
        batch.host_monotonic,
        batch.cumulative_total,
        batch.device_seconds & 0xFFFFFFFF,
        batch.count & 0xFFFF,
    )
    body = b''.join(struct.pack('<H', channel) for channel in batch.channels)
    return header + body


class EntropyPool:
    """Accumulates conditioned entropy and releases full-entropy blocks.

    The pool is a running HMAC-SHA-512. Extracting finalises it, splits the
    512-bit digest into a 256-bit output block and a 256-bit chaining value,
    and restarts the pool keyed by the chaining value so that no absorbed
    entropy is ever thrown away.
    """

    def __init__(self, key: bytes = CONDITIONER_KEY) -> None:
        self._key = key
        self._mac = hmac.new(self._key, digestmod=hashlib.sha512)
        self._bits = 0.0
        self._absorbed = 0
        self._blocks_out = 0
        self._total_bits_banked = 0.0
        self._bits_dropped = 0.0

    # ---------------------------------------------------------------- absorb

    def absorb(self, data: bytes, entropy_bits: float = 0.0) -> float:
        """Mix ``data`` into the pool, crediting it with ``entropy_bits``.

        Credit saturates at :data:`STATE_CAPACITY_BITS`, because the pool
        cannot hold more entropy than its 512-bit state. Returns the credit
        actually taken, so callers can tell when entropy is being dropped on
        the floor and extract more eagerly.
        """
        self._mac.update(struct.pack('<I', len(data)))
        self._mac.update(data)
        self._absorbed += len(data)
        if entropy_bits <= 0:
            return 0.0
        headroom = max(0.0, STATE_CAPACITY_BITS - self._bits)
        credited = min(entropy_bits, headroom)
        if credited < entropy_bits:
            self._bits_dropped += entropy_bits - credited
        self._bits += credited
        self._total_bits_banked += credited
        return credited

    def absorb_unaccounted(self, data: bytes) -> None:
        """Mix in data that gets no entropy credit (host jitter, counters)."""
        self.absorb(data, 0.0)

    # --------------------------------------------------------------- extract

    @property
    def entropy_bits(self) -> float:
        return self._bits

    @property
    def blocks_available(self) -> int:
        """Blocks the pool could release right now.

        Capped by the state size, so this is 0 or 1 in practice. Banking more
        than one draw is the reservoir's job, not the pool's.
        """
        return int(self._bits // BLOCK_COST_BITS)

    @property
    def capacity_bits(self) -> float:
        return float(STATE_CAPACITY_BITS)

    @property
    def fill_fraction(self) -> float:
        """How full the pool is toward releasing its next block, 0..1."""
        return min(1.0, self._bits / BLOCK_COST_BITS)

    def ready(self) -> bool:
        return self._bits >= BLOCK_COST_BITS

    def extract_block(self) -> bytes:
        """Release one 256-bit full-entropy block, consuming banked entropy.

        Raises:
            RuntimeError: if the pool has not banked enough entropy yet.
        """
        if not self.ready():
            raise RuntimeError(
                f'pool holds {self._bits:.1f} bits, needs {BLOCK_COST_BITS} '
                f'for a full-entropy block'
            )
        digest = self._mac.digest()
        block, chain = digest[:BLOCK_BYTES], digest[BLOCK_BYTES:]
        self._bits -= BLOCK_COST_BITS
        self._blocks_out += 1
        # Re-key from the chaining half so residual entropy carries forward.
        self._mac = hmac.new(self._key + chain, digestmod=hashlib.sha512)
        self._mac.update(struct.pack('<Q', self._blocks_out))
        return block

    def stats(self) -> dict:
        return {
            'entropy_bits': self._bits,
            'total_bits_banked': self._total_bits_banked,
            'bytes_absorbed': self._absorbed,
            'blocks_released': self._blocks_out,
            'bits_dropped_at_capacity': self._bits_dropped,
            'capacity_bits': STATE_CAPACITY_BITS,
        }


class HmacDrbg:
    """SP 800-90A HMAC_DRBG with SHA-512, 256-bit security strength."""

    OUTLEN = hashlib.sha512().digest_size  # 64
    SECURITY_STRENGTH_BITS = 256
    #: SP 800-90A caps a single generate request; we chunk internally.
    MAX_BYTES_PER_REQUEST = 1 << 16
    #: Requests between *mandatory* reseeds, where generation blocks until the
    #: detector supplies fresh entropy.
    #:
    #: SP 800-90A permits up to 2^48 requests for HMAC_DRBG, so this is still
    #: enormously conservative. It is deliberately not set to something like
    #: 1024: the detector yields a 256-bit block roughly every 26 seconds, so a
    #: tight mandatory interval would stall generation for half a minute every
    #: few tens of megabytes while buying nothing -- the DRBG's security does
    #: not decay with output volume. Reseeding still happens *opportunistically*
    #: every time a fresh block is available (see Generator._maybe_reseed),
    #: which is what actually provides forward and backward secrecy against
    #: state compromise.
    RESEED_INTERVAL = 1 << 20

    def __init__(self, entropy: bytes, nonce: bytes = b'', personalization: bytes = b'') -> None:
        if len(entropy) * 8 < self.SECURITY_STRENGTH_BITS:
            raise ValueError(
                f'need at least {self.SECURITY_STRENGTH_BITS // 8} bytes of '
                f'entropy to instantiate, got {len(entropy)}'
            )
        self._key = b'\x00' * self.OUTLEN
        self._v = b'\x01' * self.OUTLEN
        self._reseed_counter = 0
        self._update(entropy + nonce + personalization)
        self._reseed_counter = 1
        self.reseeds = 0
        self.bytes_generated = 0
        self.last_reseed_monotonic = time.perf_counter()

    def _hmac(self, key: bytes, data: bytes) -> bytes:
        return hmac.new(key, data, hashlib.sha512).digest()

    def _update(self, provided_data: bytes = b'') -> None:
        self._key = self._hmac(self._key, self._v + b'\x00' + provided_data)
        self._v = self._hmac(self._key, self._v)
        if provided_data:
            self._key = self._hmac(self._key, self._v + b'\x01' + provided_data)
            self._v = self._hmac(self._key, self._v)

    def reseed(self, entropy: bytes, additional: bytes = b'') -> None:
        if len(entropy) * 8 < self.SECURITY_STRENGTH_BITS:
            raise ValueError('insufficient entropy for reseed')
        self._update(entropy + additional)
        self._reseed_counter = 1
        self.reseeds += 1
        self.last_reseed_monotonic = time.perf_counter()

    @property
    def needs_reseed(self) -> bool:
        return self._reseed_counter > self.RESEED_INTERVAL

    def generate(self, n_bytes: int, additional: bytes = b'') -> bytes:
        """Produce ``n_bytes`` of DRBG output."""
        if n_bytes <= 0:
            return b''
        out = bytearray()
        remaining = n_bytes
        first = True
        while remaining > 0:
            chunk = min(remaining, self.MAX_BYTES_PER_REQUEST)
            out += self._generate_one(chunk, additional if first else b'')
            remaining -= chunk
            first = False
        self.bytes_generated += n_bytes
        return bytes(out)

    def _generate_one(self, n_bytes: int, additional: bytes) -> bytes:
        if additional:
            self._update(additional)
        temp = bytearray()
        while len(temp) < n_bytes:
            self._v = self._hmac(self._key, self._v)
            temp += self._v
        self._update(additional)
        self._reseed_counter += 1
        return bytes(temp[:n_bytes])

    def stats(self) -> dict:
        return {
            'reseeds': self.reseeds,
            'bytes_generated': self.bytes_generated,
            'requests_since_reseed': self._reseed_counter,
            'seconds_since_reseed': time.perf_counter() - self.last_reseed_monotonic,
        }
