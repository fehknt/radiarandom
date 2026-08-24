"""Pool accounting and DRBG behaviour."""

from __future__ import annotations

import os

import pytest

from radiarandom.conditioner import (
    BLOCK_BYTES,
    BLOCK_COST_BITS,
    EntropyPool,
    HmacDrbg,
    serialize_batch,
)
from radiarandom.device import Batch


def make_batch(seq: int, channels: tuple) -> Batch:
    return Batch(seq=seq, host_time=1.0 * seq, host_monotonic=0.5 * seq,
                 device_seconds=seq, channels=channels, count=len(channels),
                 cumulative_total=100 + seq)


# ------------------------------------------------------------------- pool


def test_pool_starts_empty_and_not_ready():
    pool = EntropyPool()
    assert pool.entropy_bits == 0
    assert not pool.ready()


def test_pool_refuses_to_extract_before_it_has_enough():
    pool = EntropyPool()
    pool.absorb(b'some data', 100.0)
    with pytest.raises(RuntimeError, match='needs'):
        pool.extract_block()


def test_pool_releases_a_block_only_after_the_full_cost():
    pool = EntropyPool()
    pool.absorb(b'x', BLOCK_COST_BITS - 1)
    assert not pool.ready()
    pool.absorb(b'y', 1)
    assert pool.ready()
    block = pool.extract_block()
    assert len(block) == BLOCK_BYTES
    assert pool.entropy_bits == pytest.approx(0.0)


def test_pool_charges_each_block_the_full_cost():
    """Two blocks must cost twice as much; no free second block."""
    pool = EntropyPool()
    pool.absorb(b'x', 2 * BLOCK_COST_BITS)
    assert pool.blocks_available == 2
    pool.extract_block()
    assert pool.blocks_available == 1
    pool.extract_block()
    assert pool.blocks_available == 0
    assert not pool.ready()


def test_unaccounted_data_gets_no_credit():
    pool = EntropyPool()
    pool.absorb_unaccounted(os.urandom(4096))
    assert pool.entropy_bits == 0.0
    assert not pool.ready()


def test_successive_blocks_differ():
    pool = EntropyPool()
    blocks = []
    for i in range(8):
        pool.absorb(os.urandom(64), BLOCK_COST_BITS)
        blocks.append(pool.extract_block())
    assert len(set(blocks)) == len(blocks)


def test_pool_is_deterministic_for_identical_input():
    """Same absorbed bytes and credits must give the same block.

    Not a security property -- a sanity check that the conditioner is a
    function of its input and nothing else (no clock, no address, no os.urandom
    sneaking in).
    """
    def run():
        pool = EntropyPool()
        for i in range(4):
            pool.absorb(bytes([i]) * 32, BLOCK_COST_BITS / 2)
        return pool.extract_block()
    assert run() == run()


def test_block_carries_forward_residual_entropy():
    """Extraction rekeys from the chaining half rather than starting over."""
    pool = EntropyPool()
    pool.absorb(b'first', BLOCK_COST_BITS)
    first = pool.extract_block()
    pool.absorb(b'second', BLOCK_COST_BITS)
    second = pool.extract_block()

    fresh = EntropyPool()
    fresh.absorb(b'second', BLOCK_COST_BITS)
    assert second != fresh.extract_block()
    assert second != first


def test_serialize_batch_is_stable_and_covers_the_channels():
    batch = make_batch(3, (1, 2, 1023))
    encoded = serialize_batch(batch)
    assert encoded == serialize_batch(batch)
    assert serialize_batch(make_batch(3, (1, 2, 1022))) != encoded


def test_serialize_batch_distinguishes_multiplicity():
    assert serialize_batch(make_batch(1, (5, 5))) != serialize_batch(make_batch(1, (5,)))


# ------------------------------------------------------------------- DRBG


def test_drbg_requires_a_full_strength_seed():
    with pytest.raises(ValueError, match='at least'):
        HmacDrbg(b'\x00' * 16)


def test_drbg_is_deterministic_given_the_same_seed():
    a = HmacDrbg(b'\x01' * 32, b'\x02' * 32, b'p')
    b = HmacDrbg(b'\x01' * 32, b'\x02' * 32, b'p')
    assert a.generate(256) == b.generate(256)


def test_drbg_differs_on_seed_nonce_and_personalization():
    base = HmacDrbg(b'\x01' * 32, b'\x02' * 32, b'p').generate(64)
    assert HmacDrbg(b'\x03' * 32, b'\x02' * 32, b'p').generate(64) != base
    assert HmacDrbg(b'\x01' * 32, b'\x04' * 32, b'p').generate(64) != base
    assert HmacDrbg(b'\x01' * 32, b'\x02' * 32, b'q').generate(64) != base


def test_drbg_does_not_repeat_itself():
    drbg = HmacDrbg(os.urandom(32), os.urandom(32))
    chunks = [drbg.generate(64) for _ in range(64)]
    assert len(set(chunks)) == 64


def test_drbg_returns_exact_lengths_including_across_the_chunk_boundary():
    drbg = HmacDrbg(os.urandom(32))
    for n in (0, 1, 63, 64, 65, 4096, HmacDrbg.MAX_BYTES_PER_REQUEST + 1):
        assert len(drbg.generate(n)) == n


def test_drbg_reseed_changes_the_stream():
    seed = os.urandom(32)
    a = HmacDrbg(seed, b'n' * 32)
    b = HmacDrbg(seed, b'n' * 32)
    a.generate(64)
    b.generate(64)
    a.reseed(b'\xaa' * 32)
    assert a.generate(64) != b.generate(64)
    assert a.reseeds == 1


def test_drbg_reseed_rejects_a_weak_seed():
    drbg = HmacDrbg(os.urandom(32))
    with pytest.raises(ValueError):
        drbg.reseed(b'\x00' * 8)


def test_drbg_reseed_interval_is_not_absurdly_tight():
    """A tight interval would stall generation against a 1.2 byte/s source."""
    assert HmacDrbg.RESEED_INTERVAL >= 1 << 16


def test_drbg_output_has_no_gross_bias():
    drbg = HmacDrbg(os.urandom(32), os.urandom(32))
    data = drbg.generate(1 << 18)
    ones = sum(bin(byte).count('1') for byte in data)
    total = len(data) * 8
    assert abs(ones / total - 0.5) < 0.01
    assert len(set(data)) == 256
