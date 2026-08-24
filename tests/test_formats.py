"""Output formatting, and above all the unbiasedness of range reduction."""

from __future__ import annotations

import collections
import math
import os

import pytest

from radiarandom import formats


def urandom_reader(n: int) -> bytes:
    return os.urandom(n)


def test_format_round_trips():
    data = bytes(range(16))
    assert bytes.fromhex(formats.format_bytes(data, 'hex')) == data
    import base64
    assert base64.b64decode(formats.format_bytes(data, 'base64')) == data
    assert base64.urlsafe_b64decode(formats.format_bytes(data, 'base64url')) == data


def test_hex_grouping():
    data = bytes(range(8))
    assert formats.format_bytes(data, 'hex', group=4) == '00010203 04050607'
    assert formats.format_bytes(data, 'hex', group=0) == '0001020304050607'


def test_bits_and_dec_formats():
    assert formats.format_bytes(b'\x01\x80', 'bits') == '0000000110000000'
    assert formats.format_bytes(b'\x00\xff', 'dec') == '0 255'


def test_c_format_is_valid_looking():
    text = formats.format_bytes(b'\xde\xad', 'c')
    assert '0xde' in text and '0xad' in text and '[2]' in text


def test_every_advertised_text_format_renders():
    """Anything argparse accepts must actually work.

    'uuid4' was once listed in FORMATS with no implementation behind it, so
    `gen --format uuid4` crashed. This pins the two lists together.
    """
    data = bytes(range(16))
    for fmt in formats.TEXT_FORMATS:
        rendered = formats.format_bytes(data, fmt)
        assert isinstance(rendered, str) and rendered


def test_bin_is_advertised_but_not_text_renderable():
    assert 'bin' in formats.FORMATS
    assert 'bin' not in formats.TEXT_FORMATS
    with pytest.raises(ValueError, match='raw output'):
        formats.format_bytes(bytes([0]), 'bin')


def test_unknown_format_raises():
    with pytest.raises(ValueError):
        formats.format_bytes(b'\x00', 'klingon')


def test_random_below_stays_in_range():
    for bound in (1, 2, 3, 7, 256, 257, 1000, 2 ** 32 + 1):
        for _ in range(50):
            value = formats.random_below(urandom_reader, bound)
            assert 0 <= value < bound


def test_random_below_rejects_bad_bounds():
    with pytest.raises(ValueError):
        formats.random_below(urandom_reader, 0)


def test_random_int_inclusive_bounds():
    values = {formats.random_int(urandom_reader, 5, 7) for _ in range(300)}
    assert values <= {5, 6, 7}
    assert values == {5, 6, 7}


def test_random_int_rejects_inverted_range():
    with pytest.raises(ValueError):
        formats.random_int(urandom_reader, 10, 1)


def test_random_int_single_value_range():
    assert formats.random_int(urandom_reader, 42, 42) == 42


def test_range_reduction_is_unbiased():
    """A modulo would skew a 3-way split; rejection sampling must not.

    Three outcomes drawn from two bits is the classic case where ``% 3`` biases
    the first outcome by 33%. Chi-square over many draws catches that easily.
    """
    counts = collections.Counter(
        formats.random_below(urandom_reader, 3) for _ in range(30000)
    )
    expected = 30000 / 3
    chi2 = sum((counts[value] - expected) ** 2 / expected for value in range(3))
    # 99.9th percentile of chi-square with 2 dof is about 13.8.
    assert chi2 < 13.8, counts


def test_random_float_is_in_unit_interval():
    values = [formats.random_float(urandom_reader) for _ in range(2000)]
    assert all(0.0 <= value < 1.0 for value in values)
    assert 0.4 < sum(values) / len(values) < 0.6


def test_uuid4_has_correct_version_and_variant():
    import uuid
    for _ in range(50):
        text = formats.random_uuid4(urandom_reader)
        parsed = uuid.UUID(text)
        assert parsed.version == 4
        assert (parsed.int >> 62) & 0b11 == 0b10


def test_shuffle_is_a_permutation():
    items = list(range(50))
    shuffled = formats.random_shuffle(urandom_reader, items)
    assert sorted(shuffled) == items
    assert items == list(range(50))  # input untouched


def test_shuffle_actually_moves_things():
    items = list(range(200))
    assert formats.random_shuffle(urandom_reader, items) != items


def test_password_uses_only_the_alphabet():
    alphabet = 'abcXYZ019'
    password = formats.random_password(urandom_reader, 200, alphabet)
    assert len(password) == 200
    assert set(password) <= set(alphabet)


def test_password_rejects_empty_alphabet():
    with pytest.raises(ValueError):
        formats.random_password(urandom_reader, 10, '')


def test_password_entropy_arithmetic():
    assert formats.password_entropy_bits(10, 64) == pytest.approx(60.0)
    assert formats.password_entropy_bits(0, 64) == 0.0
    assert formats.password_entropy_bits(10, 1) == 0.0


def test_unambiguous_alphabet_omits_confusable_characters():
    alphabet = formats.ALPHABETS['unambiguous']
    for character in '0O1lI':
        assert character not in alphabet


def test_all_named_alphabets_have_no_duplicates():
    for name, alphabet in formats.ALPHABETS.items():
        assert len(set(alphabet)) == len(alphabet), name


def test_bytes_needed_for():
    assert formats.bytes_needed_for(1) == 1
    assert formats.bytes_needed_for(256) == 1
    assert formats.bytes_needed_for(257) == 2
    assert formats.bytes_needed_for(1 << 24) == 3
