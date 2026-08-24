"""Output formatting for the command line tool.

Everything here consumes a ``read(n) -> bytes`` callable so the same code
serves both output modes. Integer generation uses rejection sampling: taking
``value % range`` would bias the low end of the interval, which is exactly the
sort of quiet defect a hardware RNG is supposed to avoid.
"""

from __future__ import annotations

import base64
import math
from typing import Callable, Iterator, Sequence

Reader = Callable[[int], bytes]

#: Formats accepted by ``--format``. ``bin`` is handled by the caller, which
#: writes the bytes straight to a binary stream rather than routing them
#: through :func:`format_bytes`; it is listed so argparse accepts it.
FORMATS = ('bin', 'hex', 'base64', 'base64url', 'dec', 'bits', 'c')
#: The subset :func:`format_bytes` can render as text.
TEXT_FORMATS = tuple(f for f in FORMATS if f != 'bin')


def format_bytes(data: bytes, fmt: str, group: int = 0) -> str:
    """Render ``data`` in one of the textual formats.

    ``bin`` is not handled here: raw bytes have no textual rendering, and the
    caller writes them to a binary stream directly.
    """
    if fmt == 'bin':
        raise ValueError(
            "format 'bin' is raw output; write the bytes to a binary stream "
            'instead of calling format_bytes')
    if fmt == 'hex':
        text = data.hex()
        if group:
            text = ' '.join(text[i:i + group * 2] for i in range(0, len(text), group * 2))
        return text
    if fmt == 'base64':
        return base64.b64encode(data).decode('ascii')
    if fmt == 'base64url':
        return base64.urlsafe_b64encode(data).decode('ascii')
    if fmt == 'dec':
        return ' '.join(str(byte) for byte in data)
    if fmt == 'bits':
        return ''.join(format(byte, '08b') for byte in data)
    if fmt == 'c':
        body = ', '.join(f'0x{byte:02x}' for byte in data)
        return f'static const unsigned char random_bytes[{len(data)}] = {{ {body} }};'
    raise ValueError(f'unknown format {fmt!r}')


def bytes_needed_for(bound: int) -> int:
    """Number of whole bytes needed to hold values in ``[0, bound)``."""
    if bound <= 1:
        return 1
    return max(1, (bound - 1).bit_length() + 7 >> 3)


def random_below(read: Reader, bound: int) -> int:
    """Uniform integer in ``[0, bound)`` by rejection sampling.

    Draws ``bit_length(bound-1)`` bits at a time and discards any draw that
    lands outside the interval, so no value is over-represented. The expected
    number of draws is under 2 regardless of ``bound``.
    """
    if bound <= 0:
        raise ValueError('bound must be positive')
    if bound == 1:
        return 0
    bits = (bound - 1).bit_length()
    n_bytes = (bits + 7) // 8
    mask = (1 << bits) - 1
    while True:
        value = int.from_bytes(read(n_bytes), 'big') & mask
        if value < bound:
            return value


def random_int(read: Reader, low: int, high: int) -> int:
    """Uniform integer in the inclusive range ``[low, high]``."""
    if high < low:
        raise ValueError('high must be >= low')
    return low + random_below(read, high - low + 1)


def random_ints(read: Reader, low: int, high: int, count: int) -> Iterator[int]:
    for _ in range(count):
        yield random_int(read, low, high)


def random_float(read: Reader) -> float:
    """Uniform double in ``[0, 1)`` with all 53 mantissa bits set from entropy."""
    value = int.from_bytes(read(7), 'big') >> 3  # 53 bits
    return value / float(1 << 53)


def random_uuid4(read: Reader) -> str:
    """RFC 4122 version 4 UUID built from physical randomness."""
    raw = bytearray(read(16))
    raw[6] = (raw[6] & 0x0F) | 0x40  # version 4
    raw[8] = (raw[8] & 0x3F) | 0x80  # variant 10
    hexed = bytes(raw).hex()
    return f'{hexed[:8]}-{hexed[8:12]}-{hexed[12:16]}-{hexed[16:20]}-{hexed[20:]}'


def random_choice(read: Reader, items: Sequence):
    return items[random_below(read, len(items))]


def random_shuffle(read: Reader, items: list) -> list:
    """Fisher-Yates shuffle driven by the generator."""
    result = list(items)
    for i in range(len(result) - 1, 0, -1):
        j = random_below(read, i + 1)
        result[i], result[j] = result[j], result[i]
    return result


def random_password(read: Reader, length: int, alphabet: str) -> str:
    """Password of ``length`` characters drawn uniformly from ``alphabet``."""
    if not alphabet:
        raise ValueError('alphabet must not be empty')
    return ''.join(random_choice(read, alphabet) for _ in range(length))


def password_entropy_bits(length: int, alphabet_size: int) -> float:
    if alphabet_size <= 1 or length <= 0:
        return 0.0
    return length * math.log2(alphabet_size)


ALPHABETS = {
    'alnum': 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
    'lower': 'abcdefghijklmnopqrstuvwxyz',
    'hex': '0123456789abcdef',
    'digits': '0123456789',
    # Excludes characters that are easy to confuse when transcribed by hand.
    'unambiguous': 'abcdefghjkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789',
    'ascii': (
        'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        '!#$%&()*+,-./:;<=>?@[]^_{|}~'
    ),
}
