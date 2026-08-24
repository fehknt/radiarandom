"""Platform integration: the pipe/socket servers and the seed-file writer.

These exercise the transports with a stub reader, so no detector is needed.
The Windows named-pipe server in particular is ctypes code against Win32 and
would otherwise ship untested.
"""

from __future__ import annotations

import os
import socket
import stat
import sys
import threading
import time

import pytest

from radiarandom.pools import service


def counting_reader():
    """A deterministic stand-in for the generator."""
    state = {'n': 0}

    def read(n: int) -> bytes:
        out = bytes((state['n'] + i) & 0xFF for i in range(n))
        state['n'] += n
        return out
    return read


# ------------------------------------------------------------- seed files


def test_write_seed_file_round_trips(tmp_path):
    path = tmp_path / 'seed.bin'
    data = os.urandom(64)
    service.write_seed_file(data, str(path))
    assert path.read_bytes() == data


def test_write_seed_file_survives_newline_bytes(tmp_path):
    """Every byte value must round-trip, including 0x0A and 0x0D.

    Windows opens descriptors in text mode by default, which silently rewrites
    0x0A as 0x0D 0x0A. With random data that corrupts roughly one seed file in
    five and would be invisible without an explicit test.
    """
    path = tmp_path / 'seed.bin'
    data = bytes(range(256)) * 4
    service.write_seed_file(data, str(path))
    written = path.read_bytes()
    assert len(written) == len(data)
    assert written == data


def test_write_seed_file_overwrites_atomically(tmp_path):
    path = tmp_path / 'seed.bin'
    service.write_seed_file(b'a' * 32, str(path))
    service.write_seed_file(b'b' * 32, str(path))
    assert path.read_bytes() == b'b' * 32
    assert not (tmp_path / 'seed.bin.tmp').exists()


def test_write_seed_file_creates_missing_directories(tmp_path):
    path = tmp_path / 'nested' / 'deeper' / 'seed.bin'
    service.write_seed_file(b'x' * 16, str(path))
    assert path.exists()


@pytest.mark.skipif(os.name != 'posix', reason='POSIX permission bits')
def test_write_seed_file_is_not_world_readable(tmp_path):
    path = tmp_path / 'seed.bin'
    service.write_seed_file(b'secret' * 8, str(path))
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode & 0o077 == 0, oct(mode)


# -------------------------------------------------------------------- TCP


def test_tcp_server_streams_bytes():
    server = service.TcpEntropyServer(counting_reader(), host='127.0.0.1', port=0,
                                      chunk_size=256)
    host, port = server.address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                with socket.create_connection((host, port), timeout=5) as sock:
                    received = b''
                    while len(received) < 1024:
                        chunk = sock.recv(4096)
                        if not chunk:
                            break
                        received += chunk
                    assert len(received) >= 1024
                    assert received[:4] == bytes([0, 1, 2, 3])
                    return
            except ConnectionRefusedError:
                time.sleep(0.05)
        pytest.fail('server never accepted a connection')
    finally:
        server.stop()


def test_tcp_server_binds_loopback_by_default():
    server = service.TcpEntropyServer(counting_reader(), port=0)
    try:
        assert server.address[0] == '127.0.0.1'
    finally:
        server.stop()


# ------------------------------------------------------------------ FIFO


@pytest.mark.skipif(os.name != 'posix', reason='FIFOs are POSIX-only')
def test_fifo_server_streams_bytes(tmp_path):
    path = str(tmp_path / 'entropy')
    stop = threading.Event()
    thread = threading.Thread(
        target=service.serve_fifo,
        kwargs={'read': counting_reader(), 'path': path, 'chunk_size': 256, 'stop': stop},
        daemon=True,
    )
    thread.start()
    deadline = time.time() + 5
    while not os.path.exists(path) and time.time() < deadline:
        time.sleep(0.02)
    assert os.path.exists(path)
    assert stat.S_ISFIFO(os.stat(path).st_mode)
    with open(path, 'rb') as handle:
        assert len(handle.read(512)) == 512
    stop.set()


@pytest.mark.skipif(os.name == 'posix', reason='checks the non-POSIX guard')
def test_fifo_server_refuses_on_windows(tmp_path):
    with pytest.raises(RuntimeError, match='POSIX-only'):
        service.serve_fifo(counting_reader(), str(tmp_path / 'x'))


# ---------------------------------------------------------- Windows pipe


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows named pipes')
def test_named_pipe_server_streams_bytes():
    from radiarandom.pools.windows import NamedPipeEntropyServer

    name = r'\\.\pipe\radiarandom-test-{}'.format(os.getpid())
    server = NamedPipeEntropyServer(counting_reader(), pipe_name=name, chunk_size=256)
    ready = threading.Event()
    thread = threading.Thread(
        target=lambda: server.serve_forever(on_ready=lambda _: ready.set()),
        daemon=True,
    )
    thread.start()
    assert ready.wait(5), 'server never signalled ready'

    deadline = time.time() + 10
    received = b''
    while time.time() < deadline:
        try:
            with open(name, 'rb') as pipe:
                while len(received) < 1024:
                    chunk = pipe.read(512)
                    if not chunk:
                        break
                    received += chunk
            break
        except OSError:
            time.sleep(0.1)
    server.stop()

    assert len(received) >= 1024, f'only got {len(received)} bytes'
    assert received[:4] == bytes([0, 1, 2, 3])
    assert server.stats()['clients_served'] >= 1


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows-only')
def test_windows_reports_no_os_pool_support():
    from radiarandom.pools import windows

    assert windows.os_pool_contribution_supported() is False
    explanation = windows.os_pool_explanation()
    assert 'BCRYPT_RNG_USE_ENTROPY_IN_BUFFER' in explanation
    assert 'ignored' in explanation


# ------------------------------------------------------------------ Linux


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Linux-only')
def test_linux_ioctl_constants():
    """RNDADDENTROPY is _IOW('R', 0x03, int[2]); pin the encoding."""
    from radiarandom.pools import linux

    assert linux.RNDADDENTROPY == 0x40085203
    assert linux.RNDGETENTCNT == 0x80045200


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Linux-only')
def test_linux_reports_entropy_avail():
    from radiarandom.pools import linux

    value = linux.entropy_avail()
    assert value is None or value >= 0


@pytest.mark.skipif(sys.platform.startswith('linux'), reason='checks the non-Linux guard')
def test_linux_feeder_refuses_elsewhere():
    from radiarandom.pools import linux

    with pytest.raises(linux.NotLinux, match='Linux-only'):
        linux._require_linux()


@pytest.mark.skipif(
    not sys.platform.startswith('linux') or getattr(os, 'geteuid', lambda: 1)() != 0,
    reason='needs root on Linux for RNDADDENTROPY',
)
def test_linux_kernel_pool_accepts_credited_entropy():
    """The real ioctl, against the real kernel.

    Verified on WSL2 kernel 6.18: can_credit is True and the contribution is
    accepted. Note that entropy_avail does not visibly rise on kernels 5.6 and
    later, where the pool is a fixed 256-bit "initialised" state rather than a
    depleting counter -- so its value is not a useful assertion here.
    """
    from radiarandom.pools import linux

    with linux.KernelPool() as pool:
        assert pool.can_credit is True
        credited = pool.add(os.urandom(32), 256)
        assert credited == 256
        assert pool.stats()['bits_credited'] == 256


@pytest.mark.skipif(
    not sys.platform.startswith('linux') or getattr(os, 'geteuid', lambda: 1)() != 0,
    reason='needs root on Linux for RNDADDENTROPY',
)
def test_linux_never_credits_more_bits_than_the_buffer_holds():
    """Over-crediting the kernel is worse than not contributing at all."""
    from radiarandom.pools import linux

    with linux.KernelPool() as pool:
        assert pool.add(b'\x01\x02\x03\x04', 9999) == 32
