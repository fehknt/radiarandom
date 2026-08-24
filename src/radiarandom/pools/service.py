"""Cross-platform ways to hand entropy to other processes.

* :class:`TcpEntropyServer` -- a loopback socket that streams bytes. Works
  everywhere and is the easy way to wire up a container or a VM.
* :func:`serve_fifo` -- a POSIX named pipe, the closest thing to a userspace
  ``/dev/random`` you can offer without a kernel module.
* :func:`write_seed_file` -- a one-shot dump, for boot-time seeding.

All of these are **opt-in**: a consumer has to choose to read from them. None
of them replaces the operating system's own RNG, and none of them should be
used as a drop-in for ``/dev/urandom`` in code you do not control. The right
pattern is to feed the OS pool where the OS allows it (Linux) and to use these
services to seed specific applications where it does not (Windows).
"""

from __future__ import annotations

import logging
import os
import socket
import socketserver
import stat
import threading
from typing import Callable, Optional

_log = logging.getLogger(__name__)

DEFAULT_TCP_HOST = '127.0.0.1'
DEFAULT_TCP_PORT = 7373


class TcpEntropyServer:
    """Streams generator output to any client that connects.

    Binds to loopback by default. Binding anywhere else means shipping your
    key material over the network in the clear, so the CLI refuses to do it
    without an explicit flag.
    """

    def __init__(
        self,
        read: Callable[[int], bytes],
        host: str = DEFAULT_TCP_HOST,
        port: int = DEFAULT_TCP_PORT,
        chunk_size: int = 4096,
    ) -> None:
        self.read = read
        self.chunk_size = chunk_size
        outer = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self) -> None:
                peer = self.client_address
                _log.info('client connected from %s', peer)
                served = 0
                try:
                    while not outer._stop.is_set():
                        self.request.sendall(outer.read(outer.chunk_size))
                        served += outer.chunk_size
                        with outer._lock:
                            outer.bytes_served += outer.chunk_size
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    _log.info('client %s disconnected after %d bytes', peer, served)

        class Server(socketserver.ThreadingTCPServer):
            allow_reuse_address = True
            daemon_threads = True

        self._server = Server((host, port), Handler)
        self._stop = threading.Event()
        self._serving = threading.Event()
        self._closed = False
        self._lock = threading.Lock()
        self.bytes_served = 0

    @property
    def address(self) -> tuple[str, int]:
        return self._server.server_address  # type: ignore[return-value]

    def serve_forever(self, on_ready: Optional[Callable[[tuple], None]] = None) -> None:
        if on_ready is not None:
            on_ready(self.address)
        _log.info('serving entropy on tcp://%s:%d', *self.address)
        self._serving.set()
        try:
            self._server.serve_forever(poll_interval=0.5)
        finally:
            self._serving.clear()
            self._close()

    def _close(self) -> None:
        if not self._closed:
            self._closed = True
            self._server.server_close()

    def stop(self) -> None:
        """Stop serving and release the socket.

        ``shutdown()`` blocks until ``serve_forever`` acknowledges it, and
        deadlocks outright if ``serve_forever`` was never entered -- so only
        call it when the server is actually running. A construct-then-stop
        sequence (an aborted start-up, for instance) must still release the
        listening socket.
        """
        self._stop.set()
        if self._serving.is_set():
            self._server.shutdown()
        else:
            self._close()

    def stats(self) -> dict:
        with self._lock:
            return {'address': self.address, 'bytes_served': self.bytes_served}


def serve_fifo(
    read: Callable[[int], bytes],
    path: str,
    chunk_size: int = 4096,
    mode: int = 0o600,
    stop: Optional[threading.Event] = None,
    on_ready: Optional[Callable[[str], None]] = None,
) -> None:
    """Serve entropy through a POSIX FIFO, recreating it as readers come and go.

    A FIFO write blocks until a reader opens the other end, so this loop simply
    reopens after each reader disconnects. The FIFO is created 0600 so it is
    readable only by its owner.
    """
    if os.name != 'posix':
        raise RuntimeError('FIFOs are POSIX-only; use the named pipe server on Windows')

    stop = stop or threading.Event()
    if os.path.exists(path):
        if not stat.S_ISFIFO(os.stat(path).st_mode):
            raise RuntimeError(f'{path} exists and is not a FIFO')
    else:
        os.mkfifo(path, mode)
    os.chmod(path, mode)
    if on_ready is not None:
        on_ready(path)
    _log.info('serving entropy on FIFO %s', path)

    try:
        while not stop.is_set():
            # Blocks until a reader appears.
            fd = os.open(path, os.O_WRONLY)
            try:
                while not stop.is_set():
                    os.write(fd, read(chunk_size))
            except BrokenPipeError:
                _log.debug('FIFO reader disconnected')
            finally:
                os.close(fd)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def write_seed_file(data: bytes, path: str, mode: int = 0o600) -> None:
    """Write a seed file atomically with restrictive permissions.

    Creates with the final mode from the start rather than chmod-ing after, so
    the contents are never briefly world-readable.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = path + '.tmp'
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, 'O_NOFOLLOW'):
        flags |= os.O_NOFOLLOW
    if hasattr(os, 'O_BINARY'):
        # Windows opens descriptors in *text* mode by default, which rewrites
        # every 0x0A byte as 0x0D 0x0A. On a seed file that is silent
        # corruption: the file grows, the bytes shift, and it only shows up
        # when the random data happens to contain a newline -- about 22% of
        # the time for 64 bytes, which is exactly the kind of bug that ships.
        flags |= os.O_BINARY
    fd = os.open(tmp, flags, mode)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
