"""Windows integration -- and a straight answer about what is possible.

**Windows has no supported API for contributing entropy to the operating
system's RNG.** This is not an oversight in this project; it is how Windows
works:

* ``BCryptGenRandom`` accepts ``BCRYPT_RNG_USE_ENTROPY_IN_BUFFER`` (0x1), which
  once mixed the caller's buffer into the result. Microsoft's own reference for
  the function states, verbatim: *"Windows 8 and later: This flag is ignored in
  Windows 8 and later."* On anything modern it is a no-op.
  https://learn.microsoft.com/en-us/windows/win32/api/bcrypt/nf-bcrypt-bcryptgenrandom
* ``CryptGenRandom`` (the legacy CryptoAPI call that did mix caller data) is
  deprecated and layered over the same CNG DRBG.
* The ``HKLM\\SOFTWARE\\Microsoft\\Cryptography\\RNG\\Seed`` registry value is a
  legacy artefact and is not an entropy input for the modern CNG pool.
* The kernel's entropy gathering lives in ``cng.sys`` and has no user-mode
  contribution interface analogous to Linux's ``RNDADDENTROPY``.

So there is no equivalent of ``rngd`` on Windows, and any tool claiming to
"add entropy to Windows" on a current build is either wrong or is doing what
this module does: **serving entropy to applications that opt in.**

What this module actually provides
----------------------------------
A named-pipe server at ``\\\\.\\pipe\\radiarandom``. Any process that opens the
pipe and reads from it gets detector-backed random bytes. Consumers choose to
use it; nothing is injected into anything.

The pipe is created with ``PIPE_REJECT_REMOTE_CLIENTS`` so it is reachable only
from the local machine, and with the default security descriptor, which grants
access to the creating user and to administrators.

Reading it from PowerShell::

    $pipe = New-Object System.IO.Pipes.NamedPipeClientStream(
        '.', 'radiarandom', [System.IO.Pipes.PipeDirection]::In)
    $pipe.Connect(5000)
    $buffer = New-Object byte[] 32
    $pipe.Read($buffer, 0, 32) | Out-Null
    [BitConverter]::ToString($buffer)

...or from Python::

    with open(r'\\\\.\\pipe\\radiarandom', 'rb') as pipe:
        key = pipe.read(32)
"""

from __future__ import annotations

import ctypes
import logging
import sys
import threading
import time
from ctypes import wintypes
from typing import Callable, Optional

_log = logging.getLogger(__name__)

DEFAULT_PIPE_NAME = r'\\.\pipe\radiarandom'

# --- Win32 constants -------------------------------------------------------
PIPE_ACCESS_OUTBOUND = 0x00000002
PIPE_TYPE_BYTE = 0x00000000
PIPE_READMODE_BYTE = 0x00000000
PIPE_WAIT = 0x00000000
PIPE_REJECT_REMOTE_CLIENTS = 0x00000008
PIPE_UNLIMITED_INSTANCES = 255
INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value
ERROR_PIPE_CONNECTED = 535
ERROR_NO_DATA = 232
ERROR_BROKEN_PIPE = 109

DEFAULT_BUFFER_SIZE = 65536


class NotWindows(RuntimeError):
    pass


def _require_windows() -> None:
    if sys.platform != 'win32':
        raise NotWindows('the named pipe server is Windows-only')


def _kernel32():
    _require_windows()
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

    kernel32.CreateNamedPipeW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
        wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
    ]
    kernel32.CreateNamedPipeW.restype = wintypes.HANDLE

    kernel32.ConnectNamedPipe.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
    kernel32.ConnectNamedPipe.restype = wintypes.BOOL

    kernel32.WriteFile.argtypes = [
        wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p,
    ]
    kernel32.WriteFile.restype = wintypes.BOOL

    kernel32.FlushFileBuffers.argtypes = [wintypes.HANDLE]
    kernel32.FlushFileBuffers.restype = wintypes.BOOL

    kernel32.DisconnectNamedPipe.argtypes = [wintypes.HANDLE]
    kernel32.DisconnectNamedPipe.restype = wintypes.BOOL

    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    return kernel32


def os_pool_contribution_supported() -> bool:
    """Always False on Windows. Present so callers can ask rather than assume."""
    return False


def os_pool_explanation() -> str:
    return (
        'Windows exposes no supported API for adding entropy to the system '
        'RNG: BCRYPT_RNG_USE_ENTROPY_IN_BUFFER is documented as ignored on '
        'Windows 8 and later, and CNG has no user-mode equivalent of Linux '
        'RNDADDENTROPY. Use "radiarandom serve" and have consumers read the '
        'named pipe, or "radiarandom gen" to seed an application directly.'
    )


class NamedPipeEntropyServer:
    """Serves generator output over a local named pipe, one thread per client."""

    def __init__(
        self,
        read: Callable[[int], bytes],
        pipe_name: str = DEFAULT_PIPE_NAME,
        chunk_size: int = 4096,
        max_clients: int = 16,
    ) -> None:
        _require_windows()
        self.read = read
        self.pipe_name = pipe_name
        self.chunk_size = chunk_size
        self.max_clients = max_clients
        self._kernel32 = _kernel32()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.clients_served = 0
        self.bytes_served = 0
        self._lock = threading.Lock()

    def stop(self) -> None:
        self._stop.set()

    def _create_instance(self) -> int:
        handle = self._kernel32.CreateNamedPipeW(
            self.pipe_name,
            PIPE_ACCESS_OUTBOUND,
            PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
            PIPE_UNLIMITED_INSTANCES,
            DEFAULT_BUFFER_SIZE,
            DEFAULT_BUFFER_SIZE,
            0,
            None,
        )
        if handle == INVALID_HANDLE_VALUE:
            raise ctypes.WinError(ctypes.get_last_error())
        return handle

    def _serve_client(self, handle: int) -> None:
        kernel32 = self._kernel32
        written = wintypes.DWORD(0)
        served = 0
        try:
            while not self._stop.is_set():
                data = self.read(self.chunk_size)
                buffer = ctypes.create_string_buffer(data, len(data))
                ok = kernel32.WriteFile(
                    handle, buffer, len(data), ctypes.byref(written), None
                )
                if not ok:
                    err = ctypes.get_last_error()
                    if err in (ERROR_BROKEN_PIPE, ERROR_NO_DATA):
                        break
                    raise ctypes.WinError(err)
                served += written.value
                with self._lock:
                    self.bytes_served += written.value
        except OSError as exc:  # client went away mid-write
            _log.debug('client disconnected: %s', exc)
        finally:
            kernel32.FlushFileBuffers(handle)
            kernel32.DisconnectNamedPipe(handle)
            kernel32.CloseHandle(handle)
            _log.info('client disconnected after %d bytes', served)

    def serve_forever(self, on_ready: Optional[Callable[[str], None]] = None) -> None:
        """Accept clients until :meth:`stop` is called."""
        kernel32 = self._kernel32
        if on_ready is not None:
            on_ready(self.pipe_name)
        _log.info('serving entropy on %s', self.pipe_name)
        while not self._stop.is_set():
            handle = self._create_instance()
            connected = kernel32.ConnectNamedPipe(handle, None)
            if not connected and ctypes.get_last_error() != ERROR_PIPE_CONNECTED:
                kernel32.CloseHandle(handle)
                if self._stop.is_set():
                    break
                time.sleep(0.1)
                continue

            self._threads = [t for t in self._threads if t.is_alive()]
            if len(self._threads) >= self.max_clients:
                _log.warning('client limit (%d) reached; refusing', self.max_clients)
                kernel32.DisconnectNamedPipe(handle)
                kernel32.CloseHandle(handle)
                continue

            with self._lock:
                self.clients_served += 1
            thread = threading.Thread(
                target=self._serve_client, args=(handle,),
                name='radiarandom-pipe-client', daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def stats(self) -> dict:
        with self._lock:
            return {
                'pipe_name': self.pipe_name,
                'clients_served': self.clients_served,
                'bytes_served': self.bytes_served,
                'active_clients': sum(1 for t in self._threads if t.is_alive()),
            }
