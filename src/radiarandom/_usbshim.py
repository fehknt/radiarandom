"""Force PyUSB to use a known-good libusb-1.0 backend.

On Windows the interpreter frequently finds a stray ``libusb-1.0.dll`` on
``PATH`` (shipped by unrelated applications) that PyUSB will happily load and
then crash inside during the very first control transfer -- the failure mode is
an access violation while reading the USB string descriptors, i.e. a hard
segfault with no Python traceback.

``libusb_package`` ships a matched DLL, so we pin PyUSB to that backend before
``radiacode`` gets a chance to call :func:`usb.core.find` (which it does without
passing a backend of its own).

Importing this module is idempotent and is a no-op when ``libusb_package`` is
unavailable, in which case PyUSB's normal discovery applies.
"""

from __future__ import annotations

import logging

_log = logging.getLogger(__name__)

_installed = False


def install() -> bool:
    """Pin PyUSB's default backend. Returns True if the shim was applied."""
    global _installed
    if _installed:
        return True

    try:
        import usb.core
    except ImportError:  # pragma: no cover - pyusb is a hard dependency
        return False

    try:
        import libusb_package
    except ImportError:
        _log.debug('libusb_package not installed; using PyUSB default backend')
        return False

    backend = libusb_package.get_libusb1_backend()
    if backend is None:  # pragma: no cover
        _log.warning('libusb_package returned no backend')
        return False

    original_find = usb.core.find

    def find(*args, **kwargs):
        kwargs.setdefault('backend', backend)
        return original_find(*args, **kwargs)

    find.__wrapped__ = original_find  # type: ignore[attr-defined]
    usb.core.find = find
    _installed = True
    _log.debug('pinned PyUSB to libusb_package backend %r', backend)
    return True
