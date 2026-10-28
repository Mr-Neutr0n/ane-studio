"""Retain immutable prediction feeds until Core ML releases its borrowed arrays.

This CPython 3.12 workaround avoids final NumPy destruction on Core ML's native
reset queue. Submit fresh arrays and do not mutate them after ``hold()``.
The native GIL fix is upstream: https://github.com/apple/coremltools/pull/2829.
"""

from __future__ import annotations

import atexit
import ctypes
import sys
import time
from collections.abc import Mapping

_owners: dict[int, object] = {}


def _check_runtime() -> None:
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12):
        raise RuntimeError("Core ML input ownership workaround requires CPython 3.12.")


def hold(feed: Mapping[str, object]) -> None:
    """Keep fresh input arrays alive across predictions and model replacement."""
    _check_runtime()
    for array in feed.values():
        address = id(array)
        if address not in _owners:
            _owners[address] = array


def prune() -> int:
    """Release arrays only when this registry is their sole remaining owner.

    Never retrieve values or use sys.getrefcount here: temporary references
    would require extra INCREF/DECREF operations while the native queue can
    decrement the same object's count without the GIL. The registry keeps each
    address valid until deletion on this Python thread.
    """
    _check_runtime()
    released = 0
    for address in tuple(_owners):
        # A subclass destructor can reenter prune and remove another entry.
        if address in _owners and ctypes.c_ssize_t.from_address(address).value == 1:
            del _owners[address]
            released += 1
    return released


def _drain_at_exit() -> None:
    # Best effort only. Never force-release arrays still held by native code.
    # Core ML may keep owners beyond this deadline; the upstream native fix is
    # still required for a general interpreter-shutdown guarantee.
    deadline = time.monotonic() + 2.0
    while _owners:
        prune()
        remaining = deadline - time.monotonic()
        if not _owners or remaining <= 0:
            return
        time.sleep(min(0.01, remaining))


atexit.register(_drain_at_exit)
