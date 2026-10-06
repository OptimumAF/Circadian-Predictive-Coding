"""Observe every Windows loaded image and executable process memory region.

Inputs are the current process only. Outputs bind full image files and all
executable memory bytes, including private regions. This does not prove that
memory corresponds to an approved original binary or attest source provenance.
Unsupported platforms fail explicitly; no selected module subset is accepted.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from hashlib import sha256
import os
from pathlib import Path
import stat
import sys
from typing import Any


def whole_runtime_file(path: Path) -> dict[str, Any]:
    path = path.absolute()
    if path.is_symlink() or any(
        getattr(part, "is_junction", lambda: False)() for part in (path, *path.parents)
    ):
        raise ValueError("runtime file cannot be a symlink or junction")
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("runtime file must be regular")
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or len(raw) != after.st_size:
        raise ValueError("runtime file changed during complete read")
    return {
        "path": path.resolve(strict=True).as_posix(),
        "device": after.st_dev,
        "inode": after.st_ino,
        "link_count": after.st_nlink,
        "byte_count": len(raw),
        "sha256": sha256(raw).hexdigest(),
    }


class _MemoryRegion(ctypes.Structure):
    _fields_ = [
        ("base", ctypes.c_void_p),
        ("allocation_base", ctypes.c_void_p),
        ("allocation_protect", wintypes.DWORD),
        ("partition", wintypes.WORD),
        ("size", ctypes.c_size_t),
        ("state", wintypes.DWORD),
        ("protect", wintypes.DWORD),
        ("kind", wintypes.DWORD),
    ]


class WindowsRuntimeImages:
    """Prepare standard native APIs before freezing loaded code membership."""

    def __init__(self) -> None:
        if os.name != "nt" or ctypes.sizeof(ctypes.c_void_p) != 8:
            raise ValueError("complete native runtime observation requires 64-bit Windows")
        self._kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self._process = self._kernel.GetCurrentProcess
        self._process.argtypes, self._process.restype = [], wintypes.HANDLE
        self._handle = self._process()
        self._enumerate = self._kernel.K32EnumProcessModulesEx
        self._enumerate.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.HMODULE),
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            wintypes.DWORD,
        ]
        self._enumerate.restype = wintypes.BOOL
        self._name = self._kernel.K32GetModuleFileNameExW
        self._name.argtypes = [wintypes.HANDLE, wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
        self._name.restype = wintypes.DWORD
        self._query = self._kernel.VirtualQueryEx
        self._query.argtypes = [
            wintypes.HANDLE,
            ctypes.c_void_p,
            ctypes.POINTER(_MemoryRegion),
            ctypes.c_size_t,
        ]
        self._query.restype = ctypes.c_size_t
        self._read = self._kernel.ReadProcessMemory
        self._read.argtypes = [
            wintypes.HANDLE,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self._read.restype = wintypes.BOOL
        # Why this: ctypes creates these helper classes lazily. Create before
        # the Python graph is frozen, so the observer cannot add late classes.
        self._image_buffer = (wintypes.HMODULE * 4096)()
        self._path_buffer = ctypes.create_unicode_buffer(32768)
        self._block_buffer = ctypes.create_string_buffer(1024 * 1024)

    def _images(self) -> list[dict[str, Any]]:
        required = wintypes.DWORD()
        if not self._enumerate(
            self._handle,
            self._image_buffer,
            ctypes.sizeof(self._image_buffer),
            ctypes.byref(required),
            3,
        ):
            raise ValueError("native runtime image enumeration failed")
        if required.value % ctypes.sizeof(wintypes.HMODULE) or required.value > ctypes.sizeof(
            self._image_buffer
        ):
            raise ValueError("native runtime image enumeration would be partial")
        rows = []
        for address in self._image_buffer[: required.value // ctypes.sizeof(wintypes.HMODULE)]:
            length = self._name(self._handle, address, self._path_buffer, len(self._path_buffer))
            if not length or length >= len(self._path_buffer):
                raise ValueError("native runtime image path was unavailable or truncated")
            rows.append(
                {"base": address, "file": whole_runtime_file(Path(self._path_buffer.value))}
            )
        if not rows:
            raise ValueError("native runtime image membership is empty")
        return sorted(rows, key=lambda row: row["base"])

    def _memory_identity(self, address: int, size: int) -> dict[str, Any]:
        digest = sha256()
        offset = 0
        while offset < size:
            count = min(size - offset, len(self._block_buffer))
            observed = ctypes.c_size_t()
            if (
                not self._read(
                    self._handle,
                    address + offset,
                    self._block_buffer,
                    count,
                    ctypes.byref(observed),
                )
                or observed.value != count
            ):
                raise ValueError("native executable memory cannot be completely observed")
            digest.update(self._block_buffer.raw[:count])
            offset += count
        return {"byte_count": size, "sha256": digest.hexdigest()}

    def _executable_regions(self) -> list[dict[str, Any]]:
        rows = []
        address = 0
        executable = {0x10, 0x20, 0x40, 0x80}
        while address < sys.maxsize:
            region = _MemoryRegion()
            observed = self._query(
                self._handle, address, ctypes.byref(region), ctypes.sizeof(region)
            )
            if not observed:
                # ERROR_INVALID_PARAMETER is the documented address-space end.
                if ctypes.get_last_error() == 87:
                    break
                raise ValueError("native address-space enumeration failed")
            base = region.base or 0
            if (
                observed != ctypes.sizeof(region)
                or region.size == 0
                or base + region.size <= address
            ):
                raise ValueError("native address-space enumeration did not advance completely")
            if region.state == 0x1000 and region.protect & 0xFF in executable:
                if region.protect & 0x100:
                    raise ValueError("guarded executable native memory cannot be observed")
                rows.append(
                    {
                        "base": base,
                        "allocation_base": region.allocation_base,
                        "size": region.size,
                        "protect": region.protect,
                        "kind": region.kind,
                        "identity": self._memory_identity(base, region.size),
                    }
                )
            address = base + region.size
        if not rows:
            raise ValueError("native executable memory membership is empty")
        return rows

    def capture(self) -> dict[str, Any]:
        images = self._images()
        regions = self._executable_regions()
        if self._images() != images:
            raise ValueError("native images changed during executable memory observation")
        return {"platform": "windows", "images": images, "executable_regions": regions}
