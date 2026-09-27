from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
from dataclasses import dataclass
from typing import Any

try:
    import resource
except ModuleNotFoundError:  # pragma: no cover - exercised on Windows
    resource = None  # type: ignore[assignment]


class _VmStatistics64(ctypes.Structure):
    _fields_ = [
        ("free_count", ctypes.c_uint32),
        ("active_count", ctypes.c_uint32),
        ("inactive_count", ctypes.c_uint32),
        ("wire_count", ctypes.c_uint32),
        ("zero_fill_count", ctypes.c_uint64),
        ("reactivations", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64),
        ("pageouts", ctypes.c_uint64),
        ("faults", ctypes.c_uint64),
        ("cow_faults", ctypes.c_uint64),
        ("lookups", ctypes.c_uint64),
        ("hits", ctypes.c_uint64),
        ("purges", ctypes.c_uint64),
        ("purgeable_count", ctypes.c_uint32),
        ("speculative_count", ctypes.c_uint32),
        ("decompressions", ctypes.c_uint64),
        ("compressions", ctypes.c_uint64),
        ("swapins", ctypes.c_uint64),
        ("swapouts", ctypes.c_uint64),
        ("compressor_page_count", ctypes.c_uint32),
        ("throttled_count", ctypes.c_uint32),
        ("external_page_count", ctypes.c_uint32),
        ("internal_page_count", ctypes.c_uint32),
        ("total_uncompressed_pages_in_compressor", ctypes.c_uint64),
    ]


class _XswUsage(ctypes.Structure):
    _fields_ = [
        ("xsu_total", ctypes.c_uint64),
        ("xsu_avail", ctypes.c_uint64),
        ("xsu_used", ctypes.c_uint64),
        ("xsu_pagesize", ctypes.c_uint32),
        ("xsu_encrypted", ctypes.c_int32),
    ]


class _ProcessMemoryCounters(ctypes.Structure):
    """Windows PROCESS_MEMORY_COUNTERS without requiring pywin32 or psutil."""

    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _resource_peak_rss_bytes() -> int | None:
    if resource is None:
        return None
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux and the other supported Unix targets report KiB.
    return peak if platform.system() == "Darwin" else peak * 1024


@dataclass(slots=True)
class NativeSnapshot:
    process_rss_bytes: int | None = None
    process_peak_rss_bytes: int | None = None
    system_total_bytes: int | None = None
    system_used_bytes: int | None = None
    system_available_bytes: int | None = None
    system_free_bytes: int | None = None
    compressed_bytes: int | None = None
    swap_used_bytes: int | None = None
    swapout_bytes: int | None = None
    pageout_bytes: int | None = None
    memory_pressure_level: str | None = None
    memory_pressure_value: int | None = None


class MacOSMetrics:
    """Reads macOS memory counters in-process; no per-sample subprocesses."""

    HOST_VM_INFO64 = 4

    def __init__(self) -> None:
        self._lib = ctypes.CDLL(
            ctypes.util.find_library("System") or "libSystem.B.dylib"
        )
        self._page_size = int(os.sysconf("SC_PAGE_SIZE"))
        self._total = self._sysctl_uint64("hw.memsize")
        self._psutil: Any | None = None
        try:
            import psutil

            self._psutil = psutil
            self._process = psutil.Process(os.getpid())
        except ImportError:
            self._process = None

    def _sysctl_raw(self, name: str, value: Any) -> bool:
        size = ctypes.c_size_t(ctypes.sizeof(value))
        result = self._lib.sysctlbyname(
            name.encode(), ctypes.byref(value), ctypes.byref(size), None, 0
        )
        return result == 0

    def _sysctl_uint64(self, name: str) -> int | None:
        value = ctypes.c_uint64()
        return int(value.value) if self._sysctl_raw(name, value) else None

    def _vm_stats(self) -> _VmStatistics64 | None:
        stats = _VmStatistics64()
        count = ctypes.c_uint32(ctypes.sizeof(stats) // ctypes.sizeof(ctypes.c_int))
        host = self._lib.mach_host_self()
        result = self._lib.host_statistics64(
            host, self.HOST_VM_INFO64, ctypes.byref(stats), ctypes.byref(count)
        )
        return stats if result == 0 else None

    def _swap_used(self) -> int | None:
        usage = _XswUsage()
        return int(usage.xsu_used) if self._sysctl_raw("vm.swapusage", usage) else None

    def _pressure(
        self, available: int | None, total: int | None
    ) -> tuple[str | None, int | None]:
        value = ctypes.c_int()
        if self._sysctl_raw("kern.memorystatus_vm_pressure_level", value):
            # Kernel flags are NORMAL=1, WARN=2, CRITICAL=4.
            level = (
                "critical"
                if value.value & 4
                else "warning"
                if value.value & 2
                else "normal"
            )
            return level, int(value.value)
        if available is None or not total:
            return None, None
        ratio = available / total
        return (
            "critical" if ratio < 0.05 else "warning" if ratio < 0.10 else "normal"
        ), None

    def snapshot(self) -> NativeSnapshot:
        vm = self._vm_stats()
        compressed = int(vm.compressor_page_count * self._page_size) if vm else None
        free = int(vm.free_count * self._page_size) if vm else None
        total = self._total
        rss = None
        used = None
        available = None
        if self._psutil is not None:
            try:
                rss = int(self._process.memory_info().rss)
                mem = self._psutil.virtual_memory()
                total, used, available, free = map(
                    int, (mem.total, mem.used, mem.available, mem.free)
                )
            except (OSError, RuntimeError, AttributeError):
                # A process can disappear only in unusual embedding scenarios;
                # the native counters below still produce a useful sample.
                rss = None
        peak = _resource_peak_rss_bytes()
        if used is None and total is not None and free is not None:
            available = free + int(vm.inactive_count * self._page_size) if vm else free
            used = max(0, total - available)
        pressure, pressure_value = self._pressure(available, total)
        return NativeSnapshot(
            process_rss_bytes=rss,
            process_peak_rss_bytes=peak,
            system_total_bytes=total,
            system_used_bytes=used,
            system_available_bytes=available,
            system_free_bytes=free,
            compressed_bytes=compressed,
            swap_used_bytes=self._swap_used(),
            swapout_bytes=(int(vm.swapouts * self._page_size) if vm else None),
            pageout_bytes=(int(vm.pageouts * self._page_size) if vm else None),
            memory_pressure_level=pressure,
            memory_pressure_value=pressure_value,
        )


class PortableMetrics:
    """Best-effort fallback used by tests and non-macOS development."""

    def __init__(self) -> None:
        self._get_process_memory_info: Any | None = None
        self._process_handle: int | None = None
        if os.name == "nt":
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            psapi.GetProcessMemoryInfo.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_ProcessMemoryCounters),
                ctypes.c_ulong,
            ]
            psapi.GetProcessMemoryInfo.restype = ctypes.c_int
            self._process_handle = kernel32.GetCurrentProcess()
            self._get_process_memory_info = psapi.GetProcessMemoryInfo

    def snapshot(self) -> NativeSnapshot:
        peak = _resource_peak_rss_bytes()
        if peak is None and self._get_process_memory_info is not None:
            counters = _ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            if self._get_process_memory_info(
                self._process_handle, ctypes.byref(counters), counters.cb
            ):
                peak = int(counters.PeakWorkingSetSize)
        return NativeSnapshot(process_peak_rss_bytes=peak)


def make_native_metrics() -> MacOSMetrics | PortableMetrics:
    return MacOSMetrics() if platform.system() == "Darwin" else PortableMetrics()
