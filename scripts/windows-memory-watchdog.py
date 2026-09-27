"""High-priority Windows memory guard for a single llama-server process."""

import argparse
import ctypes
import json
import os
import time
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
PROCESS_TERMINATE = 0x0001
SYNCHRONIZE = 0x00100000
STILL_ACTIVE = 259
HIGH_PRIORITY_CLASS = 0x00000080


def snapshot():
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    return {
        "memory_load_percent": int(status.dwMemoryLoad),
        "total_phys_bytes": int(status.ullTotalPhys),
        "available_phys_bytes": int(status.ullAvailPhys),
        "available_pagefile_bytes": int(status.ullAvailPageFile),
    }


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--minimum-available-gib", type=float, default=12.0)
    parser.add_argument("--maximum-memory-load", type=int, default=82)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()

    kernel32 = ctypes.windll.kernel32
    kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), HIGH_PRIORITY_CLASS)
    handle = kernel32.OpenProcess(
        PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_TERMINATE | SYNCHRONIZE,
        False,
        args.pid,
    )
    if not handle:
        raise ctypes.WinError()

    threshold = int(args.minimum_available_gib * 1024**3)
    log_path = args.run_directory / "memory-watchdog.jsonl"
    result_path = args.run_directory / "memory-watchdog-state.json"
    breaches = 0
    try:
        with log_path.open("a", encoding="utf-8", buffering=1) as log:
            while True:
                exit_code = wintypes.DWORD()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    raise ctypes.WinError()
                if exit_code.value != STILL_ACTIVE:
                    write_json(result_path, {"status": "server_exited", "exit_code": exit_code.value})
                    return

                row = snapshot()
                row["timestamp"] = datetime.now(timezone.utc).isoformat()
                row["pid"] = args.pid
                log.write(json.dumps(row, separators=(",", ":")) + "\n")
                unsafe = (
                    row["available_phys_bytes"] < threshold
                    or row["memory_load_percent"] >= args.maximum_memory_load
                )
                breaches = breaches + 1 if unsafe else 0
                if breaches >= 2:
                    row["status"] = "terminated_for_memory_safety"
                    row["minimum_available_bytes"] = threshold
                    row["maximum_memory_load"] = args.maximum_memory_load
                    write_json(result_path, row)
                    kernel32.TerminateProcess(handle, 200)
                    return
                time.sleep(args.interval)
    finally:
        kernel32.CloseHandle(handle)


if __name__ == "__main__":
    main()
