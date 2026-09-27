"""Evict reclaimable file-backed pages from one Windows process working set."""

import argparse
import ctypes
import json
from ctypes import wintypes
from pathlib import Path


PROCESS_SET_QUOTA = 0x0100
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(
        PROCESS_SET_QUOTA | PROCESS_QUERY_LIMITED_INFORMATION, False, args.pid
    )
    if not handle:
        raise ctypes.WinError()
    try:
        if not ctypes.windll.psapi.EmptyWorkingSet(handle):
            raise ctypes.WinError()
        result = {"status": "trimmed", "pid": args.pid}
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output)
    finally:
        kernel32.CloseHandle(handle)


if __name__ == "__main__":
    main()
