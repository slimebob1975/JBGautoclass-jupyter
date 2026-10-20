"""Summarize Voilà kernel starts and XSRF-rejected shutdowns from JBG server logs.

This is *diagnostic only*: a 403 does not prove a kernel is still running.
No kernel is stopped or touched by this script.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re

START = re.compile(r"Kernel started:\s*([a-f0-9-]{36})", re.IGNORECASE)
SHUTDOWN_403 = re.compile(r"403 POST /voila/api/shutdown/([a-f0-9-]{36})", re.IGNORECASE)


def summarize(text: str) -> dict:
    started = []
    rejected = []
    for line in text.splitlines():
        match = START.search(line)
        if match:
            started.append(match.group(1).lower())
        match = SHUTDOWN_403.search(line)
        if match:
            rejected.append(match.group(1).lower())
    counts = Counter(rejected)
    unique_started = set(started)
    unique_rejected = set(rejected)
    return {
        "started": len(started),
        "rejected_requests": len(rejected),
        "rejected_kernels": len(unique_rejected),
        "started_without_rejected_shutdown": sorted(unique_started - unique_rejected),
        "rejected_shutdowns": dict(sorted(counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server_log", type=Path)
    args = parser.parse_args()
    report = summarize(args.server_log.read_text(encoding="utf-8-sig", errors="replace"))
    print(f"Kernel starts logged: {report['started']}")
    print(f"Shutdown POSTs rejected (HTTP 403): {report['rejected_requests']}")
    print(f"Distinct kernel IDs with rejected shutdown: {report['rejected_kernels']}")
    print(f"Started IDs without a logged rejected shutdown: {len(report['started_without_rejected_shutdown'])}")
    print("Note: these counts are NOT a count of live or orphaned kernels.")
    print("Check active processes or Jupyter kernel-manager state independently.")
    for kernel_id, count in report["rejected_shutdowns"].items():
        print(f"  rejected {kernel_id}: {count} request(s)")


if __name__ == "__main__":
    main()
