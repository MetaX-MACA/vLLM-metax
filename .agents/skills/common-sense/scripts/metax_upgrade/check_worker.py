#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Supervise a check even when its coordinator is killed by quota/interruption."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from processes import active_group_members, same_process


def main():
    parent = json.loads(sys.argv[1])
    release = Path(sys.argv[2])
    timeout = float(sys.argv[3])
    command = sys.argv[4:]
    result_file = release.parent / "worker-result.json"

    def report(reason, code):
        result_file.write_text(json.dumps({"reason": reason, "returncode": code}))

    waiting_since = time.monotonic()
    while not release.is_file():
        if not same_process(parent) or time.monotonic() - waiting_since > 10:
            return 125
        time.sleep(0.05)
    if not same_process(parent):
        return 125
    child = subprocess.Popen(command)
    started = time.monotonic()
    while child.poll() is None:
        if not same_process(parent) or time.monotonic() - started > timeout:
            report("coordinator_lost" if not same_process(parent) else "timeout", -9)
            os.killpg(os.getpgrp(), signal.SIGKILL)
        time.sleep(0.1)
    remaining = [
        record
        for record in active_group_members(os.getpgrp())
        if record["pid"] != os.getpid()
    ]
    if remaining:
        # The supervisor is still group leader, so these are this check's children.
        for record in remaining:
            if same_process(record):
                try:
                    os.kill(record["pid"], signal.SIGKILL)
                except ProcessLookupError:
                    pass
        report("leftover_processes", 124)
        return 124
    report("completed", child.returncode)
    return child.returncode


if __name__ == "__main__":
    sys.exit(main())
