# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Linux process birth identities prevent cleaning up a reused PID."""

import os
from pathlib import Path
import signal


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {
            "pid": pid,
            "group": int(fields[2]),
            "session": int(fields[3]),
            "start_ticks": fields[19],
        }
    except (OSError, IndexError, ValueError):
        return None


def same_process(record):
    return bool(record) and identity(record["pid"]) == record


def terminate(record):
    if same_process(record) and record["group"] == record["pid"]:
        try:
            os.killpg(record["group"], signal.SIGKILL)
            return True
        except ProcessLookupError:
            pass
    return False


def active_group_members(group):
    members = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        record = identity(int(path.name))
        if record and record["group"] == group and record["session"] == group:
            try:
                status = (path / "stat").read_text().rsplit(")", 1)[1].split()[0]
                if status != "Z":
                    members.append(record)
            except OSError:
                pass
    return members
