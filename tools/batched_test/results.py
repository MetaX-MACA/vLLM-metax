# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Execution results independent of CSV column names and presentation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Status(str, Enum):
    # Task completed and met its success criteria.
    PASSED = "passed"
    # Execution completed but correctness criteria were not met.
    FAILED = "failed"
    # Execution or cleanup raised an error.
    ERROR = "error"
    # Allocation, service readiness, or task execution exceeded a deadline.
    TIMEOUT = "timeout"
    # Execution was cancelled without a timeout.
    CANCELLED = "cancelled"
    # A matching successful result was reused without allocating resources.
    SKIPPED = "skipped"


class Phase(str, Enum):
    # Waiting for or acquiring the resource lease.
    ALLOCATING = "allocating"
    # Planning ranks, starting processes, or waiting for service readiness.
    STARTING = "starting"
    # Running inference cases or the performance sweep.
    TESTING = "testing"
    # Stopping processes and releasing resources; used for cleanup failures.
    CLEANUP = "cleanup"
    # Task execution reached its normal end.
    COMPLETE = "complete"


@dataclass(frozen=True)
class CaseResult:
    # Suite name responsible for this result.
    suite: str
    # Zero-based case index within the suite.
    index: int
    # Whether the case satisfied its scoring criteria.
    passed: bool
    # Model response or embedding similarity summary.
    response: str
    # Explanation of a failed scoring decision; empty on success.
    reason: str = ""


@dataclass
class TaskOutcome:
    # Business result returned by a task before runner cleanup.
    status: Status = Status.PASSED
    # Per-case inference results; empty for performance tasks.
    cases: tuple[CaseResult, ...] = ()
    # Task-level explanation such as an unmet accuracy threshold.
    reason: str = ""
    # Output paths and reproduction metadata consumed by the reporter.
    artifacts: dict[str, Any] = field(default_factory=dict)

    @property
    def correct_ratio(self) -> float:
        return (
            sum(case.passed for case in self.cases) / len(self.cases)
            if self.cases
            else 0.0
        )


@dataclass
class TaskResult:
    # Final result after timeout handling and cleanup.
    status: Status
    # Completion phase, or the phase where execution/cleanup failed.
    phase: Phase
    # Wall-clock seconds including allocation and cleanup, excluding executor queue time.
    elapsed: float
    # Task-produced scores and artifacts; defaults to empty if execution failed early.
    outcome: TaskOutcome = field(default_factory=TaskOutcome)
    # Execution/cancellation/timeout message, independent of cleanup failure.
    error: str = ""
    # Cleanup failure details; nonempty means resources were retained.
    cleanup_error: str = ""
