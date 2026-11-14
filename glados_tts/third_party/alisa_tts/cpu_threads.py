"""Pin inference thread pools to the CPUs the container may actually use.

In an LXC limited to 2 cores, onnxruntime (Piper) and torch (silero-stress) still size their pools
from the host CPU count and onnxruntime spin-waits between ops; under the cgroup CPU quota that
oversubscription gets throttled. Reproduced locally with `systemd-run --scope -p CPUQuota=200%`:
time to first audio 0.35–0.55 s instead of 0.1–0.2 s, matching the LXC (docs/plan.md amendments).

Thread count: $TTS_THREADS if set, else the smaller of the CPU affinity size and the cgroup v2
`cpu.max` quota. `install()` must run before the first PiperVoice.load (docker/entrypoint.sh).
"""

from __future__ import annotations

import math
import os


def thread_count() -> int:
    if os.environ.get("TTS_THREADS"):
        return max(1, int(os.environ["TTS_THREADS"]))
    n = len(os.sched_getaffinity(0))
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            n = min(n, max(1, math.ceil(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return n


def install() -> int:
    import onnxruntime
    import torch

    n = thread_count()
    torch.set_num_threads(n)
    torch.set_num_interop_threads(1)

    base = onnxruntime.SessionOptions
    if getattr(base, "_cpu_threads", False):
        return n

    class SessionOptions(base):  # piper.voice creates onnxruntime.SessionOptions() itself
        _cpu_threads = True

        def __init__(self) -> None:
            super().__init__()
            self.intra_op_num_threads = n
            self.inter_op_num_threads = 1
            self.add_session_config_entry("session.intra_op.allow_spinning", "0")

    onnxruntime.SessionOptions = SessionOptions
    return n
