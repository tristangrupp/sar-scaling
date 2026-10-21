"""Run one secondary GPU job politely (exp5 has priority).

  python gpu_job.py <name> <log> -- <command ...>

Rules (coordinator, round 3):
  - one secondary job at a time across exp2 and exp3: hold E:\\SAR_superresolution\\gpu_secondary.lock
    (created with the job name, deleted when the job ends); wait while it exists
  - start only with >= 2 GB VRAM free and exp5's logs advancing (or exp5 not training)
  - stop and requeue if exp5's training logs stop advancing while its trainers are alive, or if this
    job slows to under a third of its own early speed (spill to shared memory)
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
LOCK = PROJ / "gpu_secondary.lock"
EXP5_LOGS = PROJ / "exp5_cross_band" / "logs_r4"
MIN_FREE_MB = 2048
STALL_MIN = 15


def free_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().split(",")
    return int(out[1]) - int(out[0])


def exp5_alive():
    out = subprocess.run(["powershell", "-NoProfile", "-c",
                          "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                          "Where-Object { $_.CommandLine -match 'e5r4_train' }).Count"],
                         capture_output=True, text=True).stdout.strip()
    return int(out or 0) > 0


_growth = {"sizes": None, "t": time.time()}


def exp5_stalled():
    """True when exp5 trainers are alive but none of their logs has grown for STALL_MIN minutes.
    Uses file size, not mtime: Windows does not update the mtime of a log held open for append."""
    logs = list(EXP5_LOGS.glob("*.log")) if EXP5_LOGS.exists() else []
    if not logs or not exp5_alive():
        _growth.update(sizes=None, t=time.time())
        return False
    sizes = {str(f): f.stat().st_size for f in logs}
    if sizes != _growth["sizes"]:
        _growth.update(sizes=sizes, t=time.time())
        return False
    return time.time() - _growth["t"] > STALL_MIN * 60


def step_secs(log):
    """Seconds per 200 steps, from train_sr.py's {'step': n, ..., 'sec': s} records."""
    recs = re.findall(r"'step': (\d+).*?'sec': (\d+)", Path(log).read_text(errors="ignore")) if Path(log).exists() else []
    recs = [(int(a), int(b)) for a, b in recs]
    return [(b2 - b1) / max(a2 - a1, 1) * 200 for (a1, b1), (a2, b2) in zip(recs, recs[1:])]


def run_once(name, log, cmd):
    while True:
        if not LOCK.exists() and free_mb() >= MIN_FREE_MB and not exp5_stalled():
            try:
                fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{name} pid={os.getpid()} started={time.strftime('%Y-%m-%d %H:%M')}\n".encode())
                os.close(fd)
                break
            except FileExistsError:
                pass
        time.sleep(60)
    print(f"{time.strftime('%H:%M')} {name}: lock taken, {free_mb()} MB free", flush=True)
    try:
        with open(log, "w") as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        reason = None
        while proc.poll() is None:
            time.sleep(60)
            sp = step_secs(log)
            if len(sp) >= 6 and min(sp[-3:]) > 3 * min(sp[:3]):      # validation spikes excluded
                reason = f"slowed to {min(sp[-3:]):.0f}s/200 steps (early {min(sp[:3]):.0f}s): possible spill"
            elif exp5_stalled():
                reason = "exp5 logs stopped advancing"
            if reason:
                proc.kill()
                proc.wait()
                break
        return reason, proc.returncode
    finally:
        LOCK.unlink(missing_ok=True)


def main():
    name, log = sys.argv[1], sys.argv[2]
    cmd = sys.argv[sys.argv.index("--") + 1:]
    for attempt in range(6):
        reason, rc = run_once(name, log, cmd)
        if reason is None:
            print(f"{time.strftime('%H:%M')} {name}: finished rc={rc}", flush=True)
            return
        print(f"{time.strftime('%H:%M')} {name}: stopped ({reason}); requeue in 10 min", flush=True)
        time.sleep(600)
    print(f"{name}: gave up after 6 attempts", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
