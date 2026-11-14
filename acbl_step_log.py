"""Run one acbl_all.bat step, show its output, and also save it to a log file.

usage: python acbl_step_log.py LOG_FILE script.py [args...]

The step runs with the same Python, unbuffered, with stderr merged into
stdout. Output goes to the console and is appended to LOG_FILE as it
arrives. The exit code is the step's own. On Ctrl+C the step (which gets
the same Ctrl+C) is waited for, and the exit code is non-zero, so
acbl_all.bat's STEP_OK check still stops the pipeline.

Polars 2 writes spill files to disk when memory runs short. Its default
folder is under %TEMP% on C:, which filled up during step 3a on 2026-11-13
(28 GB free): hundreds of "failed to create spill file ... os error 112"
panics. The step therefore runs with POLARS_OOC_SPILL_DIR set to the first
folder in SPILL_DIRS whose drive has SPILL_MIN_FREE_GB free.

Polars also has its own limit on spill size, POLARS_OOC_DISK_BUDGET_MB.
Going over it fails the query ("query aborted, raise
POLARS_OOC_DISK_BUDGET_MB"), which a full disk did not. So the limit is set
to the drive's free space less SPILL_RESERVE_GB. Either variable is left
alone when it is already set.
"""

import os
import shutil
import subprocess
import sys

SPILL_DIRS = (r"F:\polars-spill",)
SPILL_MIN_FREE_GB = 200
SPILL_RESERVE_GB = 100


def spill_settings(environ, spill_dirs=SPILL_DIRS, min_free_gb=SPILL_MIN_FREE_GB,
                   reserve_gb=SPILL_RESERVE_GB):
    """Returns (variables to add to the step's environment, message)."""
    current = environ.get("POLARS_OOC_SPILL_DIR")
    if current:
        return {}, f"Polars spill folder: {current} (POLARS_OOC_SPILL_DIR already set)"
    for folder in spill_dirs:
        try:
            os.makedirs(folder, exist_ok=True)
            free_gb = shutil.disk_usage(folder).free / 2**30
        except OSError:
            continue
        if free_gb < min_free_gb:
            continue
        added = {"POLARS_OOC_SPILL_DIR": folder}
        message = f"Polars spill folder: {folder} ({free_gb:,.0f} GB free)"
        budget_mb = environ.get("POLARS_OOC_DISK_BUDGET_MB")
        if not budget_mb:
            budget_mb = str(int((free_gb - reserve_gb) * 1024))
            added["POLARS_OOC_DISK_BUDGET_MB"] = budget_mb
        return added, f"{message}, spill limit {int(budget_mb) / 1024:,.0f} GB"
    return {}, (
        f"WARNING: none of {list(spill_dirs)} has {min_free_gb} GB free; "
        "Polars spill files go to its default folder under %TEMP%"
    )


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    log_path, cmd = sys.argv[1], sys.argv[2:]
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    spill_env, spill_message = spill_settings(env)
    env.update(spill_env)
    out = sys.stdout.buffer
    with open(log_path, "ab") as log:
        line = (spill_message + os.linesep).encode("utf-8")
        out.write(line)
        out.flush()
        log.write(line)
        log.flush()
        proc = subprocess.Popen(
            [sys.executable, *cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
        )
        interrupted = False
        while True:
            try:
                chunk = proc.stdout.read1(65536)
            except KeyboardInterrupt:
                interrupted = True  # keep reading until the step itself exits
                continue
            if not chunk:
                break
            out.write(chunk)
            out.flush()
            log.write(chunk)
            log.flush()
        while True:
            try:
                code = proc.wait()
                break
            except KeyboardInterrupt:
                interrupted = True
    if interrupted and code == 0:
        code = 130
    return code


if __name__ == "__main__":
    sys.exit(main())
