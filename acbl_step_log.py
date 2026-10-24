"""Run one acbl_all.bat step, show its output, and also save it to a log file.

usage: python acbl_step_log.py LOG_FILE script.py [args...]

The step runs with the same Python, unbuffered, with stderr merged into
stdout. Output goes to the console and is appended to LOG_FILE as it
arrives. The exit code is the step's own. On Ctrl+C the step (which gets
the same Ctrl+C) is waited for, and the exit code is non-zero, so
acbl_all.bat's STEP_OK check still stops the pipeline.
"""

import os
import subprocess
import sys


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    log_path, cmd = sys.argv[1], sys.argv[2:]
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    out = sys.stdout.buffer
    with open(log_path, "ab") as log:
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
