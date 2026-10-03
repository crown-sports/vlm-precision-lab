"""Run one experiment with a durable exit record (including killed children)."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--record", type=Path, required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    args = a.command[1:] if a.command and a.command[0] == "--" else a.command
    if not args or a.record.exists():
        p.error("Supply a command and a new record filename")
    a.record.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    completed = subprocess.run(args)
    record = {"command": args, "exit_code": completed.returncode, "seconds": time.monotonic()-start,
              "finished_at": datetime.now(timezone.utc).isoformat(), "state": "success" if completed.returncode == 0 else "failed"}
    a.record.write_text(json.dumps(record, indent=2) + "\n")
    raise SystemExit(completed.returncode if completed.returncode >= 0 else 128 - completed.returncode)


if __name__ == "__main__":
    main()
