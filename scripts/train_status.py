#!/usr/bin/env python
"""Live status of every training run under a directory tree.

The sweep runs several seeds concurrently, one per GPU, each writing its own
log. Watching that with ``tail -f`` shows one run at a time and shows nothing at
all about a run that has stopped producing output -- which is the case you most
need to see. This reads the ``status.json`` heartbeat every run writes each
epoch (``src/utils/logging.py::TrainingLogger.update_status``) and prints one
line per run.

    python scripts/train_status.py                     # one snapshot of outputs/
    python scripts/train_status.py --watch             # refresh every 10s
    python scripts/train_status.py --root outputs/full --state running
    python scripts/train_status.py --json              # machine-readable

Liveness is derived, not reported: a run whose heartbeat is older than
``--stale-factor`` times its own mean epoch time is shown as STALLED even though
its own record still says "running". That is the signal a hung dataloader or a
wedged GPU produces, and nothing else surfaces it.

Read-only. It never writes to a run directory.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

STATES = ("running", "stalled", "finished", "failed", "interrupted", "unknown")

# Only used when stdout is a TTY; piping or redirecting yields clean text.
COLORS = {
    "running": "\033[36m",     # cyan
    "stalled": "\033[33m",     # yellow
    "finished": "\033[32m",    # green
    "failed": "\033[31m",      # red
    "interrupted": "\033[35m", # magenta
    "unknown": "\033[37m",     # grey
}
RESET = "\033[0m"
BOLD = "\033[1m"


def _parse_ts(value):
    """Parse an ISO-8601 Z timestamp; return None rather than raising.

    A malformed or missing stamp must not take the monitor down -- it is
    reporting on runs that may themselves be broken.
    """
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None


def _fmt_duration(seconds):
    if seconds is None or seconds < 0:
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _fmt_float(value, spec=".6f"):
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return "-"


def find_runs(root: Path):
    """Every status.json beneath ``root``, newest heartbeat first."""
    runs = []
    for path in sorted(root.rglob("status.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            # A half-written file is expected occasionally even with atomic
            # replacement (a reader can catch the old inode mid-unlink); the
            # next refresh picks it up.
            continue
        if not isinstance(payload, dict):
            continue
        payload["_path"] = path
        runs.append(payload)
    return runs


def classify(run, now, stale_factor):
    """Resolve the run's reported state into a displayed one.

    A run only claims "running"; whether that is still true is decided here from
    how long ago it last wrote, measured against its own epoch time rather than
    a fixed threshold -- arms in this repo range from ~6s to several minutes an
    epoch, so any constant would be wrong for most of them.
    """
    state = str(run.get("state", "unknown")).lower()
    if state not in STATES:
        state = "unknown"
    if state != "running":
        return state, None

    updated = _parse_ts(run.get("updated_at"))
    if updated is None:
        return "unknown", None
    age = (now - updated).total_seconds()

    epoch_time = run.get("epoch_time_s")
    try:
        epoch_time = float(epoch_time)
    except (TypeError, ValueError):
        epoch_time = None
    # Before the first epoch lands there is no epoch time to scale against, so
    # fall back to a generous floor rather than flagging a slow-starting run
    # (data loading and model construction can take minutes).
    budget = max(epoch_time * stale_factor, 60.0) if epoch_time else 900.0

    # A dead process is dead regardless of timing -- but only trust the PID when
    # the heartbeat came from this machine.
    if _pid_is_gone(run):
        return "failed", age
    return ("stalled" if age > budget else "running"), age


def _pid_is_gone(run) -> bool:
    """True only when we can prove the run's process is no longer alive."""
    import socket

    if run.get("host") != socket.gethostname():
        return False
    pid = run.get("pid")
    if not isinstance(pid, int):
        return False
    try:
        os.kill(pid, 0)          # signal 0: existence check, no signal sent
    except ProcessLookupError:
        return True
    except PermissionError:
        return False             # exists, owned by someone else
    except OSError:
        return False
    return False


def build_rows(runs, now, stale_factor):
    rows = []
    for run in runs:
        state, age = classify(run, now, stale_factor)
        epoch = run.get("epoch") or 0
        total = run.get("epochs_total") or 0
        rows.append({
            "state": state,
            "run": run.get("run") or run["_path"].parent.parent.name,
            "seed": run.get("seed", "-"),
            "progress": f"{epoch}/{total}" if total else str(epoch),
            "pct": (epoch / total * 100.0) if total else 0.0,
            "train": run.get("train_loss"),
            "val": run.get("val_loss"),
            "best": run.get("best_val"),
            "best_epoch": run.get("best_epoch", "-"),
            "lr": run.get("lr"),
            "es": f"{run.get('early_stop_counter', '-')}/{run.get('early_stop_patience', '-')}",
            "eta": run.get("eta_s"),
            "epoch_s": run.get("epoch_time_s"),
            "mem": run.get("gpu_mem_peak_mb"),
            "age": age,
            "path": run["_path"],
            "error": run.get("error"),
            "nonfinite": run.get("nonfinite_batches") or 0,
        })
    # Attention first: broken, then stalled, then live, then done.
    order = {"failed": 0, "interrupted": 1, "stalled": 2, "running": 3,
             "unknown": 4, "finished": 5}
    rows.sort(key=lambda r: (order.get(r["state"], 9), r["run"]))
    return rows


def render(rows, color: bool) -> str:
    if not rows:
        return "no runs found (no status.json under the search root)\n"

    header = (
        f"{'STATE':<11} {'RUN':<34} {'EPOCH':>11} {'TRAIN':>10} {'VAL':>10} "
        f"{'BEST':>10} {'LR':>9} {'ES':>7} {'EP/s':>7} {'ETA':>7}"
    )
    lines = [BOLD + header + RESET if color else header, "-" * len(header)]

    for r in rows:
        tint = COLORS.get(r["state"], "") if color else ""
        end = RESET if color else ""
        lines.append(
            f"{tint}{r['state'].upper():<11}{end} "
            f"{r['run'][:34]:<34} "
            f"{r['progress']:>11} "
            f"{_fmt_float(r['train']):>10} "
            f"{_fmt_float(r['val']):>10} "
            f"{_fmt_float(r['best']):>10} "
            f"{_fmt_float(r['lr'], '.2e'):>9} "
            f"{r['es']:>7} "
            f"{_fmt_float(r['epoch_s'], '.1f'):>7} "
            f"{_fmt_duration(r['eta']):>7}"
        )
        # Anything that needs a human goes on its own line rather than being
        # squeezed into a column nobody reads.
        if r["state"] == "stalled":
            lines.append(f"    ^ no heartbeat for {_fmt_duration(r['age'])} "
                         f"(epoch takes {_fmt_float(r['epoch_s'], '.1f')}s) -- {r['path'].parent}")
        if r["error"]:
            lines.append(f"    ^ {r['error']}")
        if r["nonfinite"]:
            lines.append(f"    ^ {r['nonfinite']} non-finite batch(es) in the last epoch")

    counts = {}
    for r in rows:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    lines.append("-" * len(header))
    lines.append("  ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    return "\n".join(lines) + "\n"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs",
                    help="Directory tree to scan for status.json (default: outputs)")
    ap.add_argument("--watch", action="store_true",
                    help="Refresh continuously until interrupted")
    ap.add_argument("--interval", type=float, default=10.0,
                    help="Seconds between refreshes with --watch (default: 10)")
    ap.add_argument("--state", choices=STATES, default=None,
                    help="Show only runs in this state")
    ap.add_argument("--stale-factor", type=float, default=5.0,
                    help="Flag a run STALLED when its heartbeat is older than "
                         "this many of its own epoch times (default: 5)")
    ap.add_argument("--json", action="store_true",
                    help="Emit JSON instead of a table")
    ap.add_argument("--no-color", action="store_true", help="Disable colour")
    return ap.parse_args()


def snapshot(root: Path, stale_factor: float, state_filter=None):
    rows = build_rows(find_runs(root), datetime.now(timezone.utc), stale_factor)
    if state_filter:
        rows = [r for r in rows if r["state"] == state_filter]
    return rows


def main():
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        print(f"no such directory: {root}", file=sys.stderr)
        return 1

    color = not args.no_color and sys.stdout.isatty()

    def once():
        rows = snapshot(root, args.stale_factor, args.state)
        if args.json:
            print(json.dumps(
                [{k: (str(v) if isinstance(v, Path) else v) for k, v in r.items()}
                 for r in rows],
                indent=2, default=str,
            ))
        else:
            print(render(rows, color), end="")
        return rows

    if not args.watch:
        once()
        return 0

    try:
        while True:
            if color:
                print("\033[H\033[J", end="")   # home + clear, no scrollback spam
            print(f"train_status  {root}  "
                  f"{datetime.now().strftime('%H:%M:%S')}  (ctrl-c to exit)\n")
            once()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
