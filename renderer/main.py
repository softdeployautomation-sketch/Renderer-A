#!/usr/bin/env python3
"""Entry point for the Channelry render worker.

Usage:
    python -m renderer.main            # run the worker loop (heartbeat+claim+render)
    python -m renderer.main --once     # heartbeat + handle a single claim, then exit

All configuration comes from the environment (see renderer/config.py and
.env.example). GitHub Actions sets these via repo secrets — never commit a
value into this repo.
"""
from __future__ import annotations

import argparse
import logging
import sys

from . import config
from .worker import run_batch, run_forever, run_once


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="renderer-a")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Heartbeat + handle one claimed job, then exit (for testing).",
    )
    parser.add_argument(
        "--max-jobs", type=int, default=0,
        help="Render up to N jobs then exit (ephemeral runner batch; 0 = forever).",
    )
    parser.add_argument(
        "--max-seconds", type=float, default=0,
        help="Stop after N seconds (ephemeral runner); 0 = no time bound.",
    )
    parser.add_argument("--debug", action="store_true", help="Verbose logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[render] %(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    try:
        if args.once:
            handled = run_once()
            print("handled_job" if handled else "no_job", file=sys.stderr)
            return 0
        if args.max_jobs > 0:
            run_batch(args.max_jobs, args.max_seconds or float("inf"))
            return 0
        run_forever()
        return 0
    except Exception as exc:  # noqa: BLE001
        # Config errors (missing required env) surface here at startup.
        logging.getLogger("render.main").error("startup failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())