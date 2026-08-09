"""Command line entry point for unified benchmark campaigns."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from benchmarks.campaign.config import load_campaign_config
from benchmarks.campaign.orchestrator import run_campaign


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one shared-provider benchmark campaign")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    result = run_campaign(load_campaign_config(args.config), resume=args.resume)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
