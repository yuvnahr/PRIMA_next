"""Fail if critical modules lose measured covered branches."""

import json
import sys
from pathlib import Path

BASELINE = {
    "events/event_store.py": 7,
    "memory/embedding_rebuild.py": 31,
    "memory/maintenance/background_supervisor.py": 66,
    "state/state_manager.py": 17,
    "tools/tool_executor.py": 17,
    "workflow/orchestration_engine.py": 78,
}


def check(report_path: Path) -> int:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["meta"]["branch_coverage"] is not True:
        raise ValueError("Expected a branch coverage report")
    counts = {name.replace("\\", "/"): result["summary"]["covered_branches"]
              for name, result in report["files"].items()}
    regressions = []
    for name, minimum in BASELINE.items():
        actual = counts.get(name, 0)
        print(f"{name}  {actual} covered branches (baseline {minimum})")
        if actual < minimum:
            regressions.append(name)
    return int(bool(regressions))


if __name__ == "__main__":
    raise SystemExit(check(Path(sys.argv[1])))
