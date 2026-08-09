"""Report optional benchmark capabilities without importing or installing them."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypedDict


@dataclass(frozen=True)
class CapabilitySpec:
    """Modules and dependency group required by one optional capability."""

    name: str
    modules: tuple[str, ...]
    requirement_file: str | None


class CapabilityRecord(TypedDict):
    """Serializable availability result for one optional capability."""

    name: str
    available: bool
    missing: list[str]
    requirement_file: str | None


class CapabilityReport(TypedDict):
    """Versioned benchmark capability report."""

    schema_version: str
    capabilities: list[CapabilityRecord]


CAPABILITIES = (
    CapabilitySpec("rouge_l", (), None),
    CapabilitySpec("bertscore", ("bert_score",), "requirements-semantic-metrics.txt"),
    CapabilitySpec("goemotions_encoder", ("torch", "transformers", "sklearn"), "requirements-encoder.txt"),
)


def module_available(module: str) -> bool:
    """Return whether Python can resolve a module without importing it."""

    return importlib.util.find_spec(module) is not None


def missing_modules(modules: Sequence[str], check: Callable[[str], bool] = module_available) -> tuple[str, ...]:
    """Return unresolved module names without importing or installing them."""

    return tuple(module for module in modules if not check(module))


def capability_report(check: Callable[[str], bool] = module_available) -> CapabilityReport:
    """Build the versioned optional-capability report."""

    records: list[CapabilityRecord] = []
    for capability in CAPABILITIES:
        missing = list(missing_modules(capability.modules, check))
        records.append(
            {
                "name": capability.name,
                "available": not missing,
                "missing": missing,
                "requirement_file": capability.requirement_file,
            }
        )
    return {"schema_version": "1.0", "capabilities": records}


def main() -> None:
    """Print optional benchmark capabilities and exit successfully when absent."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit a versioned JSON report")
    args = parser.parse_args()
    report = capability_report()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    for capability in report["capabilities"]:
        status = "available" if capability["available"] else f"unavailable ({', '.join(capability['missing'])})"
        install = f" [{capability['requirement_file']}]" if capability["requirement_file"] else " [built in]"
        print(f"{capability['name']}: {status}{install}")


if __name__ == "__main__":
    main()
