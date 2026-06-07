"""Command-line entrypoint for quick PRIMA-NEXT affect checks."""

from __future__ import annotations

import json

from affect import DynamicAffectEngine


def main() -> None:
    engine = DynamicAffectEngine()
    print("PRIMA-NEXT Affect Subsystem")
    print("Enter text to process, or an empty line to exit.")

    while True:
        text = input("> ").strip()
        if not text:
            break
        update = engine.process(text)
        print(json.dumps(update.to_dict(), indent=2))


if __name__ == "__main__":
    main()
