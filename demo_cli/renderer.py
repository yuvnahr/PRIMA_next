"""Restrained terminal rendering; no model calls or runtime state inspection."""

from __future__ import annotations

import os
import re
import shutil
import sys
import textwrap
from collections.abc import Iterable
from typing import TextIO

from demo_cli.config import TraceMode
from demo_cli.diagnostics import DemoTurnDiagnostics


def _terminal_text(text: str) -> str:
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    return "".join(char for char in text if char in "\n\t" or (ord(char) >= 32 and not 127 <= ord(char) <= 159))


class Renderer:
    def __init__(self, *, no_color: bool = False, output: TextIO | None = None, width: int | None = None) -> None:
        self.output = output if output is not None else sys.stdout
        self.width = max(20, width if width is not None else shutil.get_terminal_size((80, 24)).columns)
        self.ansi = (self.output.isatty() and os.getenv("TERM") != "dumb"
                     and (os.name != "nt" or bool(os.getenv("WT_SESSION"))))
        self.color = not no_color and "NO_COLOR" not in os.environ and self.ansi

    def clear(self) -> None:
        if self.ansi:
            print("\033[2J\033[H", end="", file=self.output, flush=True)
        else:
            print("\n" * max(1, shutil.get_terminal_size((80, 24)).lines), file=self.output)

    def message(self, text: str, *, error: bool = False) -> None:
        value = _terminal_text(text)
        if self.color:
            value = f"\033[{31 if error else 90}m{value}\033[0m"
        print(value, file=self.output)

    def rows(self, title: str, rows: Iterable[tuple[str, str]]) -> None:
        self.message(title)
        self.message("-" * min(self.width, 40))
        for label, value in rows:
            for line in textwrap.wrap(
                _terminal_text(value), width=self.width, initial_indent=f"{label:<21} ",
                subsequent_indent=" " * 22, break_long_words=False, break_on_hyphens=False,
            ):
                self.message(line)
        self.message("-" * min(self.width, 40))

    def answer(self, text: str) -> None:
        fenced = False
        for index, line in enumerate(_terminal_text(text).split("\n")):
            prefix = "PRIMA > " if index == 0 else " " * 8
            if line.lstrip().startswith(("```", "~~~")):
                fenced = not fenced
                print(prefix + line, file=self.output)
            elif fenced or line.startswith(("    ", "\t")) or not line:
                print(prefix + line, file=self.output)
            else:
                wrapper = textwrap.TextWrapper(
                    width=self.width, initial_indent=prefix, subsequent_indent=" " * 8,
                    break_long_words=False, break_on_hyphens=False, replace_whitespace=False,
                    drop_whitespace=False, expand_tabs=False,
                )
                print(wrapper.fill(line), file=self.output)

    def error(self, reason: str, *, provider: str | None = None, stage: str | None = None) -> None:
        # Error text may contain echoed authorization values even when no request object is printed.
        reason = re.sub(r"(?i)\bBearer\s+[^\s,;]+", "Bearer [REDACTED]", reason)
        reason = re.sub(r"(?i)\b((?:api[_-]?key|authorization|password|secret|access_token)\s*[:=]\s*)[^\s,;]+", r"\1[REDACTED]", reason)
        self.message("PRIMA request failed", error=True)
        self.rows("", (("Provider", provider or "n/a"), ("Stage", stage or "n/a"), ("Reason", reason[:500])))

    def trace(self, diagnostics: DemoTurnDiagnostics, mode: TraceMode) -> None:
        if mode is TraceMode.OFF:
            return
        self.rows("PRIMA TRACE", diagnostics.summary)
        if mode is TraceMode.FULL:
            self.rows("DETAILS", diagnostics.details)
            total = str(diagnostics.event_count) if diagnostics.event_count is not None else "unknown"
            self.message(f"TRACE EVENTS (showing {len(diagnostics.events)} of {total})")
            for event in diagnostics.events:
                self.message(event)
