"""Deterministic identity normalization for memory embedding text."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

PRONOUNS = {"he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs"}
INTRO_PATTERNS = (
    re.compile(r"\bmy name is (?P<name>[A-Z][a-z]+)\b"),
    re.compile(r"\bi am (?P<name>[A-Z][a-z]+)\b"),
    re.compile(r"\bi'm (?P<name>[A-Z][a-z]+)\b"),
    re.compile(r"\bcall me (?P<name>[A-Z][a-z]+)\b"),
    re.compile(r"\b(?P<name>[A-Z][a-z]+) goes by (?P<alias>[A-Z][a-z]+)\b"),
    re.compile(r"\b(?P<name>[A-Z][a-z]+) is called (?P<alias>[A-Z][a-z]+)\b"),
)


@dataclass(slots=True)
class IdentityChain:
    canonical: str
    aliases: set[str] = field(default_factory=set)

    def add(self, alias: str) -> None:
        clean = alias.strip()
        if clean:
            self.aliases.add(clean)

    def to_dict(self) -> dict[str, object]:
        return {"canonical": self.canonical, "aliases": sorted(self.aliases)}


@dataclass(slots=True)
class IdentityNormalizationResult:
    text: str
    replacements: dict[str, str]
    chains: list[IdentityChain]

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "replacements": dict(sorted(self.replacements.items())),
            "chains": [chain.to_dict() for chain in self.chains],
        }


class IdentityNormalizer:
    """Resolve high-confidence aliases without benchmark-specific knowledge."""

    def __init__(self) -> None:
        self._chains: dict[str, IdentityChain] = {}

    def observe(self, text: str, speaker: str | None = None) -> IdentityNormalizationResult:
        chains: dict[str, IdentityChain] = {}
        claims: dict[str, set[str]] = {}
        if speaker:
            chains[speaker] = IdentityChain(speaker, {speaker})
        for pattern in INTRO_PATTERNS:
            for match in pattern.finditer(text):
                name = match["name"]
                alias = match.groupdict().get("alias") or name
                chains.setdefault(name, IdentityChain(name, {name})).add(alias)
                claims.setdefault(alias, set()).add(name)
        replacements = {alias: next(iter(names)) for alias, names in claims.items() if len(names) == 1}
        self._chains = chains
        return IdentityNormalizationResult(self._replace_aliases(text, replacements), replacements, list(chains.values()))

    def chains(self) -> list[IdentityChain]:
        return list(self._chains.values())

    def _replace_aliases(self, text: str, replacements: dict[str, str]) -> str:
        normalized = text
        for alias, canonical in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
            if alias == canonical:
                continue
            normalized = re.sub(rf"\b{re.escape(alias)}\b", canonical, normalized)
        return normalized

def normalize_identity_text(text: str, speaker: str | None = None) -> IdentityNormalizationResult:
    return IdentityNormalizer().observe(text, speaker=speaker)
