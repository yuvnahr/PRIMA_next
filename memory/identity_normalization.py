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
        self._alias_to_canonical: dict[str, str] = {}
        self._last_named_entity: str | None = None

    def observe(self, text: str, speaker: str | None = None) -> IdentityNormalizationResult:
        replacements: dict[str, str] = {}
        if speaker:
            self._remember(speaker, speaker)
            self._last_named_entity = speaker
        for pattern in INTRO_PATTERNS:
            for match in pattern.finditer(text):
                name = match.groupdict().get("name")
                alias = match.groupdict().get("alias") or name
                if name and alias:
                    canonical = self._canonical_for(name)
                    self._remember(canonical, name)
                    self._remember(canonical, alias)
                    replacements[alias] = canonical
                    self._last_named_entity = canonical
        for name in _capitalized_names(text):
            canonical = self._canonical_for(name)
            self._remember(canonical, name)
            replacements[name] = canonical
            self._last_named_entity = canonical
        normalized = self._replace_aliases(text, replacements)
        normalized = self._replace_pronouns(normalized, replacements)
        return IdentityNormalizationResult(text=normalized, replacements=replacements, chains=list(self._chains.values()))

    def chains(self) -> list[IdentityChain]:
        return list(self._chains.values())

    def _canonical_for(self, alias: str) -> str:
        key = alias.lower()
        if key in self._alias_to_canonical:
            return self._alias_to_canonical[key]
        return alias

    def _remember(self, canonical: str, alias: str) -> None:
        chain = self._chains.setdefault(canonical.lower(), IdentityChain(canonical=canonical))
        chain.add(alias)
        self._alias_to_canonical[alias.lower()] = chain.canonical

    def _replace_aliases(self, text: str, replacements: dict[str, str]) -> str:
        normalized = text
        for alias, canonical in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
            if alias == canonical:
                continue
            normalized = re.sub(rf"\b{re.escape(alias)}\b", canonical, normalized)
        return normalized

    def _replace_pronouns(self, text: str, replacements: dict[str, str]) -> str:
        if not self._last_named_entity:
            return text

        def repl(match: re.Match[str]) -> str:
            pronoun = match.group(0)
            replacements[pronoun] = self._last_named_entity or pronoun
            return self._last_named_entity or pronoun

        return re.sub(r"\b(he|him|his|she|her|hers|they|them|their|theirs)\b", repl, text, flags=re.IGNORECASE)


def normalize_identity_text(text: str, speaker: str | None = None) -> IdentityNormalizationResult:
    return IdentityNormalizer().observe(text, speaker=speaker)


def _capitalized_names(text: str) -> list[str]:
    names: list[str] = []
    for token in re.findall(r"\b[A-Z][a-z]+\b", text):
        if token.lower() in {"i", "we", "the", "a", "an"}:
            continue
        names.append(token)
    return list(dict.fromkeys(names))
