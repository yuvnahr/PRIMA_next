"""Resource-backed query analysis and deterministic expansion for retrieval."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']+|\d{4}")
NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
PRONOUNS = {"he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs"}

STOPWORDS = {
    "a", "an", "and", "are", "did", "do", "does", "for", "from", "has", "have", "how", "is", "of",
    "on", "or", "the", "to", "was", "were", "what", "when", "where", "which", "who", "whom", "why",
}
RESOURCE_DIR = Path(__file__).resolve().parent / "resources"


@dataclass(frozen=True, slots=True)
class TemporalConstraint:
    operator: str
    anchor: str | None = None
    expression: str = ""

    def to_dict(self) -> dict[str, str | None]:
        return {"operator": self.operator, "anchor": self.anchor, "expression": self.expression}


@dataclass(frozen=True, slots=True)
class QueryAnalysis:
    entities: tuple[str, ...] = ()
    relations: tuple[str, ...] = ()
    temporal_expressions: tuple[str, ...] = ()
    preference_terms: tuple[str, ...] = ()
    identity_attributes: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    intents: tuple[str, ...] = ()
    events: tuple[str, ...] = ()
    temporal_constraints: tuple[TemporalConstraint, ...] = ()
    aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    pronouns: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": list(self.entities),
            "relations": list(self.relations),
            "temporal_expressions": list(self.temporal_expressions),
            "preference_terms": list(self.preference_terms),
            "identity_attributes": list(self.identity_attributes),
            "keywords": list(self.keywords),
            "intents": list(self.intents),
            "events": list(self.events),
            "temporal_constraints": [constraint.to_dict() for constraint in self.temporal_constraints],
            "aliases": {key: list(value) for key, value in self.aliases.items()},
            "pronouns": list(self.pronouns),
        }


@dataclass(frozen=True, slots=True)
class ExpandedQuery:
    text: str
    terms: tuple[str, ...] = ()
    expansion_map: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "terms": list(self.terms),
            "expansion_map": {key: list(value) for key, value in self.expansion_map.items()},
        }


class QueryAnalyzer:
    """Rule-based semantic parser kept independent of benchmark code."""

    def analyze(self, query: str, entity_context: dict[str, Any] | None = None) -> QueryAnalysis:
        lower_query = query.lower()
        tokens = [token.lower().strip("'") for token in TOKEN_RE.findall(query)]
        token_set = set(tokens)
        intent_matches = self._detect_intents(lower_query, token_set)
        relation_terms = self._concept_terms("relationship", lower_query, token_set, intent_matches)
        preference_terms = self._concept_terms("preference", lower_query, token_set, intent_matches)
        identity_terms = self._concept_terms("identity", lower_query, token_set, intent_matches)
        events = self._concept_terms("event", lower_query, token_set, intent_matches)
        temporal_expressions, temporal_constraints = self._temporal(query, lower_query, token_set, intent_matches)
        entities, aliases, pronouns = self._entities(query, tokens, entity_context or {})
        keywords = tuple(dict.fromkeys(token for token in tokens if len(token) > 2 and token not in STOPWORDS))
        return QueryAnalysis(
            entities=entities,
            relations=relation_terms,
            temporal_expressions=temporal_expressions,
            preference_terms=preference_terms,
            identity_attributes=identity_terms,
            keywords=keywords,
            intents=intent_matches,
            events=events,
            temporal_constraints=temporal_constraints,
            aliases=aliases,
            pronouns=pronouns,
        )

    def expand(self, query: str, analysis: QueryAnalysis) -> ExpandedQuery:
        relation_expansions = _resource("relation_expansions.json")
        lexical_expansions = _resource("lexical_expansions.json")
        temporal_expansions = _resource("temporal_patterns.json")
        intent_patterns = _resource("intent_patterns.json")
        expansion_map: dict[str, tuple[str, ...]] = {}

        for term in (*analysis.keywords, *analysis.relations, *analysis.preference_terms, *analysis.identity_attributes, *analysis.events):
            values = _lookup(term, relation_expansions) + _lookup(term, lexical_expansions) + _lookup(term, temporal_expansions)
            if values:
                expansion_map[term] = tuple(dict.fromkeys(values))

        for intent in analysis.intents:
            intent_config = intent_patterns.get(intent, {})
            values = intent_config.get("expansions", []) if isinstance(intent_config, dict) else []
            if values:
                expansion_map[f"intent:{intent}"] = tuple(str(value).lower() for value in values)

        for constraint in analysis.temporal_constraints:
            values = _lookup(constraint.operator, temporal_expansions)
            if constraint.anchor:
                values.extend(_lookup(constraint.anchor, lexical_expansions))
            if values:
                expansion_map[f"temporal:{constraint.operator}"] = tuple(dict.fromkeys(values))

        expansion_terms = tuple(dict.fromkeys(term for values in expansion_map.values() for term in values))
        expanded_text = " ".join((query, *expansion_terms)).strip()
        return ExpandedQuery(text=expanded_text, terms=expansion_terms, expansion_map=expansion_map)

    def _detect_intents(self, lower_query: str, token_set: set[str]) -> tuple[str, ...]:
        matches: list[str] = []
        for intent, config in _resource("intent_patterns.json").items():
            patterns = config.get("patterns", []) if isinstance(config, dict) else []
            for pattern in patterns:
                pattern_text = str(pattern).lower()
                if " .*" in pattern_text or pattern_text.startswith("where ") or pattern_text.startswith("who "):
                    if re.search(pattern_text, lower_query):
                        matches.append(str(intent))
                        break
                elif pattern_text in lower_query or pattern_text in token_set:
                    matches.append(str(intent))
                    break
        return tuple(dict.fromkeys(matches))

    def _concept_terms(self, concept: str, lower_query: str, token_set: set[str], intents: tuple[str, ...]) -> tuple[str, ...]:
        config = _resource("intent_patterns.json").get(concept, {})
        patterns = config.get("patterns", []) if isinstance(config, dict) else []
        terms: list[str] = []
        for pattern in patterns:
            value = str(pattern).lower()
            if " .*" in value:
                if re.search(value, lower_query):
                    terms.append(value)
            elif value in lower_query or value in token_set:
                terms.append(value)
        if concept in intents:
            terms.append(concept)
        return tuple(dict.fromkeys(_canonical_term(term) for term in terms))

    def _temporal(
        self,
        query: str,
        lower_query: str,
        token_set: set[str],
        intents: tuple[str, ...],
    ) -> tuple[tuple[str, ...], tuple[TemporalConstraint, ...]]:
        expressions: list[str] = []
        constraints: list[TemporalConstraint] = []
        phrase_patterns = {
            "before": ("before moving", "before marriage", "before college", "before graduation"),
            "after": ("after graduation", "after college", "after marriage", "after moving"),
            "during": ("during vacation", "during college", "during school"),
            "while": ("while working", "when living in", "while living in"),
        }
        for operator, phrases in phrase_patterns.items():
            for phrase in phrases:
                if phrase in lower_query:
                    expressions.append(phrase)
                    constraints.append(TemporalConstraint(operator=operator, anchor=phrase.split()[-1], expression=phrase))
        for operator in ("before", "after", "during", "while", "first", "last", "earliest", "latest", "earlier", "later"):
            if operator in token_set:
                expressions.append(operator)
                constraints.append(TemporalConstraint(operator=_canonical_temporal_operator(operator), expression=operator))
        for phrase in ("last week", "last month", "last year", "next week", "next month", "earlier that summer"):
            if phrase in lower_query:
                expressions.append(phrase)
        expressions.extend(token for token in sorted(token_set) if token.isdigit() and len(token) == 4)
        if "temporal" in intents and not constraints:
            constraints.append(TemporalConstraint(operator="temporal", expression="temporal"))
        return tuple(dict.fromkeys(expressions)), tuple(dict.fromkeys(constraints))

    def _entities(
        self,
        query: str,
        tokens: list[str],
        entity_context: dict[str, Any],
    ) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]], tuple[str, ...]]:
        aliases: dict[str, tuple[str, ...]] = {}
        entities: list[str] = [
            match.group(0)
            for match in NAME_RE.finditer(query)
            if match.group(0).lower() not in STOPWORDS and match.group(0).lower() not in {"who", "what", "where", "when"}
        ]
        configured_aliases = entity_context.get("aliases", {}) if isinstance(entity_context, dict) else {}
        if isinstance(configured_aliases, dict):
            for canonical, values in configured_aliases.items():
                alias_values = tuple(str(value) for value in values) if isinstance(values, list | tuple) else (str(values),)
                aliases[str(canonical)] = alias_values
                if any(alias.lower() in query.lower() for alias in alias_values):
                    entities.append(str(canonical))
        for entity in list(entities):
            parts = entity.split()
            if len(parts) > 1:
                aliases.setdefault(entity, tuple(dict.fromkeys((parts[0], parts[-1], entity))))
        recent_entities = entity_context.get("recent_entities", ()) if isinstance(entity_context, dict) else ()
        pronouns = tuple(dict.fromkeys(token for token in tokens if token in PRONOUNS))
        if pronouns and recent_entities:
            entities.extend(str(entity) for entity in recent_entities if entity)
        return tuple(dict.fromkeys(entities)), aliases, pronouns


def temporal_agreement(analysis: QueryAnalysis, timestamp: datetime | None) -> float:
    """Return a conservative temporal match signal for confidence scoring."""

    if not analysis.temporal_expressions and not analysis.temporal_constraints:
        return 0.5
    if timestamp is None:
        return 0.0
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    terms = set(analysis.temporal_expressions)
    if str(timestamp.year) in terms:
        return 1.0
    age_days = max(0.0, (now - timestamp).total_seconds() / 86400.0)
    if "today" in terms and age_days <= 1:
        return 1.0
    if "yesterday" in terms and 1 <= age_days <= 2:
        return 1.0
    if "last week" in terms and age_days <= 7:
        return 1.0
    if "last month" in terms and age_days <= 31:
        return 1.0
    operators = {constraint.operator for constraint in analysis.temporal_constraints}
    if operators & {"first", "last", "before", "after", "during", "while", "temporal"}:
        return 0.6
    return 0.25


@lru_cache(maxsize=None)
def _resource(filename: str) -> dict[str, Any]:
    path = RESOURCE_DIR / filename
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {}


def _lookup(term: str, resource: dict[str, Any]) -> list[str]:
    value = resource.get(term.lower()) or resource.get(_canonical_term(term))
    if isinstance(value, list):
        return [str(item).lower() for item in value]
    return []


def _canonical_term(term: str) -> str:
    value = term.lower().strip()
    if value.endswith("ies"):
        return f"{value[:-3]}y"
    if value.endswith("s") and len(value) > 4:
        return value[:-1]
    return value


def _canonical_temporal_operator(operator: str) -> str:
    return {"earliest": "first", "latest": "last", "earlier": "before", "later": "after"}.get(operator, operator)

