"""Strict LoCoMo-to-conversation adaptation."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from benchmarks.common.interfaces import Conversation, ConversationQuestion, ConversationTurn

EVIDENCE_ID_RE = re.compile(r"D\d+:\d+")
TIMESTAMP_FORMATS = (
    "%I:%M %p on %d %B, %Y",
    "%I:%M %p on %B %d, %Y",
    "%d %B %Y %I:%M %p",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
)
VALID_CATEGORIES = frozenset({"1", "2", "3", "4", "5"})
KNOWN_EVIDENCE_CORRECTIONS = {
    ("conv-42", "D10:19"): "D20:15",
    ("conv-42", "D"): None,
    ("conv-43", "D:11:26"): "D11:26",
    ("conv-47", "D4:36"): "D13:3",
    ("conv-50", "D30:05"): "D30:5",
}


def parse_timestamp(value: str, *, location: str = "timestamp") -> datetime:
    """Parse known LoCoMo or ISO timestamps as UTC-aware datetimes."""
    text = value.strip()
    if not text:
        raise ValueError(f"{location}: timestamp must not be empty")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        parsed = None
    if parsed is None:
        for pattern in TIMESTAMP_FORMATS:
            try:
                parsed = datetime.strptime(text, pattern)
                break
            except ValueError:
                continue
    if parsed is None:
        raise ValueError(f"{location}: unsupported timestamp {value!r}")
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).astimezone(timezone.utc)


class LoCoMoAdapter:
    """Validate raw LoCoMo records and expose benchmark-neutral conversations."""

    def adapt(self, raw_data: Any) -> list[Conversation]:
        records = raw_data if isinstance(raw_data, list) else raw_data.get("data") if isinstance(raw_data, dict) else None
        if not isinstance(records, list):
            raise ValueError("LoCoMo root must be a list or contain a 'data' list")
        conversations = []
        seen = set()
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"LoCoMo record {index}: expected object")
            conversation = self._adapt_record(record, index)
            if conversation.id in seen:
                raise ValueError(f"LoCoMo record {index}: duplicate conversation ID {conversation.id!r}")
            seen.add(conversation.id)
            conversations.append(conversation)
        return conversations

    def _adapt_record(self, record: dict[str, Any], index: int) -> Conversation:
        conversation_id = str(record.get("sample_id") or record.get("id") or "").strip()
        if not conversation_id:
            raise ValueError(f"LoCoMo record {index}: missing sample_id/id")
        payload = record.get("conversation")
        if not isinstance(payload, dict):
            raise ValueError(f"LoCoMo conversation {conversation_id}: 'conversation' must be an object")
        qa = record.get("qa")
        if not isinstance(qa, list):
            raise ValueError(f"LoCoMo conversation {conversation_id}: 'qa' must be a list")
        speakers = {}
        for key in ("speaker_a", "speaker_b"):
            value = payload.get(key, record.get(key))
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"LoCoMo conversation {conversation_id}: {key} must be a non-empty string")
            speakers[key] = value.strip()
        turns, turn_ids = self._normalize_turns(payload, conversation_id)
        questions = self._normalize_questions(qa, conversation_id, turn_ids)
        metadata = {key: value for key, value in record.items() if key not in {"conversation", "qa"}}
        metadata["speakers"] = speakers
        metadata["session_count"] = len({turn.session_id for turn in turns})
        return Conversation(conversation_id, tuple(turns), tuple(questions), metadata)

    def _normalize_turns(
        self, payload: dict[str, Any], conversation_id: str,
    ) -> tuple[list[ConversationTurn], set[str]]:
        turns, ids = [], set()
        for session_id in self._session_keys(payload):
            session_turns = payload[session_id]
            if not isinstance(session_turns, list) or not session_turns:
                raise ValueError(
                    f"LoCoMo conversation {conversation_id} {session_id}: session must be a non-empty list"
                )
            timestamp_value = payload.get(f"{session_id}_date_time")
            if not isinstance(timestamp_value, str):
                raise ValueError(f"LoCoMo conversation {conversation_id} {session_id}: missing timestamp")
            timestamp = parse_timestamp(
                timestamp_value, location=f"LoCoMo conversation {conversation_id} {session_id}",
            ).isoformat()
            for turn_index, turn in enumerate(session_turns):
                location = f"LoCoMo conversation {conversation_id} {session_id} turn {turn_index}"
                if not isinstance(turn, dict):
                    raise ValueError(f"{location}: expected object")
                speaker, text = turn.get("speaker"), turn.get("text")
                turn_id = str(turn.get("dia_id") or "").strip()
                if not isinstance(speaker, str) or not speaker.strip():
                    raise ValueError(f"{location}: speaker must be a non-empty string")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"{location}: text must be a non-empty string")
                if not EVIDENCE_ID_RE.fullmatch(turn_id):
                    raise ValueError(f"{location}: invalid dia_id {turn_id!r}")
                if turn_id in ids:
                    raise ValueError(f"{location}: duplicate dia_id {turn_id!r}")
                ids.add(turn_id)
                metadata = {key: value for key, value in turn.items() if key not in {"speaker", "text", "dia_id"}}
                metadata.update(source_turn_id=turn_id, source_session_id=session_id, source_timestamp=timestamp)
                turns.append(ConversationTurn(speaker.strip(), text.strip(), turn_id, session_id, timestamp, metadata))
        return turns, ids

    def _normalize_questions(
        self, qa: list[Any], conversation_id: str, turn_ids: set[str],
    ) -> list[ConversationQuestion]:
        questions, ids = [], set()
        for index, raw in enumerate(qa):
            location = f"LoCoMo conversation {conversation_id} question {index}"
            if not isinstance(raw, dict):
                raise ValueError(f"{location}: expected object")
            question = raw.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"{location}: question must be a non-empty string")
            category = str(raw.get("category", "")).strip()
            if category not in VALID_CATEGORIES:
                raise ValueError(f"{location}: category must be one of {sorted(VALID_CATEGORIES)}")
            question_id = str(raw.get("question_id") or raw.get("id") or index).strip()
            if not question_id or question_id in ids:
                raise ValueError(f"{location}: missing or duplicate question ID {question_id!r}")
            ids.add(question_id)
            answer = raw.get("answer")
            if answer is None and category == "5":
                answer = "No information available"
            if isinstance(answer, bool) or not isinstance(answer, (str, int, float)):
                raise ValueError(f"{location}: answer must be a non-empty scalar value")
            answer_text = str(answer).strip()
            if not answer_text:
                raise ValueError(f"{location}: answer must be a non-empty scalar value")
            raw_evidence = raw.get("evidence", ())
            source_values = (
                [raw_evidence] if isinstance(raw_evidence, str)
                else list(raw_evidence) if isinstance(raw_evidence, (list, tuple))
                else None
            )
            corrected_evidence = raw_evidence if source_values is None else [
                replacement for item in source_values
                if (replacement := KNOWN_EVIDENCE_CORRECTIONS.get((conversation_id, item), item)) is not None
            ]
            evidence = self._evidence_ids(corrected_evidence, location)
            unknown = sorted(set(evidence) - turn_ids)
            if unknown:
                raise ValueError(f"{location}: evidence IDs not present in conversation: {unknown}")
            metadata = {
                key: value for key, value in raw.items()
                if key not in {"question", "answer", "category", "evidence", "question_id", "id"}
            }
            if source_values is not None and corrected_evidence != source_values:
                metadata["source_evidence"] = tuple(source_values)
            questions.append(ConversationQuestion(
                question.strip(), answer_text, question_id, category, tuple(evidence), metadata,
            ))
        return questions

    @staticmethod
    def _evidence_ids(raw: Any, location: str) -> list[str]:
        if raw in (None, "", []):
            return []
        values = [raw] if isinstance(raw, str) else raw if isinstance(raw, (list, tuple)) else None
        if values is None or any(not isinstance(value, str) for value in values):
            raise ValueError(f"{location}: evidence must be a string or list of strings")
        ids = [identifier for value in values for identifier in EVIDENCE_ID_RE.findall(value)]
        residue = " ".join(values)
        residue = EVIDENCE_ID_RE.sub(" ", residue).replace(";", " ").replace(",", " ").strip()
        if residue or (values and not ids):
            raise ValueError(f"{location}: malformed evidence annotation {values!r}")
        return list(dict.fromkeys(ids))

    @staticmethod
    def _session_keys(payload: dict[str, Any]) -> list[str]:
        sessions = [key for key in payload if re.fullmatch(r"session_\d+", key)]
        if not sessions:
            raise ValueError("LoCoMo conversation must contain at least one numbered session")
        return sorted(sessions, key=lambda value: int(value.removeprefix("session_")))
