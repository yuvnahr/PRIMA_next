"""LoCoMo-to-conversation adapter."""

from __future__ import annotations

from typing import Any

from benchmarks.common.interfaces import Conversation, ConversationQuestion, ConversationTurn


class LoCoMoAdapter:
    """Convert raw LoCoMo JSON records into benchmark-neutral conversations."""

    def adapt(self, raw_data: Any) -> list[Conversation]:
        """Adapt a raw LoCoMo JSON document into conversations."""

        if isinstance(raw_data, list):
            records = raw_data
        elif isinstance(raw_data, dict):
            records = raw_data.get("data", [])
        else:
            raise ValueError("LoCoMo raw data must be a list or dictionary.")
        if not isinstance(records, list):
            raise ValueError("LoCoMo raw data must be a list or contain a 'data' list.")

        return [self._adapt_record(record, index) for index, record in enumerate(records)]

    def _adapt_record(self, record: dict[str, Any], index: int) -> Conversation:
        conversation_id = str(record.get("sample_id") or record.get("id") or index)
        conversation_payload = record.get("conversation", {})

        return Conversation(
            id=conversation_id,
            turns=self._normalize_turns(conversation_payload),
            questions=self._normalize_questions(record.get("qa", [])),
            metadata=self._normalize_metadata(record, conversation_payload),
        )

    def _normalize_turns(self, conversation_payload: dict[str, Any]) -> list[ConversationTurn]:
        turns: list[ConversationTurn] = []

        for session_id in self._session_keys(conversation_payload):
            session_turns = conversation_payload.get(session_id, [])
            timestamp = conversation_payload.get(f"{session_id}_date_time")
            if not isinstance(session_turns, list):
                continue

            for turn_index, turn in enumerate(session_turns):
                if not isinstance(turn, dict):
                    continue
                metadata = {
                    key: value
                    for key, value in turn.items()
                    if key not in {"speaker", "text", "dia_id"}
                }
                turns.append(
                    ConversationTurn(
                        speaker=str(turn.get("speaker", "")),
                        text=str(turn.get("text", "")),
                        turn_id=self._optional_string(turn.get("dia_id") or f"{session_id}:{turn_index}"),
                        session_id=session_id,
                        timestamp=self._optional_string(timestamp),
                        metadata=metadata,
                    )
                )

        return turns

    def _normalize_questions(self, qa_payload: Any) -> list[ConversationQuestion]:
        if not isinstance(qa_payload, list):
            return []

        questions: list[ConversationQuestion] = []
        for index, question in enumerate(qa_payload):
            if not isinstance(question, dict):
                continue
            metadata = {
                key: value
                for key, value in question.items()
                if key not in {"question", "answer", "category", "evidence", "question_id", "id"}
            }
            evidence = question.get("evidence", ())
            if isinstance(evidence, str):
                evidence = (evidence,)
            elif not isinstance(evidence, list | tuple):
                evidence = ()

            questions.append(
                ConversationQuestion(
                    question=str(question.get("question", "")),
                    answer=self._optional_string(question.get("answer")),
                    question_id=self._optional_string(question.get("question_id") or question.get("id") or index),
                    category=self._optional_string(question.get("category")),
                    evidence=tuple(str(item) for item in evidence),
                    metadata=metadata,
                )
            )

        return questions

    def _normalize_metadata(self, record: dict[str, Any], conversation_payload: dict[str, Any]) -> dict[str, Any]:
        metadata = {
            key: value
            for key, value in record.items()
            if key not in {"conversation", "qa"}
        }
        metadata["speakers"] = {
            "speaker_a": conversation_payload.get("speaker_a"),
            "speaker_b": conversation_payload.get("speaker_b"),
        }
        return metadata

    def _session_keys(self, conversation_payload: dict[str, Any]) -> list[str]:
        return sorted(
            (
                key
                for key, value in conversation_payload.items()
                if key.startswith("session_") and isinstance(value, list)
            ),
            key=self._session_sort_key,
        )

    def _session_sort_key(self, session_id: str) -> tuple[int, str]:
        suffix = session_id.removeprefix("session_")
        return (int(suffix), session_id) if suffix.isdigit() else (10**9, session_id)

    def _optional_string(self, value: Any) -> str | None:
        if value is None:
            return None
        return str(value)
