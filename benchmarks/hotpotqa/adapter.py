"""HotpotQA-to-conversation adapter."""
from __future__ import annotations
from typing import Any
from benchmarks.common.interfaces import Conversation, ConversationQuestion, ConversationTurn
from benchmarks.hotpotqa.config import CONTEXT_SOURCES

class HotpotQAAdapter:
    """Adapt supplied HotpotQA contexts without exposing gold labels to inference."""
    def __init__(self, mode: str = "distractor") -> None:
        if mode not in CONTEXT_SOURCES:
            raise ValueError(f"Unsupported HotpotQA mode: {mode}")
        self.mode = mode

    def adapt(self, records: list[dict[str, Any]]) -> list[Conversation]:
        return [self._adapt(record) for record in records]

    def _adapt(self, record: dict[str, Any]) -> Conversation:
        sample_id = record["_id"]
        if self.mode == "oracle" and "supporting_facts" not in record:
            raise ValueError(f"Oracle mode requires supporting_facts for sample {sample_id}")
        supporting = {(title, sentence_id) for title, sentence_id in record.get("supporting_facts", [])}
        turns = []
        for paragraph_index, (title, sentences) in enumerate(record["context"]):
            for sentence_id, sentence in enumerate(sentences):
                if self.mode == "oracle" and (title, sentence_id) not in supporting:
                    continue
                turns.append(ConversationTurn(
                    speaker=title,
                    text=sentence,
                    turn_id=f"{sample_id}:{paragraph_index}:{sentence_id}",
                    session_id=sample_id,
                    metadata={
                        "document_id": f"{sample_id}:{paragraph_index}:{sentence_id}",
                        "sample_id": sample_id,
                        "source_title": title,
                        "paragraph_index": paragraph_index,
                        "sentence_id": sentence_id,
                        "original_sentence_text": sentence,
                        "benchmark_source": CONTEXT_SOURCES[self.mode],
                    },
                ))
        question_metadata = {}
        if "supporting_facts" in record:
            question_metadata["supporting_facts"] = [list(fact) for fact in record["supporting_facts"]]
        return Conversation(
            id=sample_id,
            turns=tuple(turns),
            questions=(ConversationQuestion(
                question=record["question"],
                answer=record.get("answer"),
                question_id=sample_id,
                category=record.get("type"),
                metadata=question_metadata,
            ),),
            metadata={"level": record.get("level"), "benchmark_mode": self.mode, "context_source": CONTEXT_SOURCES[self.mode], "diagnostic": self.mode == "oracle"},
        )
