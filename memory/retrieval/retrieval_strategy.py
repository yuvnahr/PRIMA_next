"""Retrieval strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from memory.memory_repository import MemoryRepository
from memory.retrieval.retrieval_request import RetrievalRequest
from memory.retrieval.retrieval_result import RetrievalResult


class RetrievalStrategy(ABC):
    name: str

    @abstractmethod
    def retrieve(self, request: RetrievalRequest, repository: MemoryRepository) -> list[RetrievalResult]:
        raise NotImplementedError
