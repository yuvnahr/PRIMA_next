import unittest

from memory.memory_context import StateSnapshot
from memory.memory_note import MemoryNote, stable_embedding
from memory.memory_repository import InMemoryMemoryRepository
from memory.memory_types import MemoryType, RetrievalWindow
from memory.retrieval.dense_strategy import DenseRetrievalStrategy
from memory.retrieval.hybrid_fusion import HybridFusion
from memory.retrieval.retrieval_confidence import RetrievalConfidenceEstimator
from memory.retrieval.retrieval_controller import RetrievalController
from memory.retrieval.retrieval_request import RetrievalRequest
from memory.retrieval.sparse_strategy import SparseRetrievalStrategy
from memory.retrieval.temporal_strategy import TemporalRetrievalStrategy


class RetrievalPhase2Test(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = InMemoryMemoryRepository()
        self.sourdough = self.repository.add(
            MemoryNote.create(
                "I baked sourdough bread for the surprise party",
                memory_type=MemoryType.EPISODIC,
                embedding=stable_embedding("sourdough bread surprise party"),
            )
        )
        self.exam = self.repository.add(
            MemoryNote.create(
                "I studied for the exam tomorrow",
                memory_type=MemoryType.EPISODIC,
                embedding=stable_embedding("exam tomorrow study"),
                state_snapshot=StateSnapshot(task_state={"topic": "study"}),
            )
        )

    def test_dense_sparse_temporal_retrieval(self) -> None:
        request = RetrievalRequest(
            query="sourdough party",
            query_embedding=tuple(stable_embedding("sourdough bread surprise party")),
            memory_types=(MemoryType.EPISODIC,),
            temporal_window=RetrievalWindow.LONG_TERM,
            top_k=2,
        )

        dense = DenseRetrievalStrategy().retrieve(request, self.repository)
        sparse = SparseRetrievalStrategy().retrieve(request, self.repository)
        temporal = TemporalRetrievalStrategy().retrieve(request, self.repository)

        self.assertEqual(dense[0].note.id, self.sourdough.id)
        self.assertEqual(sparse[0].note.id, self.sourdough.id)
        self.assertTrue(temporal)

    def test_hybrid_fusion_and_confidence(self) -> None:
        request = RetrievalRequest(
            query="sourdough party",
            query_embedding=tuple(stable_embedding("sourdough bread surprise party")),
            memory_types=(MemoryType.EPISODIC,),
            top_k=2,
        )
        controller = RetrievalController(self.repository)
        response = controller.retrieve(request)
        confidence = RetrievalConfidenceEstimator().estimate(list(response.results), requested_k=2)

        self.assertEqual(response.results[0].note.id, self.sourdough.id)
        self.assertGreater(confidence.confidence, 0)

        fused = HybridFusion().fuse({"dense": list(response.results)}, top_k=1)
        self.assertEqual(len(fused), 1)

    def test_state_aware_retrieval_boost_exists(self) -> None:
        request = RetrievalRequest(
            query="exam",
            query_embedding=tuple(stable_embedding("exam tomorrow study")),
            memory_types=(MemoryType.EPISODIC,),
            state_filter={"task_state": {"topic": "study"}},
            top_k=2,
        )
        response = RetrievalController(self.repository).retrieve(request)
        exam_result = next(result for result in response.results if result.note.id == self.exam.id)
        self.assertIn("state", exam_result.strategy_scores)


if __name__ == "__main__":
    unittest.main()
