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


def test_observation_cutoff_allows_known_plans_and_excludes_late_reports():
    from datetime import UTC, datetime

    repository = InMemoryMemoryRepository()
    known = datetime(2023, 1, 1, tzinfo=UTC)
    late = datetime(2023, 1, 3, tzinfo=UTC)
    for key, observed, effective in (("plan", known, late), ("report", late, known)):
        repository.add(
            MemoryNote.create(
                "meeting",
                note_id=key,
                timestamp=known,
                embedding=(1, 0),
                context={"source_timestamp": observed.isoformat(), "effective_from": effective.isoformat()},
            )
        )
    request = RetrievalRequest("meeting", reference_time=known, query_embedding=(1, 0))
    for strategy in (DenseRetrievalStrategy(), SparseRetrievalStrategy(), TemporalRetrievalStrategy()):
        assert {r.note.id for r in strategy.retrieve(request, repository)} == {"plan"}


def test_sparse_admission_requires_real_matches_including_exact_numbers():
    from memory.retrieval.query_analysis import QueryAnalyzer

    repository = InMemoryMemoryRepository()
    repository.add(MemoryNote.create("Apricot marmalade 217", note_id="fruit"))
    repository.add(MemoryNote.create("Nate travels on 2025-03-17", note_id="trip"))
    strategy = SparseRetrievalStrategy()
    for text, expected in (("quasar gluon", []), ("17", ["trip"]), ("Nate", ["trip"]), ("2025-03-17", ["trip"])):
        for analysis in (None, QueryAnalyzer().analyze(text)):
            request = RetrievalRequest(text, analyzed_query=analysis)
            assert [r.note.id for r in strategy.retrieve(request, repository)] == expected


def test_cutoff_limits_transcript_and_graph_bridges_before_ranking(tmp_path):
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from benchmarks.locomo.experiment import _bounded_question
    from memory.graph.graph_edge import GraphEdge
    from memory.graph.graph_node import GraphNode
    from memory.graph.graph_repository import GraphRepository
    from memory.memory_repository import ChromaMemoryRepository
    from memory.retrieval.graph_strategy import GraphTraversalStrategy

    repository = ChromaMemoryRepository(str(tmp_path))
    graph = GraphRepository()
    turns = []
    for key, day in (("one", 1), ("two", 1), ("future", 2)):
        dt = datetime(2024, 1, day, 23, tzinfo=UTC)
        repository.add(MemoryNote.create(key, note_id=key, timestamp=dt))
        graph.add_node(GraphNode(key, key))
        turns.append(SimpleNamespace(text=key, timestamp=dt.isoformat(), speaker="Ada", metadata={}))
    graph.add_edge(GraphEdge("one", "future", "link"))
    graph.add_edge(GraphEdge("future", "two", "link"))
    source = ChromaMemoryRepository(str(tmp_path))
    request = RetrievalRequest("one", reference_time="2024-01-01", top_k=1)

    class Seed(DenseRetrievalStrategy):
        def retrieve(self, request, repository):
            from memory.retrieval.retrieval_result import RetrievalResult

            return [RetrievalResult(repository.get("one"), 1)]

    result = GraphTraversalStrategy(graph, seed_strategy=Seed()).retrieve(request, source)
    assert all(row.note.id == "one" for row in result)
    assert {n.id for n in request.as_of_view(source).list()} == {"one", "two"}
    question = SimpleNamespace(question="Which?", metadata={"reference_time": "2024-01-01"})
    text = _bounded_question(SimpleNamespace(turns=turns), question, 1)
    assert "two" in text and "future" not in text
