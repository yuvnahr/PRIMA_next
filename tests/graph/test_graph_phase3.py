import unittest

from memory.graph.graph_builder import GraphBuilder
from memory.graph.graph_reasoning_engine import GraphReasoningEngine
from memory.graph.graph_repository import GraphRepository
from memory.memory_note import MemoryNote
from memory.memory_repository import InMemoryMemoryRepository
from memory.memory_types import MemoryType
from memory.retrieval.graph_strategy import GraphTraversalStrategy
from memory.retrieval.retrieval_request import RetrievalRequest


class GraphPhase3Test(unittest.TestCase):
    def test_graph_build_traversal_communities_and_retrieval(self) -> None:
        repository = InMemoryMemoryRepository()
        note_a = repository.add(MemoryNote.create("I baked sourdough bread", memory_type=MemoryType.EPISODIC))
        note_b = repository.add(MemoryNote.create("The sourdough starter needs feeding", memory_type=MemoryType.EPISODIC))
        repository.add(MemoryNote.create("I studied for an exam", memory_type=MemoryType.EPISODIC))

        graph = GraphRepository()
        GraphBuilder(graph).rebuild_for_type(repository, MemoryType.EPISODIC)
        reasoning = GraphReasoningEngine(graph)

        node_a = graph.find_by_memory_id(note_a.id)
        node_b = graph.find_by_memory_id(note_b.id)
        self.assertIsNotNone(node_a)
        self.assertIsNotNone(node_b)
        self.assertTrue(reasoning.path_search(node_a.id, node_b.id))
        self.assertTrue(reasoning.detect_communities())

        request = RetrievalRequest(query="sourdough", memory_types=(MemoryType.EPISODIC,), top_k=2)
        results = GraphTraversalStrategy(graph).retrieve(request, repository)
        self.assertTrue(any(result.note.id == note_b.id for result in results))


if __name__ == "__main__":
    unittest.main()
