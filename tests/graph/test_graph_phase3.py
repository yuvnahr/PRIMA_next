import unittest
from typing import cast

from memory.graph.graph_builder import GraphBuilder
from memory.graph.graph_edge import GraphEdge
from memory.graph.graph_node import GraphNode
from memory.graph.graph_reasoning_engine import GraphReasoningEngine
from memory.graph.graph_repository import GraphRepository
from memory.memory_note import MemoryNote, _smart_overlap, calculate_smart_overlap
from memory.memory_repository import InMemoryMemoryRepository
from memory.memory_types import MemoryType
from memory.retrieval.graph_strategy import GraphTraversalStrategy
from memory.retrieval.retrieval_request import RetrievalRequest


class GraphPhase3Test(unittest.TestCase):
    def test_equal_weight_neighbors_follow_node_insertion_order(self) -> None:
        graph = GraphRepository()
        for node_id in ("root", "third", "first", "second"):
            graph.add_node(GraphNode(node_id, node_id))
        for node_id in ("second", "first", "third"):
            graph.add_edge(GraphEdge("root", node_id, "shared_context", weight=0.5))
        self.assertEqual([node.id for node, _ in graph.get_neighbors("root")], ["third", "first", "second"])
        graph.add_node(GraphNode("first", "first", labels=("changed",)))
        graph.add_edge(GraphEdge("root", "second", "shared_context", weight=0.8))
        self.assertEqual([node.id for node, _ in graph.get_neighbors("root")], ["second", "third", "first"])

    def test_overlap_cache_preserves_direction_duplicates_and_mutation(self) -> None:
        _smart_overlap.cache_clear()
        self.assertEqual(calculate_smart_overlap(["a", "ab", "a"], ["ab"]), 2)
        self.assertEqual(calculate_smart_overlap(["ab"], ["a", "ab", "a"]), 1)
        keywords = ["bread"]
        self.assertEqual(calculate_smart_overlap(["sourdough"], keywords), 0)
        self.assertEqual(calculate_smart_overlap(["sourdough"], keywords), 0)
        self.assertEqual(_smart_overlap.cache_info().hits, 1)
        keywords.append("sourdough starter")
        self.assertEqual(calculate_smart_overlap(["sourdough"], keywords), 1)
        self.assertEqual(_smart_overlap.cache_info().maxsize, 65536)

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
        node_a = cast(GraphNode, node_a)
        node_b = cast(GraphNode, node_b)
        self.assertTrue(reasoning.path_search(node_a.id, node_b.id))
        self.assertTrue(reasoning.detect_communities())

        request = RetrievalRequest(query="sourdough", memory_types=(MemoryType.EPISODIC,), top_k=2)
        results = GraphTraversalStrategy(graph).retrieve(request, repository)
        self.assertTrue(any(result.note.id == note_b.id for result in results))


if __name__ == "__main__":
    unittest.main()


def test_refresh_removes_only_obsolete_generated_relations(tmp_path):
    from memory.graph.graph_repository import ChromaGraphRepository
    from memory.memory_index import MemoryIndex

    for graph in (GraphRepository(), ChromaGraphRepository(str(tmp_path))):
        index = MemoryIndex(InMemoryMemoryRepository(), graph)
        first = index.add(MemoryNote.create("sourdough bread", note_id="a"))
        index.add(MemoryNote.create("sourdough starter", note_id="b"))
        manual = GraphEdge("node_b", "node_a", "shared_context", 0.9, {"producer": "curator"})
        graph.add_edge(manual)
        assert len(graph.edges) == 2
        index.update(first.with_updates(content="quasar", retrieval_metadata={"keywords": ["quasar"]}))
        assert list(graph.edges.values()) == [manual]
        assert graph.degree("node_a") == 1
        if isinstance(graph, ChromaGraphRepository):
            assert list(ChromaGraphRepository(str(tmp_path)).edges.values()) == [manual]
        index.delete("a", MemoryType.EPISODIC)
        assert graph.degree("node_b") == 0


def test_legacy_graph_remains_available_after_reconciliation(tmp_path):
    import chromadb

    from memory.graph.graph_repository import ChromaGraphRepository
    from memory.memory_metadata import encode_metadata

    client = chromadb.PersistentClient(path=str(tmp_path))
    legacy = client.get_or_create_collection(name="graph_index")
    legacy.add(ids=["graph_node_old"], documents=["gone"], embeddings=[[0.] * 8],
               metadatas=[encode_metadata({"record_type": "node", "node_id": "old", "memory_id": "gone",
                                           "labels": ["episodic"], "metadata": {}})])
    graph = ChromaGraphRepository(str(tmp_path))
    assert "old" in graph.nodes
    GraphBuilder(graph).rebuild_for_type(InMemoryMemoryRepository(), MemoryType.EPISODIC)
    assert "old" not in ChromaGraphRepository(str(tmp_path)).nodes
    assert legacy.get(ids=["graph_node_old"])["ids"] == ["graph_node_old"]
