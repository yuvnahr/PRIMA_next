from pathlib import Path

from memory.memory_repository import ChromaMemoryRepository
from runtime.prima_runtime import PrimaRuntime


def test_production_embedding_smoke(tmp_path: Path) -> None:
    path = tmp_path / "memory_db"
    first = PrimaRuntime(memory_repository=ChromaMemoryRepository(str(path)))
    stored = first.process("Please remember that my name is Nathan. Nathan is called Nate and he likes tea.")
    assert stored.memory_notes_created
    restarted = PrimaRuntime(memory_repository=ChromaMemoryRepository(str(path)))
    result = restarted.answer_question("What does Nate like?")
    assert result.retrieved_memories
    assert restarted.memory_repository.fingerprint_status == "valid"
