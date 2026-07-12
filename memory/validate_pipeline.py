"""Validate real production embedding constructors."""
from __future__ import annotations
import json
from types import SimpleNamespace
from memory.embedding_pipeline import current_embedding_metadata, get_embedding_pipeline
from memory.event_memory.event_builder import EventMemoryBuilder
from memory.event_memory.event_segmenter import EventSegment
from memory.evolution.memory_evolution_engine import MemoryEvolutionEngine
from memory.graph.graph_repository import GraphRepository
from memory.memory_note import MemoryNote
from memory.memory_repository import InMemoryMemoryRepository
from memory.memory_types import MemoryType
from reflection.reflection_lineage import ReflectionLineage
from reflection.reflection_memory import ReflectionMemory
from reflection.reflection_signal import ReflectionSignal
from reflection.reflection_types import ReflectionSignalType

def main() -> int:
    meta=current_embedding_metadata(); pipeline=get_embedding_pipeline(); note=MemoryNote.create("Nathan is called Nate and he likes tea."); query=pipeline.embed_query("What does Nate like?")
    turn=SimpleNamespace(turn_id="turn-1",speaker="Nathan",text="Nathan is called Nate and he likes tea.",session_id="s1")
    event=EventMemoryBuilder().build(EventSegment("validation",1,(turn,))).to_memory_note()
    reflection=ReflectionMemory("validation","test","Remember Nate likes tea.",ReflectionSignal.create(ReflectionSignalType.MEMORY_CONFLICT,.2,.9,"validation"),.9,ReflectionLineage()).to_memory_note()
    repo=InMemoryMemoryRepository(); repo.add(note); repo.add(MemoryNote.create("Nate likes tea.",salience_score=.9)); result=MemoryEvolutionEngine(repo,GraphRepository()).evolve(); evolved=result.evolved_memories[0] if result.evolved_memories else MemoryNote.create("Nate likes tea.",memory_type=MemoryType.SEMANTIC)
    checks={"backend_loaded":bool(meta["embedding_backend"]),"semantic_representation_active":bool(note.retrieval_metadata.get("semantic_representation")),"identity_normalization_active":bool(note.retrieval_metadata["semantic_representation"].get("identity_references")),"repository_fingerprint_valid":note.retrieval_metadata.get("backend_fingerprint")==meta["backend_fingerprint"],"query_embedding_path":len(query.vector)==meta["embedding_dimension"],"memory_embedding_path":len(note.embedding)==meta["embedding_dimension"],"event_embedding_path":event.retrieval_metadata.get("backend_fingerprint")==meta["backend_fingerprint"],"reflection_embedding_path":reflection.retrieval_metadata.get("backend_fingerprint")==meta["backend_fingerprint"],"evolution_embedding_path":evolved.retrieval_metadata.get("backend_fingerprint")==meta["backend_fingerprint"],"repository_metadata":all(note.retrieval_metadata.get(key) for key in ("embedding_backend","embedding_model","embedding_dimension","representation_version","identity_version","backend_fingerprint"))}
    checks["status"]="PASS" if all(checks.values()) else "FAIL"; print(json.dumps(checks,indent=2,sort_keys=True)); return 0 if checks["status"]=="PASS" else 1

if __name__=="__main__": raise SystemExit(main())
