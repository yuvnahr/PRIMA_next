from datetime import datetime, timezone

from benchmarks.locomo.adapter import LoCoMoAdapter
from benchmarks.locomo.evaluate import exact_match_score, f1_score, rouge_l_score
from llm.response_parser import extract_answer
from memory.memory_note import MemoryNote


def test_production_qa_correctness_contract() -> None:
    conversation = LoCoMoAdapter().adapt(
        [{"conversation": {}, "qa": [{"question": "Unknown?", "category": 5, "adversarial_answer": "made up"}]}]
    )[0]
    timestamp = datetime(2022, 1, 21, tzinfo=timezone.utc)
    note = MemoryNote.create("dated memory", embedding=[0.0], timestamp=timestamp)

    assert conversation.questions[0].answer == "No information available"
    assert extract_answer('{"answer": null, "insufficient_information": true}') == "No information available"
    assert extract_answer('{"answer": "null"}') == "No information available"
    assert note.timestamp == timestamp
    assert exact_match_score("The Friday!", "friday") == 1.0
    assert f1_score("cats cats", "cats") == 2 / 3
    assert rouge_l_score("finished screenplay", "she finished screenplay") > 0.7
