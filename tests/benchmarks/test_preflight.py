from benchmarks.preflight import capability_report


def test_preflight_reports_optional_capabilities_without_installing() -> None:
    available = {"bert_score", "torch"}
    report = capability_report(lambda module: module in available)

    assert report["schema_version"] == "1.0"
    statuses = {item["name"]: item for item in report["capabilities"]}
    assert statuses["bertscore"]["available"] is True
    assert statuses["goemotions_encoder"]["available"] is False
    assert statuses["goemotions_encoder"]["missing"] == ["transformers", "sklearn"]
    assert statuses["rouge_l"]["available"] is True

