"""HotpotQA benchmark integration."""

from benchmarks.hotpotqa.adapter import HotpotQAAdapter
from benchmarks.hotpotqa.evaluate import HotpotQAEvaluator
from benchmarks.hotpotqa.loader import HotpotQADataset
from benchmarks.hotpotqa.runner import HotpotQARunner

__all__ = ["HotpotQAAdapter", "HotpotQADataset", "HotpotQAEvaluator", "HotpotQARunner"]
