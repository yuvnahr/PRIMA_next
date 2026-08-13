# LoCoMo-v2 in PRIMA

LoCoMo-v2 uses the existing LoCoMo adapter, production embedding pipeline, dense retrieval, metrics, and campaign runner. Only the dataset and output root change.

Smoke test one conversation:

```powershell
.venv\Scripts\python.exe -m evaluation.experiments.run_campaigns --dataset locomo-v2 --max-conversations 1
```

The command writes comparable artifacts below `evaluation/locomo-v2/`. Run the full benchmark manually by omitting `--max-conversations 1`.
