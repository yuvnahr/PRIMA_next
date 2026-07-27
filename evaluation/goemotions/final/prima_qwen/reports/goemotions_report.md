# GoEmotions Report

## Configuration

```json
{
  "benchmark": "goemotions",
  "dataset_path": "benchmarks\\goemotions\\external\\goemotions\\data\\test.tsv",
  "dataset_samples": 5427,
  "dataset_sha256": "dc3aab545d76079817c3002a77239d53accc11ec8f16313aa19d8e1a8f2e412e",
  "execution": {
    "actual_batch_size": 1,
    "note": "Ollama systems issue one request per example; encoder systems batch tensors.",
    "parallel_workers": 1,
    "requested_batch_size": 1
  },
  "git_commit": "ce6cec222da4dedf0916f981340927ed959d9326",
  "hardware": {
    "machine": "AMD64",
    "platform": "Windows-11-10.0.26200-SP0",
    "processor": "Intel64 Family 6 Model 154 Stepping 3, GenuineIntel"
  },
  "labels": [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral"
  ],
  "runtime": {
    "device": "auto",
    "model": "qwen3.5:4b",
    "provider": "ollama",
    "seed": 13,
    "split": "test",
    "system": "prima_qwen",
    "temperature": 0,
    "thinking": false
  }
}
```

## Metrics

- Accuracy: `0.217247`
- Macro F1: `0.265876`
- Micro F1: `0.281790`
- Weighted F1: `0.279280`

## Per-class metrics

| Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| admiration | 0.648649 | 0.238095 | 0.348331 | 504 |
| amusement | 0.514451 | 0.674242 | 0.583607 | 264 |
| anger | 0.346154 | 0.409091 | 0.375000 | 198 |
| annoyance | 0.420290 | 0.090625 | 0.149100 | 320 |
| approval | 0.171171 | 0.162393 | 0.166667 | 351 |
| caring | 0.317073 | 0.096296 | 0.147727 | 135 |
| confusion | 0.451613 | 0.183007 | 0.260465 | 153 |
| curiosity | 0.241987 | 0.531690 | 0.332599 | 284 |
| desire | 0.314607 | 0.337349 | 0.325581 | 83 |
| disappointment | 0.108731 | 0.437086 | 0.174142 | 151 |
| disapproval | 0.210762 | 0.176030 | 0.191837 | 267 |
| disgust | 0.113879 | 0.520325 | 0.186861 | 123 |
| embarrassment | 0.500000 | 0.243243 | 0.327273 | 37 |
| excitement | 0.384615 | 0.291262 | 0.331492 | 103 |
| fear | 0.461538 | 0.538462 | 0.497041 | 78 |
| gratitude | 0.834677 | 0.588068 | 0.690000 | 352 |
| grief | 0.000000 | 0.000000 | 0.000000 | 6 |
| joy | 0.165289 | 0.621118 | 0.261097 | 161 |
| love | 0.722222 | 0.109244 | 0.189781 | 238 |
| nervousness | 0.600000 | 0.130435 | 0.214286 | 23 |
| optimism | 0.256410 | 0.107527 | 0.151515 | 186 |
| pride | 0.304348 | 0.437500 | 0.358974 | 16 |
| realization | 0.181818 | 0.165517 | 0.173285 | 145 |
| relief | 0.096774 | 0.545455 | 0.164384 | 11 |
| remorse | 0.000000 | 0.000000 | 0.000000 | 56 |
| sadness | 0.677419 | 0.269231 | 0.385321 | 156 |
| surprise | 0.148362 | 0.546099 | 0.233333 | 141 |
| neutral | 0.591885 | 0.138780 | 0.224841 | 1787 |

## Failure analysis

- Exact-set failures: `4248`
- Parse failures: `72`
- The confusion figure is a gold-label/predicted-label co-occurrence diagnostic for this multilabel task.
