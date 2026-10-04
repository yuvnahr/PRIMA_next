# What the old LoCoMo notebook establishes

Reference: `C:\Users\Yuv Nahar\Downloads\notebookc40238866e.ipynb`.
Only saved source, output and execution metadata were read. The notebook was
not executed or edited. No new tests or benchmark runs were performed.

## Saved observations

The notebook records a run on 16–17 August 2026, Python 3.12.13, two Tesla T4s,
and repository commit `6442fd58a0230ae0535827672ee583768eb92f11`. It fetched
`master`; the current notebook instead clones only `3.10_to_3.14`.

| Observation | Saved value | Interpretation |
|---|---:|---|
| Model | Qwen3.8 27B, 27.3B parameters, Q4_K_M | Same named reference model; exact digest matters |
| Model artifact | 16.52 GiB | Disk size, not total runtime VRAM |
| Digest | `22130167c4c20e20c7b71454612966ca8e8171e9b3cc8ab6ce8aa6cbfec79643` | Original saved snapshot; current default uses a separately pinned available revision |
| Context and placement | 8192, 100% GPU | Evidence for the old one-slot configuration |
| Preflight GPU memory | 9555 / 9729 MiB | Below the current 12 GiB per-GPU ceiling |
| LoCoMo source | 1986 questions; hash `553cd5a15e25f2ceccc6ed185221eba645080c93e5b91087560a91aa5961f365` | Matches the current required source identity |
| Full-run progress | 698/1986, 27049 seconds elapsed, 49912 seconds ETA | Partial run; approximately 21h23m extrapolated |
| Mean saved QA latency | 21372 ms | QA execution, including CPU retrieval/workflow; not isolated GPU decode time |
| GPU memory at the final progress sample | 3 / 3 MiB | Model weights were not resident on either GPU at that sampled instant |
| Successful saved checkpoints | 698; zero execution failures | Does not establish answer accuracy or complete coverage |

The full-run cell has `execution_failed` metadata at 05:15:38 on 17 August.
No traceback or exit reason is saved, so this cannot be attributed to Kaggle's
time limit, out-of-memory, or a particular code failure. There are no saved
HotpotQA or GoEmotions runs, RAM peaks, inference-duration breakdowns, or
two-worker measurements in this file.

## Why the old green light was unreliable

The canary output says `20/20`, approximately zero seconds elapsed, and a
0.07-hour full-run projection while its saved records have a mean latency of
20819 ms. The entire cell's saved timestamps span only about 6.24 seconds.
Those durations cannot represent a fresh sequential execution of those 20
recorded QA latencies, plus conversation replay.

The source uses fixed `locomo-canary` directories, `RESUME=True`, and resumes
when a manifest exists. Reused checkpoints are the likely explanation, but
the saved file alone does not prove the exact resume state. The current
notebook uses fresh namespaces, refuses pilot resume, includes full replay,
and accounts for setup, both candidate runs and artifact time.

The dashboard's `tokens=0` also does not prove the model was never called.
The old Ollama adapter populated prompt/completion counts but omitted their
total, while LoCoMo and the dashboard read the missing total field. The
adapter now supplies the total and carries model-load, prompt-evaluation,
decode and total provider durations when Ollama returns them. The adapter
also explicitly sets the campaign context and 24-hour residency on each
request, rather than relying only on the warmup request or server defaults.

## Source-based runtime improvements

The code at the old commit uses the same graph maintenance/retrieval paths
reviewed here. Centrality used to retrieve and sort every node's neighbor
records merely to count them. Direct adjacency degrees produce the same
rounded centrality values for the valid native graph, with O(V) work instead
of materializing and sorting all adjacency lists on every retrieval.

Neighbor ordering is now cached as lists of existing node IDs, preserving
the original descending weights and equal-weight ordering. Adding/replacing
an edge invalidates both endpoint lists when its weight/order changes or the
edge is new; metadata-only replacements retain the same order. Returned node/edge objects remain
current. This avoids repeated sorting during multi-seed, multi-hop graph
retrieval. The extra retained references scale with the existing adjacency
size (roughly two references per undirected edge); GPU/RAM guards stay active.

Graph rebuilding takes one keyword snapshot per memory instead of repeatedly
copying keyword lists for every pair. Every original pair, directional
substring-overlap calculation, edge and maintenance operation is retained.
The overlap is asymmetric, so skipping supposedly unchanged index updates
could alter weights/relations; that shortcut was not introduced.

Earlier changes cache spaCy per worker and normalized float32 retrieval
vectors, index workflow events by execution ID, consume incremental
checkpoints, keep the model resident and schedule larger conversations first.
These target specific repeated work seen in source. Their combined speedup
has not been measured, and no fixed factor is credited as an observation.

## Honest 12-hour assessment

The partial reference averages `27049/698 = 38.75` wall seconds per completed
question. Extrapolating its saved mean QA latency alone to 1986 questions gives
`1986 * 21.372 / 3600 = 11.79` hours of serial QA work. The difference from the
wall projection is about 9.59 hours of replay, maintenance, checkpointing,
monitoring, pauses or other non-QA work. This decomposition is illustrative:
later conversations and questions need not have the same costs, and part of
the currently replaying conversation may already be charged to elapsed time.

With 90 minutes for setup/pilots, a 15-minute artifact reserve and a 20%
margin, raw full campaign work must be at most about **8h32m**. The target is
8 hours. At the same single-worker QA latency, removing all non-QA overhead
still cannot meet that window; CPU QA improvements and/or useful concurrency
are necessary. The new graph retrieval changes directly target CPU QA work.

For illustration only, an effective 1.5x reduction in QA work and an 80%
reduction in other work would yield about `11.79/1.5 + 9.59*0.2 = 9.78` raw
hours, or **13.48 hours** with margin/setup/reserve: still too slow. A 2x QA
reduction and a 75% reduction in other work would yield **8.29 raw hours**,
or **11.70 hours** including those allowances. These are sensitivity cases,
not measured forecasts, and two model slots do not guarantee a 2x reduction.

There is a credible path with the same 27B model size to LoCoMo fitting 12 hours because
substantial repeated CPU work has been removed. There is insufficient evidence
to guarantee it. HotpotQA and GoEmotions cannot inherit LoCoMo's timing: they
have different example counts, prompts and routes. Each uses its full selected
Kaggle source, native two-worker path and independent session budget gate.

For a source with N independent examples, the 8-hour target requires N/28800
overall examples per second, including native evaluation/artifacts. Use the
actual selected source count, never an assumed standard split size. The
notebook now freezes the authorized Kaggle input snapshot when no optional
expected digest is supplied; every mode must retain its SHA, complete source
IDs and count throughout canary, full execution and final validation.

Decision: proceed to the notebook's guarded setup/preflight and representative
admission stage. An unconditional full-run 12-hour green light is unsupported.
No new timing runs are requested here; this assessment uses the supplied old
notebook and source inspection. If the live gate later rejects the 27B profile,
the smaller uniform model/new-experiment fallback remains necessary. The notebook
now pins the available 2026-10-05 27B registry revision explicitly; it cannot
claim the old saved model's exact identity or identical scores. The old snapshot
remains usable through a matching cache and an explicit old expected digest.

## Primary references

[Ollama generation API](https://docs.ollama.com/api/generate) describes
per-request residency and model-load/prompt/decode timing fields.
[Ollama runner source](https://github.com/ollama/ollama/blob/main/llm/llama_server.go)
keeps the requested per-slot context and allocates total context multiplied
by parallel slots; parallel memory/speed cannot be inferred from the old
single-slot placement alone.
