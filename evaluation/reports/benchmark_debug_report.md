# PRIMA-NEXT LoCoMo QA Benchmark Debug Report

Date: 2026-07-15  
Validation model: `qwen3.5:4b` via Ollama 0.32.0  
Validation scope: one production conversation, 10 questions, seed 13, top-k 5

## Result

**PASS — the QA benchmark pipeline is production-ready.** The final 10-question run completed with 10 model calls, zero runtime errors, zero truncated generations, valid raw/parsed artifacts, and real metric implementations. The low QA score is now attributable to the frozen retrieval results and two model answer errors, rather than integration or scoring failure.

Final subset metrics:

| Metric | Value |
| --- | ---: |
| Exact Match | 0.000000 |
| F1 | 0.066667 |
| BLEU | 0.004254 |
| ROUGE-L | 0.069697 |
| BERTScore | 0.134673 |
| Memory hit rate | 0.600000 |
| Mean retrieved memories | 5.000000 |
| Runtime errors | 0 |

## Bugs discovered and fixed

1. **Ollama was never called in the invalid full run.** `Settings` rejected unrelated `PRIMA_*` entries from `.env`; all 1,986 questions fell back to extractive memory text. Fixed by configuring pydantic-settings to ignore unrelated keys.
2. **Generation failures were silently scored.** The experiment exited successfully after provider errors and scored fallback prose as Qwen output. It now writes diagnostics and fails the run if any generation error remains.
3. **Benchmark dates were discarded.** Conversation session timestamps were loaded but memories received the wall-clock ingestion time. The adapter/runtime now preserve the source session timestamp through `RuntimeContext` into `MemoryNote` and the answer context.
4. **Adversarial questions lost their target.** Category-5 items with no `answer` were excluded from scoring. They now receive the intended `No information available` target; the adversarial answer remains metadata only.
5. **Raw and parsed answers were indistinguishable.** Raw Qwen text, parsed answer, Ollama completion metadata, and errors are now recorded separately.
6. **Answer format was nondeterministic.** The prompt mixed JSON for insufficient context with free text for normal answers, producing verbose explanations and one 64-token truncation. It now requires one JSON object for every response, uses 128-token headroom, and invalidates any `done_reason=length` result.
7. **Quoted null was parsed as a literal answer.** Both JSON `null` and strings `"null"`/`"none"` now normalize to `No information available`.
8. **Reported metrics were not their named metrics.** “BERTScore” was token F1, F1 used set overlap, ROUGE-L was recall-only, and BLEU was un-clipped unigram overlap. The pipeline now uses real BERTScore, multiset/stemmed F1, standard sentence BLEU with smoothing, and ROUGE-L F-score.
9. **QA artifacts duplicated retrieval traces.** Ten raw records occupied 3.8 MB and the failed full run occupied 785 MB. QA raw output now retains the five selected memories and answer diagnostics without duplicating dense/sparse/rerank traces; the final 10-question raw artifact is about 80 KB.
10. **A dead `LOCOMO_DEVICE=cpu` setting was misleading GPU diagnosis.** It was unused and could not affect Ollama; it and the unused context constant were removed.
11. **`--clean-output` removed the deferred logger directory.** The CLI now recreates `logs/` after cleanup and before the first log write.

## Ollama, thinking mode, and GPU verification

- Ollama version: `0.32.0`.
- Installed model: `qwen3.5:4b`, model ID `2a654d98e6fb`, 3.4 GB.
- Direct `/api/generate` probe used `think: false` and returned `READY`, `thinking: null`, `done: true`, `done_reason: stop`.
- Every final validation response recorded `thinking: null` and `done_reason: stop`.
- NVIDIA evidence: GeForce GTX 1650, driver 592.27, CUDA 13.1.
- `ollama ps` after inference reported `qwen3.5:4b` at `66%/34% CPU/GPU` (an earlier probe reported `53%/47% CPU/GPU`). Qwen is therefore GPU-accelerated but partially offloaded because the 3.8 GB loaded model nearly fills the 4 GB GPU.
- The benchmark does not pass a CPU device option to Ollama. Real BERTScore intentionally runs after generation on CPU, so its phase does not indicate Qwen CPU inference.

## Prompt and parser verification

- Each prompt contains the question once and each of the five selected memories once.
- Preserved source timestamps are ISO-formatted 2022 session times, not 2026 ingestion times.
- Final prompt evaluation counts were well below Ollama's 4096-token context limit; no prompt truncation was observed.
- The output contract is exactly `{"answer": "<short factual answer>"}` or `{"answer": null}`.
- All 10 raw responses were valid JSON objects; all 10 parsed answers were non-empty; all 10 generations ended with `stop`.
- Thinking content did not appear in either the raw response or parsed metric input.

## Metric verification

- Exact Match compares normalized token sets after lowercasing, punctuation removal, and article/conjunction removal.
- F1 uses Porter-stemmed token multisets and harmonic mean of precision/recall.
- BLEU uses NLTK sentence BLEU with clipped n-grams, brevity penalty, and method-1 smoothing.
- ROUGE-L uses the standard `rouge` package's LCS F-score.
- BERTScore uses `bert-score` 0.3.13 with `roberta-large`, English baseline rescaling, and the F1 component.
- Manual checks below reproduce the aggregate metrics. No gold answer is included in prompt construction.

## Manual validation examples

All examples used the same verified JSON-only prompt contract described above. “Retrieved” lists the five memories injected into that prompt in rank order.

### 1. Friends besides Joanna

- Question: Is it likely that Nate has friends besides Joanna?
- Retrieved: screenplay submission; “little friends”; Spider-Man pin; screenplay encouragement; congratulations.
- Prompt: question plus those five memories and their 2022 timestamps; JSON-only answer contract.
- Raw: `{"answer": "null"}`
- Parsed: `No information available`
- Ground truth: `Yesteammates on hisvideo game team.`
- Metrics: EM 0; F1 0; BLEU 0; ROUGE-L 0; BERTScore 0.
- Origin: frozen retrieval miss; parser behaved correctly.

### 2. Shared interests

- Question: What kind of interests do Joanna and Nate share?
- Retrieved: general support; Joanna likes superheroes; two Nate gaming memories; project encouragement.
- Prompt: question plus those five memories and timestamps; JSON-only answer contract.
- Raw/parsed: `superheroes`
- Ground truth: `Watching movies, making desserts`
- Metrics: EM 0; F1 0; BLEU 0; ROUGE-L 0; BERTScore 0.236168.
- Origin: frozen retrieval miss; model answered the supplied context.

### 3. First viewing of Eternal Sunshine

- Question: When did Joanna first watch “Eternal Sunshine of the Spotless Mind”?
- Retrieved: first-play memory; former dog; later movie recommendation without first-view year; backup advice; screenplay submission.
- Prompt: question plus those five memories and timestamps; JSON-only answer contract.
- Raw: `{"answer": "null"}`
- Parsed: `No information available`
- Ground truth: `2019`
- Metrics: all five metrics 0.
- Origin: frozen retrieval miss.

### 4. First video-game tournament

- Question: When did Nate win his first video game tournament?
- Retrieved: tournament comparison; screenplay; current tournament; fourth tournament; first tournament “last week” dated 21 January 2022.
- Prompt: question plus those five dated memories; JSON-only answer contract.
- Raw/parsed: `2021-12-31`
- Ground truth: `the week before 21Janury, 2022`
- Metrics: all five metrics 0.
- Origin: model temporal calculation error; retrieval and timestamp injection were correct.

### 5. Allergy-safe pets

- Question: What pets wouldn't cause any discomfort to Joanna?
- Retrieved: pets she may not be allergic to; allergies; stuffed dog; two generic pet comments.
- Prompt: question plus those five memories and timestamps; JSON-only answer contract.
- Raw/parsed: `pets she's not allergic to`
- Ground truth: `Hairless cats or pigs, since they don't have fur...`
- Metrics: EM 0; F1 0; BLEU 0; ROUGE-L 0; BERTScore 0.247313.
- Origin: frozen retrieval miss of the specific hairless-cat/pig evidence.

### 6. Joanna's hobbies

- Question: What are Joanna's hobbies?
- Retrieved: drawings; writing/creative projects; generic project; support; Nate's pets.
- Prompt: question plus those five memories and timestamps; JSON-only answer contract.
- Raw/parsed: `Writing`
- Ground truth: `Writing, watching movies, exploring nature, hanging with friends.`
- Metrics: EM 0; F1 0.333333; BLEU 0.003257; ROUGE-L 0.333333; BERTScore 0.222514.
- Origin: frozen retrieval partial; model and metrics correctly preserve partial credit.

### 7. Duration of turtle ownership

- Question: How long has Nate had his first two turtles?
- Retrieved: turtle check-in; generic turtle support; screenplay; movie script; expanded tank.
- Prompt: question plus those five memories and timestamps; JSON-only answer contract.
- Raw: `{"answer": "null"}`
- Parsed: `No information available`
- Ground truth: `three years`
- Metrics: EM/F1/BLEU/ROUGE-L 0; BERTScore 0.123718.
- Origin: frozen retrieval miss.

### 8. First screenplay completion

- Question: When did Joanna finish her first screenplay?
- Retrieved: later screenplay; first play; former dog; finished/printed screenplay “last Friday” dated 23 January 2022; backup advice.
- Prompt: question plus those five dated memories; JSON-only answer contract.
- Raw/parsed: `2022-01-23`
- Ground truth: `The Friday before 23January, 2022`
- Metrics: EM/F1/BLEU/ROUGE-L 0; BERTScore 0.104376.
- Origin: model used the session date instead of resolving “last Friday”; retrieval and timestamps were correct.

### 9. First two turtles

- Question: When did Nate get his first two turtles?
- Retrieved: screenplay; turtle beach trip; writing; turtle check-in; acquisition of a third turtle.
- Prompt: question plus those five memories and timestamps; JSON-only answer contract.
- Raw: `{"answer": "null"}`
- Parsed: `No information available`
- Ground truth: `2019`
- Metrics: all five metrics 0.
- Origin: frozen retrieval miss.

### 10. January 2022 achievement

- Question: What major achievement did Joanna accomplish in January 2022?
- Retrieved: first play; Nate tournament; later book completion; writing; screenplay rejection after completion.
- Prompt: question plus those five memories and timestamps; JSON-only answer contract.
- Raw/parsed: `finished up her writing for her book`
- Ground truth: `finished her screenplay and printed it`
- Metrics: EM 0; F1 0.333333; BLEU 0.039281; ROUGE-L 0.363636; BERTScore 0.412645.
- Origin: frozen retrieval partial/miss; model selected the wrong retrieved achievement.

## Files modified

- `.gitignore`
- `benchmarks/common/runtime_adapter.py`
- `benchmarks/locomo/adapter.py`
- `benchmarks/locomo/config.py`
- `benchmarks/locomo/evaluate.py`
- `benchmarks/locomo/experiment.py`
- `config/settings.py`
- `llm/provider.py`
- `llm/response_parser.py`
- `memory/memory_note.py`
- `requirements.txt`
- `runtime/prima_runtime.py`
- `tests/test_locomo_qa_correctness.py`

## Validation evidence

- Focused correctness test: `1 passed`.
- Modified production modules: `py_compile` PASS.
- Final production subset: 1 conversation, exactly 10 questions, zero generation errors.
- Ollama: provider/model correct, `stream: false`, deterministic temperature 0 and seed 13, timeout 180 seconds, two retries, thinking disabled.
- Retrieval logic, embedding logic, and scientific retrieval configuration were not modified.

## Remaining issues

No benchmark-pipeline blocker remains. The manual sample contains frozen retrieval misses and two Qwen answer errors; these are valid scientific outcomes and were not optimized away. The invalid prior full-run artifacts and temporary validation outputs were removed after this report was generated.
