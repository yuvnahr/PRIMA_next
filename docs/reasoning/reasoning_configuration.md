# Reasoning configuration

| Variable | Default | Purpose |
| --- | ---: | --- |
| `PRIMA_REASONING_MODE` | `adaptive` | Default execution mode; set `single_pass` to roll back. |
| `PRIMA_REASONING_MAX_HOPS` | `3` | Maximum retrieval hops. |
| `PRIMA_REASONING_MAX_RETRIEVAL_CALLS` | `3` | Maximum retrieval calls. |
| `PRIMA_REASONING_MAX_LLM_CALLS` | `4` | Maximum synthesis calls. |
| `PRIMA_REASONING_MAX_DOCUMENTS` | `12` | Maximum retained unique evidence items. |
| `PRIMA_REASONING_MAX_CONTEXT_TOKENS` | `1600` | Maximum evidence context tokens. |
| `PRIMA_REASONING_TIME_BUDGET_SECONDS` | `15` | Wall-clock request limit. |
| `PRIMA_REASONING_NO_PROGRESS_LIMIT` | `1` | Consecutive non-novel hops before stopping. |

Per-call `max_hops`, `max_context_tokens`, `reasoning_mode`, `session_id`, and `diagnostics` override the relevant runtime configuration. Invalid mode values fail safe to `single_pass`.
