# Settings and authentication

Supply credentials through a scoped DuckDB secret or an environment variable.
A matching secret takes precedence. Credentials are redacted by DuckDB and
cannot be passed as function options.

```sql
CREATE SECRET jev (TYPE jev, PROVIDER env); -- JEV_API_KEY
-- For an OpenAI-compatible first or second stage:
CREATE SECRET llm (TYPE openai, PROVIDER env); -- OPENAI_API_KEY
```

| Setting | Default | Meaning |
|---|---|---|
| `jevdb_backend` | `jev` | First stage: `jev` or `llm` |
| `jevdb_endpoint` | `https://api.typesafe.ai/v1/` | First-stage URL prefix |
| `jevdb_model` | `jev-1.13.0` | First-stage model |
| `jevdb_layout` | `auto` | Jev pair layout: auto, star, pack, sep |
| `jevdb_anchor` | `auto` | Star anchor: auto, left, right |
| `jevdb_k` | 50 | Maximum judgments per request |
| `jevdb_threads` | 8 | Concurrent first-stage request workers |
| `jevdb_threshold` | 0.5 | Probability threshold for holds |
| `jevdb_budget_tokens` | 50000 | Approximate budget from characters / 3.5 |
| `jevdb_auto_min_group` | 8 | Average pairs per anchor for auto to select star |
| `jevdb_prefetch` | true | Collect a probe block before sending judgments |
| `jevdb_reduce_inputs` | true | Evaluate ordinary joins that restrict one semantic input before it; see [Pruning](pruning.md) |
| `jevdb_timeout` | 120 | HTTP read/write timeout, seconds; connection timeout is 30 |
| `jevdb_ca_cert_file` | empty | CA file; otherwise SSL_CERT_FILE or system trust |
| `jevdb_max_retries` | 7 | Additional attempts for transient failures, both stages |
| `jevdb_retry_max_delay` | 60 | Exponential backoff cap, seconds |
| `jevdb_retry_after_max` | 300 | Longest wait a Retry-After header can impose, seconds |
| `jevdb_on_error` | `error` | Exhausted HTTP requests: error or explicit reject |
| `jevdb_dump_path` | empty | Jev probability/choice/score JSONL log |
| `jevdb_raw_dump_path` | empty | Request/response JSONL log, both backends and stages |
| `jevdb_cascade_model` | empty | Second-stage model; empty disables cascade |
| `jevdb_cascade_endpoint` | `https://api.openai.com/v1/` | Second-stage URL prefix |
| `jevdb_cascade_low` | 0.5 | Reject probabilities below this bound |
| `jevdb_cascade_high` | 0.9 | Accept probabilities at or above this bound |
| `jevdb_cascade_calibrate` | 0 | Sample size for automatic bounds; zero uses fixed bounds |
| `jevdb_cascade_recall_loss` | 0.05 | Maximum calibration positives rejected below the interval |
| `jevdb_cascade_precision_loss` | 0.05 | Maximum fraction of negatives accepted above the interval |
| `jevdb_cascade_batch_size` | 16 | Maximum second-stage judgments per request |
| `jevdb_cascade_threads` | 8 | Concurrent second-stage workers |
| `jevdb_cascade_fill_batches` | true | Fill batches; false spreads small pools across workers |
| `jevdb_cascade_timeout` | 120 | Second-stage read/write timeout, seconds |
| `jevdb_cascade_ca_cert_file` | empty | Second-stage HTTPS CA file |

Every decision setting can be overridden with a constant function options STRUCT,
using the name without `jevdb_`. DuckDB's SQL `threads` setting is separate from
HTTP concurrency.

Auto uses star when the average pair count per smaller input group reaches
`jevdb_auto_min_group`, otherwise pack. Single-record Jev calls are packed except
for sep. The LLM backend uses ordered batches bounded by k and the token estimate;
sep sends one item. Star anchors apply to Jev only.

The LLM backend uses Chat Completions with [strict JSON Schema output](https://developers.openai.com/api/docs/guides/structured-outputs). Configure
an endpoint and model supporting that API. It generates answers and usually costs
more and takes longer than Jev; estimated probabilities are self-reported.

```sql
SET jevdb_backend='llm';
SET jevdb_endpoint='https://api.openai.com/v1/';
SET jevdb_model='<model-with-structured-outputs>';
```

Transient connection/read/write failures, HTTP 429 and HTTP 5xx retry the same
body with 1, 2, 4, ... seconds of backoff, up to the configured cap. Each wait is
drawn at random between half of that delay and all of it, so concurrent workers
do not retry in step. Retry-After (seconds or an HTTP date) extends the wait
beyond the backoff cap, up to `jevdb_retry_after_max`. Interrupting the query
ends a wait immediately. Permanent HTTP errors, invalid answers and certificate
verification errors fail directly.
By default an exhausted or incomplete request aborts the query. Explicit
`on_error='reject'` turns unavailable first-stage Boolean answers into false and
probabilities into NULL; unavailable second-stage answers become false.
Choice and score errors always abort. This setting can discard matches.

`SELECT * FROM jevdb_stats()` reports the last semantic query. `requests` and
`cascade_requests` count batches, while `retries` and `cascade_retries` count
additional HTTP attempts. Input counts follow deduplication. Request time sums
latencies including retries; send time measures parallel dispatch wall time.
Logs exclude credentials but contain document text.

To reproduce the JEVDB-Flash result configuration, use single-stage Jev with no
key predicate and set `layout='pack'`, `k=100`, `jevdb_threads=8`,
`budget_tokens=50000`, and DuckDB `threads=1`.
