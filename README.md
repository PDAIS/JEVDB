<p align="center">
  <img src="docs/assets/jevdb-logo.png" alt="JEVDB logo" width="560">
</p>

# JEVDB
[![Paper](https://img.shields.io/badge/Paper-arXiv-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2610.02046)
[![Website](https://img.shields.io/badge/Website-JEVDB-CFB991)](https://jevdb.org)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PDAIS/JEVDB/blob/main/examples/jevdb_colab.ipynb)

**Scalable semantic SQL on decision models.**
JEVDB (Just Evaluation and Verification Database) is a scalable semantic database
built on the DuckDB ecosystem. It runs semantic filters, semantic joins,
classification and scoring over text as part of ordinary SQL.

By default JEVDB batches decisions on a Jev model in a single stage. It returns
probabilities, labels and scores directly, so ordinary semantic filters and joins
need no generated explanation. Run this way, without key pruning (SBF), JEVDB is
JEVDB-Flash.

An OpenAI-compatible LLM backend is also available for users without a Jev
account. Optional key conditions remove impossible pairs; an optional cascade
sends uncertain decisions to a second-stage model.

## Install

For Python, install the package and open a connection:

```sh
pip install jevdb
```

```python
import jevdb

con = jevdb.connect()
```

For the DuckDB CLI, build the extension from source as described in
[Building](docs/building.md), then load it:

```sql
-- start DuckDB 1.4.3 with: duckdb -unsigned
LOAD '/absolute/path/to/build/release/extension/jevdb/jevdb.duckdb_extension';
```

## Quick start

JEVDB uses [Jev](https://docs.typesafe.ai/models) as its decision model. Put your API key in
`JEV_API_KEY`; keys never appear in SQL.

```python
rows = con.execute("""
    CREATE SECRET jev (TYPE jev, PROVIDER env);   -- reads JEV_API_KEY

    SELECT jev_holds(
        'The review is clearly positive',
        'Loved every minute of it.'
    )
""").fetchall()

print(rows)
```

Run the following SQL through `con.execute(...)` in Python, or directly in the
DuckDB CLI. Use `.fetchall()` to retrieve query results.

```sql
CREATE SECRET jev (TYPE jev, PROVIDER env);   -- reads JEV_API_KEY

CREATE TABLE reviews (id INTEGER, review VARCHAR);
INSERT INTO reviews VALUES
  (1, 'Loved every minute of it.'),
  (2, 'A tedious, overlong mess.'),
  (3, 'Fine, but forgettable.');

-- Semantic filter
SELECT id FROM reviews
WHERE jev_holds('The review is clearly positive', review);

-- Probability, classification and scoring
SELECT id,
       jev_probability('The review is clearly positive', review) AS p,
       jev_choose('Classify the sentiment', review, ['positive', 'negative', 'mixed']) AS label,
       jev_score('Rate the enthusiasm', review, ['low', 'medium', 'high']) AS enthusiasm
FROM reviews;

-- Semantic join
SELECT a.id, b.id
FROM reviews a JOIN reviews b
  ON a.id < b.id
 AND jev_holds('The two reviews express the same sentiment', a.review, b.review);
```

## Semantic operators

| Operation | Write it as | Notes |
|---|---|---|
| Semantic filter | `WHERE jev_holds(condition, col)` | Ordinary predicates in the same `WHERE` are applied first |
| Semantic join | `JOIN ... ON jev_holds(condition, a.col, b.col)` | Also inside `EXISTS` and `NOT EXISTS` for semi and anti joins |
| Classification | `jev_choose(condition, col, choices)` | One label per row |
| Scoring | `jev_score(condition, col, levels)` | A position on ordered levels |

Each row or pair is a decision. JEVDB does not generate text, so extraction,
rewriting and summarization are outside its scope. There are no separate
operators for semantic aggregation or top-k: `ORDER BY jev_score(...) LIMIT k`
works, and scores every row.

## SQL functions

| Function | Returns | Use |
|---|---|---|
| `jev_holds(condition, a [, b] [, options])` | BOOLEAN | Filter on one record, or join on a pair |
| `jev_probability(condition, a [, b] [, options])` | DOUBLE | The decision model's probability |
| `jev_choose(condition, a, choices [, options])` | VARCHAR | Pick one label from a list or MAP |
| `jev_score(condition, a, levels [, options])` | DOUBLE | Rate on ordered levels, 0 to n-1 |
| `jev_keys_overlap(a_keys, b_keys)` | BOOLEAN | Declare a key condition for pruning |

Inputs can be columns or STRUCTs such as `{'title': title, 'body': body}`; field names are sent
with the values. Options are a constant STRUCT, for example `{'threshold': 0.3}`. Answers are
reused within a query for repeated inputs. See [Functions](docs/functions.md).

## Using an LLM backend

Use a Chat Completions endpoint with structured outputs. This backend generates
answers and usually takes longer and costs more than Jev. Its probabilities are
model estimates.

```sql
CREATE SECRET llm (TYPE openai, PROVIDER env); -- reads OPENAI_API_KEY
SET jevdb_backend='llm';
SET jevdb_endpoint='https://api.openai.com/v1/';
SET jevdb_model='<model-with-structured-outputs>';
-- The same jev_holds, jev_probability, jev_choose and jev_score SQL applies.
```

## Optional pruning and cascade

Ordinary SQL conditions are applied before any pair is judged, including joins
written after a semantic join that restrict one of its inputs.

Queries run without key columns. To screen pairs with supplied nullable list
columns, add `jev_keys_overlap(a.keys,b.keys)`; a NULL list keeps the pair eligible.
See [Pruning](docs/pruning.md).

A cascade sends uncertain probabilities to a second-stage model and can calibrate
its bounds from a query sample. See [Cascade](docs/cascade.md). Another extension
can supply pair probabilities through the [custom scorer API](docs/scorers.md).

## Settings

Request layout, batch size, concurrency, threshold, timeouts and logging are DuckDB settings
(`SET jevdb_layout = 'auto'`, `SET jevdb_threads = 32`, ...). Decision settings can also be
overridden per call in the options STRUCT. See [Settings](docs/settings.md). Requests retry transient failures; exhausted
requests abort the query by default.

To run the JEVDB-Flash configuration reported in the paper, see the end of the
[Settings](docs/settings.md) page.

## Citation

```bibtex
@article{jevdb2026,
  title   = {Prune First, Decide Fast: Scalable Semantic Query Processing with JevDB},
  author  = {Wang, Zhengle and Yan, Hanxu and Zhao, Fuheng and Liu, Chunwei},
  journal = {arXiv preprint arXiv:2610.02046},
  year    = {2026}
}
```

## About

JEVDB is research code from an academic project at Purdue University. It is not
affiliated with or endorsed by TypeSafe AI.
