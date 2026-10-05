# Custom first-stage scorers

Another DuckDB extension can register a function that supplies a probability for
each pair. JEVDB applies thresholds, reuses scores within the query and runs the
optional second stage. The HTTP backend is used when the binder returns nullptr.

Include `src/include/jevdb_scorer.hpp` and call
`RegisterFirstStageScorer(loader, binder)` after loading JEVDB. The binder receives
the client context, SQL function name and whether the input is pairwise. It returns
a `FirstStageScorer` with:

- `configuration` and `identity`: explicit strings distinguishing scorer behavior;
- `score_pairs(condition, pairs)`: one probability per pair, in supplied order.

Return finite probabilities from 0 through 1. Changes to scoring configuration
must change those explicit identity fields. Registration is attached to catalog
function metadata, so independently loaded extension libraries can use the API.

The [example](../examples/scorer/scorer.cpp) scores common string prefixes divided
by the longer string length. It supports pairwise holds/probability calls; it is
an API demonstration rather than a semantic model.

```sh
bash scripts/build.sh -DJEVDB_BUILD_EXAMPLE_SCORER=ON
```

```sql
LOAD '/absolute/path/to/jevdb/build/release/extension/jevdb/jevdb.duckdb_extension';
LOAD '/absolute/path/to/jevdb/build/release/extension/jevdb_example_scorer/jevdb_example_scorer.duckdb_extension';
SELECT jev_probability('shared prefix','abcd','abxy'); -- 0.5, no HTTP call
SET jevdb_example_scorer=false; -- use the configured HTTP backend
```

The example regression verifies scores, threshold overrides, query reuse,
second-stage decisions and the configured backend when the scorer is disabled.
