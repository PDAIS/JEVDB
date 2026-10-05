# Pruning with supplied keys

Queries without `jev_keys_overlap` judge all pairs allowed by their ordinary SQL
conditions. Pruning is optional: provide ordinary nullable VARCHAR list columns
and declare a necessary condition explicitly.

```sql
SELECT r.id,i.id FROM reports r JOIN issues i
  ON jev_keys_overlap(r.keys,i.keys)
 AND jev_holds('The bulletin covers the reported defect',r.text,i.text);
```

Either NULL key list means unknown and keeps the pair eligible. An empty list has
no overlap with a non-NULL list; NULL list elements do not match. If every key list
is NULL, the semantic input pool and SQL output are the same as without keys.
The supplied keys must describe a necessary condition; disjoint incorrect keys
can remove real matches. JEVDB consumes the columns without generating keys.

For supported single-stage and fixed-bound INNER joins, the operator uses an indexed key dimension to generate
possible pairs and checks the remaining key conditions. Rows on either side with
no possible partner are removed before pair generation. A fully unknown key
dimension is skipped when another dimension can screen pairs.

Automatic calibration materializes eligible rows and the distinct semantic pool
before sampling. This path retains the materialization cost.

The optimizer materializes eligible rows, judges distinct typed semantic inputs,
and attaches accepted inputs with a null-safe SEMI join. Duplicate rows and
nullable payload columns survive. SEMI and ANTI joins use the semantic operator
and query-local answer reuse. Direct positive predicates in AND conditions are
factored; OR, NOT, outer joins and aggregation boundaries use scalar execution
or stop key discovery. Automatic cascade calibration needs a supported semantic
JOIN or WHERE operator.

## Ordinary joins around a semantic join

A join written after a semantic join often restricts one of its inputs:

```sql
SELECT r.id,i.id FROM reports r JOIN issues i
  ON jev_holds('The bulletin covers the reported defect',r.text,i.text)
JOIN open_issues o ON o.issue_id=i.id;
```

JEVDB evaluates such a join inside the input it restricts before any pair is
judged, so issues without a partner in `open_issues` are never sent to the model.
This applies to INNER, SEMI and ANTI joins directly above the semantic join,
including EXISTS and NOT EXISTS, when their conditions read one semantic input;
several such joins in a row are followed. A join whose conditions read both inputs
stays above. The query result does not change. A moved INNER join can repeat rows
of that input; repeated inputs share one answer. `SET jevdb_reduce_inputs=false`
keeps the written order, as does disabling DuckDB's join ordering or filter
pushdown.

`EXPLAIN` displays the semantic operators and materialized input pool.
`jevdb_stats()` exposes distinct judged inputs, actual posting visits and pair
visits, alongside retained row counts. These distinguish generation work from
model work; a key with many unknown values may save little generation work.
