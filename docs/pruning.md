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
several such joins in a row are followed. A moved INNER join can repeat rows of
that input; repeated inputs share one answer.

A join whose conditions read both inputs stays above, but reduces each of them
first:

```sql
SELECT r.id,p.id FROM reports r JOIN plans p
  ON jev_holds('The plan applies to the report',r.text,p.applicability)
JOIN approvals a ON a.report_id=r.id AND a.plan_id=p.id;
```

Before any pair is judged, reports are semi-joined with `approvals` on
`a.report_id=r.id` and plans on `a.plan_id=p.id`, as if
`WHERE r.id IN (SELECT report_id FROM approvals) AND p.id IN (SELECT plan_id FROM approvals)`
had been written; only reports and plans named in some approval are paired. The
join itself stays above and still requires an approval naming the report and the
plan together. This applies to INNER joins, and to SEMI joins such as EXISTS or
IN over both inputs when planning makes them a plain join, through equality
conditions (`=` or `IS NOT DISTINCT FROM`) whose semantic side reads one input;
other conditions are only checked above. A condition such as `a.k=r.id+p.id`,
which reads both inputs in one expression, reduces neither. Each reduction is
placed at the input its conditions read, also inside a chain of semantic joins.
Joins that restrict the other relation, such
as `JOIN batches b ON b.id=a.batch WHERE b.open`, are applied to it first, so
their restriction reaches the semantic inputs as well. A join above it that
restricts one input reduces that input in the same way.

The other relation is evaluated again for each input it reduces, so it is used
only when it returns the same rows each time: tables combined by filters,
projections, joins, grouping with order-independent aggregates, DISTINCT and set
operations, without volatile functions or semantic decisions. A correlated EXISTS
over both inputs that planning keeps as a duplicate-eliminating join (`DELIM_JOIN`
in EXPLAIN) reduces nothing yet.

The query result does not change, including repeated rows and NULL keys.
`SET jevdb_reduce_inputs=false` keeps the written order, as does disabling
DuckDB's join ordering or filter pushdown.

`EXPLAIN` displays the semantic operators and materialized input pool.
`jevdb_stats()` exposes distinct judged inputs, actual posting visits and pair
visits, alongside retained row counts. These distinguish generation work from
model work; a key with many unknown values may save little generation work.
