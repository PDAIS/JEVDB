# Semantic SQL functions

`jev_holds(condition, input [, other_input] [, options])` returns BOOLEAN.
`jev_probability` takes the same arguments and returns DOUBLE. Jev supplies a
probability; the LLM backend supplies its own estimate between 0 and 1.
A probability at or above `jevdb_threshold` makes `jev_holds` true.

`jev_choose(condition, input, choices [, options])` returns one VARCHAR label.
Choices are a constant MAP, a list of labels, or `[[label, description], ...]`,
with 2–255 unique labels.

`jev_score(condition, input, levels [, options])` returns a DOUBLE between zero
and the number of levels minus one. Levels are an ordered constant list of 2–10
strings, STRUCTs, or lists of strings.

```sql
SELECT jev_holds('The review is positive', review),
       jev_probability('The review is positive', review),
       jev_choose('Classify sentiment', review, ['positive','negative','mixed']),
       jev_score('Rate enthusiasm', review, ['low','medium','high'])
FROM reviews;

SELECT a.id,b.id FROM reviews a JOIN reviews b
  ON a.id < b.id AND jev_holds('The reviews agree', a.review,b.review);
```

A scalar column supplies its column name. STRUCT inputs preserve names and order,
with values rendered as strings; `condition` and `question` are reserved field
names. Pair-side STRUCTs with multiple fields become `name: value; name: value`.
A NULL input or NULL STRUCT field returns SQL NULL without a model request.
Repeated input values share their answer within a query; duplicate SQL result
rows are retained. Each query and prepared execution starts with an empty cache.
`EXPLAIN` and `PREPARE` send no requests.

Conditions, criteria and options must be constant. Function options use setting
names without `jevdb_`, for example `{'threshold':0.3}`. A third STRUCT with a
recognized option field is single-record options; to use that field name in a
pair-side STRUCT, supply a fourth options argument explicitly.

The semantic operators handle one positive `jev_holds` predicate joined by AND
to ordinary predicates, in INNER, SEMI and ANTI joins and WHERE filters. Automatic
calibration uses these operators; scalar calls use fixed bounds. See
[Cascade](cascade.md), [Settings](settings.md), [Pruning](pruning.md) and
[Custom scorers](scorers.md).

## Semantic filters

A single-input `jev_holds` in `WHERE` is a semantic filter:

```sql
SELECT id FROM reviews
WHERE created_at >= DATE '2026-01-01' AND jev_holds('The review is positive', review);
```

Ordinary predicates in the same `WHERE` are applied first, so only the rows that
pass them are judged. Each distinct input is judged once; rows with a NULL input
are not judged and are not returned. `NOT jev_holds(...)` and predicates under
`OR` are evaluated row by row with the same answers and the same reuse.

## Decisions under ORDER BY ... LIMIT

A decision in the select list of a query that returns the first k rows of an
ordering is asked only for those k rows, when the ordering does not depend on the
decision:

```sql
SELECT id, jev_choose('Classify the sentiment', review, ['positive', 'negative']) AS label
FROM reviews ORDER BY created_at DESC LIMIT 100;   -- 100 questions, whatever the table size
```

This applies when the select list holds one decision beside plain columns, the
ordering reads plain columns and there is no OFFSET. Ordering or filtering by the
decision itself needs every row and is evaluated in full.

