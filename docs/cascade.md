# Escalating uncertain decisions

A cascade uses first-stage probabilities to accept clear matches, reject clear
non-matches, and ask an OpenAI-compatible model about the interval in between.
It is optional and remains experimental. Identical second-stage requests can
receive different Boolean answers between runs. This can change calibrated
bounds, returned matches and missed matches, even with fixed first-stage scores
and calibration inputs. Evaluate both recall and precision on your workload
before adopting it.

```sql
CREATE SECRET llm (TYPE openai, PROVIDER env); -- OPENAI_API_KEY
SET jevdb_cascade_model='<model-with-structured-outputs>';
SET jevdb_cascade_calibrate=400;
SELECT r.id,i.id FROM reports r JOIN issues i
  ON jev_holds('The bulletin covers the reported defect',r.text,i.text);
```

With fixed bounds, probabilities below low are false, probabilities at or above
high are true, and `[low,high)` is judged by the second stage. With automatic
calibration, the engine collects the complete distinct semantic input pool,
samples at most `calibrate` available probabilities, and chooses bounds subject
to the configured recall and precision losses on that sample. Judged sample
answers are reused when replaying results. These sample constraints are not
guarantees about full-query quality.

Automatic calibration requires a supported semantic JOIN or WHERE filter.
Projection-only calls use fixed bounds. NULL input values produce SQL NULL and
are not judged. Duplicate inputs share their probability and second-stage answer,
while duplicate SQL result rows are retained.

The second stage requests one Boolean per item in a strict JSON Schema.
Missing, invalid or short answers abort the query. Transient request failures use
the [retry settings](settings.md); the default exhausted-request behavior is an
error. `jev_probability`, `jev_choose` and `jev_score` use the first stage only.
Custom first-stage scorers can also feed this cascade; see [Scorers](scorers.md).
