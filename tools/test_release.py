"""Local regressions for request failures, LLM decisions and optional keys/scorers."""

from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import threading
import time
import unittest

PROJECT = Path(__file__).resolve().parents[1]
CLI = PROJECT / 'build/dependencies/cli/duckdb'
EXTENSION = PROJECT / 'build/release/extension/jevdb/jevdb.duckdb_extension'
SCORER = PROJECT / 'build/release/extension/jevdb_example_scorer/jevdb_example_scorer.duckdb_extension'


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def do_POST(self):
        raw = self.rfile.read(int(self.headers['Content-Length'])).decode()
        body = json.loads(raw)
        with self.server.lock:
            index = len(self.server.entries)
            self.server.entries.append((self.path, raw, self.headers.get('Authorization'), time.monotonic()))
        status = self.server.statuses[min(index, len(self.server.statuses) - 1)]
        if status is None:
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            return
        if self.path.endswith('/systemone'):
            answers = {}
            for qid, question in body['questions'].items():
                fields = {name: value for name, value in body['state'].items() if name != 'condition'}
                if isinstance(question['instructions'], dict):
                    fields.update((name, value) for name, value in question['instructions'].items() if name != 'question')
                text = ' '.join(fields.values())
                if question.get('type') == 'choice':
                    labels = list(question['criteria'])
                    answers[qid] = {'choice': labels[0] if 'good' in text else labels[1]}
                else:
                    answers[qid] = {'noul': .9 if 'good' in text else .1}
            response = {'answers': answers, 'usage': {'input_tokens': 10}}
        else:
            payload = json.loads(body['messages'][1]['content'])
            schema = body['response_format']['json_schema']['schema']['properties']['answers']['items']
            answers = []
            for item in payload['items']:
                text = item.get('left', '') + item.get('right', '') + ' '.join(f['value'] for f in item.get('fields', []))
                if schema['type'] == 'string':
                    answers.append(next((v for v in schema['enum'] if v in text), schema['enum'][-1]))
                elif schema['type'] == 'boolean':
                    answers.append('good' in text)
                elif schema['maximum'] > 1:
                    answers.append(min(text.count('!'), schema['maximum']))
                else:
                    answers.append(.9 if 'good' in text else .1)
            if self.server.invalid == 'length':
                answers = answers[:-1]
            elif self.server.invalid == 'range':
                answers = [2 for answer in answers]
            response = {'choices': [{'message': {'content': json.dumps({'answers': answers})}}],
                        'usage': {'prompt_tokens': 10}}
        encoded = json.dumps(response).encode()
        self.send_response(status)
        if self.server.retry_after and status != 200:
            self.send_header('Retry-After', self.server.retry_after)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class ReleaseBehavior(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.lock = threading.Lock()
        self.server.entries = []
        self.server.statuses = [200]
        self.server.retry_after = None
        self.server.invalid = None
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .01})
        self.thread.start()
        self.endpoint = f'http://127.0.0.1:{self.server.server_port}/v1/'

    def tearDown(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()

    def command(self, query, settings):
        env = {k: v for k, v in os.environ.items() if not k.startswith(('JEV', 'OPENAI'))}
        env.update(JEV_API_KEY='local-jev', OPENAI_API_KEY='local-llm')
        prefix = (f'LOAD {literal(EXTENSION)}; SET jevdb_endpoint={literal(self.endpoint)}; '
                  'SET jevdb_threads=1; SET jevdb_max_retries=2; SET jevdb_retry_max_delay=0;')
        return [str(CLI), '-unsigned', '-batch', '-bail', '-csv', '-noheader', ':memory:'], prefix + settings + query, env

    def execute(self, query, settings=''):
        arguments, script, env = self.command(query, settings)
        return subprocess.run(arguments, input=script, text=True, capture_output=True, env=env, timeout=30)

    def sql(self, query, settings=''):
        result = self.execute(query, settings)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.strip().splitlines()

    def test_transient_statuses_and_connection_failures_retry_identical_body(self):
        for failures in ([429, 200], [529, 503, 200], [None, 200]):
            with self.subTest(failures=failures):
                self.server.entries.clear()
                self.server.statuses = failures
                rows = self.sql("SELECT jev_probability('p','good'); SELECT requests,retries,errors FROM jevdb_stats();")
                self.assertEqual(rows, ['0.9', f'1,{len(failures)-1},0'])
                self.assertEqual(len(self.server.entries), len(failures))
                self.assertEqual(len({raw for path, raw, auth, at in self.server.entries}), 1)

    def test_retry_after_is_respected_even_when_backoff_cap_is_zero(self):
        self.server.statuses = [429, 200]
        self.server.retry_after = '1'
        self.assertEqual(self.sql("SELECT jev_holds('p','good');"), ['true'])
        self.assertGreaterEqual(self.server.entries[1][3] - self.server.entries[0][3], .95)

    def test_retry_after_wait_is_capped(self):
        self.server.statuses = [429, 200]
        self.server.retry_after = '30'
        self.assertEqual(self.sql("SELECT jev_holds('p','good');", 'SET jevdb_retry_after_max=1;'), ['true'])
        waited = self.server.entries[1][3] - self.server.entries[0][3]
        self.assertGreaterEqual(waited, .95)
        self.assertLess(waited, 5)

    def test_backoff_waits_between_half_and_all_of_the_delay(self):
        self.server.statuses = [503, 200]
        self.assertEqual(self.sql("SELECT jev_holds('p','good');", 'SET jevdb_retry_max_delay=2;'), ['true'])
        waited = self.server.entries[1][3] - self.server.entries[0][3]
        self.assertGreaterEqual(waited, .45)
        self.assertLess(waited, 1.5)

    def test_interrupt_ends_the_wait_between_attempts(self):
        self.server.statuses = [429]
        self.server.retry_after = '30'
        arguments, script, env = self.command("SELECT jev_holds('p','good');", '')
        process = subprocess.Popen(arguments, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        process.stdin.write(script)
        process.stdin.close()
        deadline = time.monotonic() + 10
        while not self.server.entries and time.monotonic() < deadline:
            time.sleep(.05)
        self.assertEqual(len(self.server.entries), 1)
        time.sleep(.2)
        process.send_signal(signal.SIGINT)
        process.wait(timeout=5)
        self.assertIn('nterrupt', process.stderr.read())
        process.stdout.close()
        process.stderr.close()
        self.assertEqual(len(self.server.entries), 1)

    def questions(self):
        return [q for path, raw, auth, at in self.server.entries for q in json.loads(raw)['questions'].values()]

    def test_decision_in_select_list_is_asked_only_for_top_rows(self):
        setup = ("CREATE TABLE reviews AS SELECT i, i % 17 AS priority, "
                 "CASE WHEN i % 2 = 0 THEN 'good-' ELSE 'bad-' END || i::VARCHAR AS text FROM range(600) t(i); "
                 "CREATE TABLE tags AS SELECT i FROM range(600) t(i); ")
        choose = "jev_choose('Classify', text, ['positive','negative'])"
        text = {i: ('good-' if i % 2 == 0 else 'bad-') + str(i) for i in range(600)}
        label = lambda value: 'positive' if value.startswith('good') else 'negative'
        top = lambda k: sorted(range(600), key=lambda i: (-(i % 17), i))[:k]
        group = [min(text[i] for i in range(600) if i % 17 == p) for p in range(3)]
        cases = [(f"SELECT i, {choose} FROM reviews ORDER BY priority DESC, i LIMIT 100",
                  [f'{i},{label(text[i])}' for i in top(100)]),
                 (f"SELECT r.i, {choose} FROM reviews r JOIN tags t ON r.i = t.i ORDER BY r.priority DESC, r.i LIMIT 7",
                  [f'{i},{label(text[i])}' for i in top(7)]),
                 ("WITH g AS (SELECT priority, min(text) AS text FROM reviews GROUP BY priority) "
                  f"SELECT priority, {choose} FROM g ORDER BY priority LIMIT 3",
                  [f'{p},{label(group[p])}' for p in range(3)])]
        for threads in (1, 4):
            for query, expected in cases:
                with self.subTest(threads=threads, query=query):
                    self.server.entries.clear()
                    self.assertEqual(self.sql(setup + query + ';', f'SET threads={threads}; SET jevdb_threads=2;'), expected)
                    self.assertEqual(len(self.questions()), len(expected))

    def test_deferred_decision_keeps_nulls_duplicates_and_empty_results(self):
        setup = ("CREATE TABLE reviews AS SELECT * FROM (VALUES (0,'bad',1),(1,'bad',NULL),(2,'good',10),(3,'good',10),"
                 "(4,NULL,9),(5,'bad',8)) t(i,text,priority); CREATE TABLE tags AS SELECT i FROM range(6) t(i); ")
        query = ("SELECT r.i, jev_choose('Classify', text, ['positive','negative']) FROM reviews r JOIN tags t ON r.i = t.i "
                 "{}ORDER BY priority DESC NULLS LAST, r.i LIMIT 4;")
        # rows 2 and 3 share one question, row 4 has no input, row 5 is the second question
        self.assertEqual(self.sql(setup + query.format('')), ['2,positive', '3,positive', '4,NULL', '5,negative'])
        self.assertEqual(len(self.questions()), 2)
        self.server.entries.clear()
        self.assertEqual(self.sql(setup + query.format('WHERE r.i < 0 ')), [])
        self.assertEqual(self.server.entries, [])

    def test_decision_stays_below_top_rows_when_the_order_or_other_columns_need_it(self):
        setup = ("CREATE TABLE reviews AS SELECT i, CASE WHEN i % 2 = 0 THEN 'good-' ELSE 'bad-' END || i::VARCHAR AS text "
                 "FROM range(30) t(i); CREATE TABLE tags AS SELECT i FROM range(30) t(i); ")
        source = "FROM reviews r JOIN tags t ON r.i = t.i"
        choose = "jev_choose('Classify', text, ['positive','negative'])"
        queries = [f"SELECT r.i, {choose} AS label {source} ORDER BY label, r.i LIMIT 2",
                   f"SELECT r.i, upper({choose}) AS label {source} ORDER BY r.i LIMIT 2",
                   f"SELECT r.i + 1 AS n, {choose} AS label {source} ORDER BY n LIMIT 2",
                   f"SELECT r.i, {choose} AS label {source} ORDER BY r.i LIMIT 2 OFFSET 1"]
        for query in queries:
            with self.subTest(query=query):
                self.server.entries.clear()
                self.assertEqual(len(self.sql(setup + query + ';')), 2)
                self.assertEqual(len(self.questions()), 30)
        self.server.entries.clear()
        two = f"SELECT r.i, {choose} AS a, jev_choose('Other', text, ['accept','reject']) AS b {source} ORDER BY r.i LIMIT 2"
        self.assertEqual(len(self.sql(setup + two + ';')), 2)
        self.assertEqual(len(self.questions()), 60)

    def test_exhaustion_and_permanent_failure_abort_by_default(self):
        for status, expected in ((503, 3), (401, 1)):
            self.server.statuses = [status]
            for expression in ("jev_holds('p','good')", "jev_probability('p','good')"):
                self.server.entries.clear()
                result = self.execute('SELECT ' + expression + ';')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('JEVDB', result.stderr)
                self.assertEqual(len(self.server.entries), expected)
        self.server.statuses = [503]
        self.assertEqual(self.sql("SELECT jev_holds('p','good'),jev_probability('p','good');",
                                  "SET jevdb_on_error='reject';"), ['false,NULL'])

    def test_fail_query_retains_all_received_jev_batches_in_raw_log(self):
        self.server.statuses = [400,200]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'requests.jsonl'
            result = self.execute("SELECT jev_holds('p',text) FROM (VALUES ('good a'),('good b'),('bad c')) r(text);",
                                  f"SET jevdb_k=1; SET jevdb_layout='pack'; SET jevdb_raw_dump_path={literal(path)};")
            self.assertNotEqual(result.returncode, 0)
            entries = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(entries), 3)
            self.assertEqual([e['raw_body'] for e in entries], [e[1] for e in self.server.entries])
            self.assertNotIn('local-jev', path.read_text())

    def test_llm_backend_all_decisions_fields_duplicates_nulls_and_auth(self):
        rows = self.sql("""
            CREATE TABLE r(id INT, text VARCHAR);
            INSERT INTO r VALUES (1,'good positive!!'),(2,'bad negative'),(3,'good positive!!'),(4,NULL);
            SELECT id,jev_holds('p',{'text':text}),jev_probability('p',{'text':text}),
                jev_choose('label',text,['positive','negative']),jev_score('rate',text,['low','medium','high'])
            FROM r ORDER BY id;
            SELECT requests,records,cache_hits,cascade_requests FROM jevdb_stats();
        """, "SET jevdb_backend='llm'; SET jevdb_model='local-decisions';")
        self.assertEqual(rows, ['1,true,0.9,positive,2.0', '2,false,0.1,negative,0.0',
                               '3,true,0.9,positive,2.0', '4,NULL,NULL,NULL,NULL', '3,6,6,0'])
        self.assertTrue(all(path == '/v1/chat/completions' and auth == 'Bearer local-llm'
                            for path, raw, auth, at in self.server.entries))
        payloads = [json.loads(json.loads(raw)['messages'][1]['content']) for path, raw, auth, at in self.server.entries]
        self.assertEqual(sum(len(p['items']) for p in payloads), 6)
        self.assertEqual(payloads[0]['items'][0]['fields'][0]['name'], 'text')

    def test_llm_first_and_second_stage_retries_are_counted_separately(self):
        self.server.statuses = [429, 200, 503, 200]
        rows = self.sql("SELECT jev_holds('p','good'); SELECT requests,retries,cascade_requests,cascade_retries FROM jevdb_stats();",
                        f"SET jevdb_backend='llm'; SET jevdb_model='local-probability'; "
                        f"SET jevdb_cascade_model='local-boolean'; SET jevdb_cascade_low=0; SET jevdb_cascade_high=1; "
                        f"SET jevdb_cascade_endpoint={literal(self.endpoint)}; SET jevdb_cascade_threads=1;")
        self.assertEqual(rows, ['true', '1,1,1,1'])
        self.assertEqual(self.server.entries[0][1], self.server.entries[1][1])
        self.assertEqual(self.server.entries[2][1], self.server.entries[3][1])

    def test_llm_invalid_answer_fails_and_budget_splits_batches(self):
        settings = "SET jevdb_backend='llm'; SET jevdb_model='local-decisions';"
        for failure in ('length', 'range'):
            self.server.invalid = failure
            self.assertNotEqual(self.execute("SELECT jev_probability('p','good');", settings).returncode, 0)
        self.server.invalid = None
        self.server.entries.clear()
        rows = self.sql("SELECT jev_probability('p',text) FROM (VALUES ('good11111111111111111'),('bad22222222222222222')) r(text);",
                        settings + 'SET jevdb_budget_tokens=10;')
        self.assertEqual(rows, ['0.9', '0.1'])
        self.assertEqual(len(self.server.entries), 2)

    def test_no_keys_judges_all_pairs_and_null_keys_preserve_complete_output(self):
        setup = """
            CREATE TABLE a(id INT, text VARCHAR, keys VARCHAR[]);
            CREATE TABLE b(id INT, text VARCHAR, keys VARCHAR[]);
            INSERT INTO a VALUES (1,'good a',['x']),(2,'good a',['x']),(3,'bad a',NULL);
            INSERT INTO b VALUES (10,'b',['x']),(11,'c',['y']);
        """
        query = "SELECT a.id,b.id FROM a JOIN b ON {}jev_holds('p',a.text,b.text) ORDER BY 1,2;"
        outputs = []
        for predicate, before in (('', ''), ('jev_keys_overlap(a.keys,b.keys) AND ', ''),
                                  ('jev_keys_overlap(a.keys,b.keys) AND ', 'UPDATE a SET keys=NULL; UPDATE b SET keys=NULL;')):
            self.server.entries.clear()
            rows = self.sql(setup + before + query.format(predicate) +
                            'SELECT pairs,cross_product_pairs,index_candidate_pairs FROM jevdb_stats();')
            outputs.append(rows)
        self.assertEqual(outputs[0], ['1,10','1,11','2,10','2,11','4,6,0'])
        self.assertEqual(outputs[1], ['1,10','2,10','3,0,4'])
        self.assertEqual(outputs[2][:-1], outputs[0][:-1])
        self.assertEqual(outputs[2][-1], '4,0,6')

    def test_semantic_filter_judges_distinct_non_null_inputs_after_ordinary_predicates(self):
        setup = """
            CREATE TABLE reviews AS SELECT * FROM (VALUES (1,'good film'),(2,'bad film'),(3,'good film'),
                (4,NULL),(5,'good plot'),(6,'bad plot')) v(id,review);
        """
        stats = ' ORDER BY 1; SELECT records,pairs,requests,cache_hits FROM jevdb_stats();'
        # query, returned ids, (records, pairs, requests, repeated inputs answered from the first answer)
        cases = {
            'a filter keeps the rows whose input holds':
                ("SELECT id FROM reviews WHERE jev_holds('p',review)", ['1', '3', '5'], '4,0,1,1'),
            'an ordinary predicate in the same WHERE is applied first':
                ("SELECT id FROM reviews WHERE id>2 AND jev_holds('p',review)", ['3', '5'], '3,0,1,0'),
            'the order of the two predicates does not matter':
                ("SELECT id FROM reviews WHERE jev_holds('p',review) AND id>2", ['3', '5'], '3,0,1,0'),
            'a threshold option applies to the filter':
                ("SELECT id FROM reviews WHERE jev_holds('p',review,{'threshold':0.95})", [], '4,0,1,1'),
            'a negated filter returns the rows that do not hold':
                ("SELECT id FROM reviews WHERE NOT jev_holds('p',review)", ['2', '6'], '4,0,1,1'),
            'a filter under OR keeps the rows either side accepts, and skips a row the other side accepted':
                ("SELECT id FROM reviews WHERE id=2 OR jev_holds('p',review)", ['1', '2', '3', '5'], '3,0,1,1'),
        }
        for name, (query, ids, expected) in cases.items():
            with self.subTest(name):
                self.server.entries.clear()
                rows = self.sql(setup + query + stats)
                self.assertEqual(rows, ids + [expected])
                questions = [json.loads(raw)['questions'] for path, raw, auth, at in self.server.entries]
                self.assertEqual(sum(map(len, questions)), int(expected.split(',')[0]))
        self.server.entries.clear()
        plan = '\n'.join(self.sql(setup + "EXPLAIN SELECT id FROM reviews WHERE id>2 AND jev_holds('p',review);"))
        self.assertIn('JEV_SEMANTIC_FILTER', plan)
        self.assertEqual(self.server.entries, [])  # EXPLAIN sends no request

    def test_ordinary_joins_restrict_semantic_inputs_before_judging(self):
        setup = """
            CREATE TABLE rep AS SELECT i AS id, CASE WHEN i % 3 = 0 THEN 'good report ' ELSE 'report ' END || i AS text,
                i % 4 AS team, NULL::VARCHAR[] AS keys FROM range(12) t(i);
            CREATE TABLE iss AS SELECT i AS id, 'issue ' || i AS text, NULL::VARCHAR[] AS keys FROM range(20) t(i);
            CREATE TABLE s(issue_id INT, note VARCHAR);
            INSERT INTO s VALUES (1,'a'),(1,'b'),(7,'c'),(13,'d'),(NULL,'e');
            CREATE TABLE c(note VARCHAR, kind VARCHAR);
            INSERT INTO c VALUES ('a','x'),('c','x'),('d','y');
            CREATE TABLE team(id INT);
            INSERT INTO team VALUES (0),(2);
        """
        holds = "jev_holds('p',rep.text,iss.text)"
        pair = f"FROM rep JOIN iss ON {holds}"
        # query, distinct pairs judged with the reduction (None: only compared with the run without it)
        cases = {
            'a later inner join restricts the right input':
                (f"SELECT rep.id,iss.id,s.note {pair} JOIN s ON s.issue_id=iss.id", 12 * 3),
            'EXISTS restricts the right input':
                (f"SELECT rep.id,iss.id {pair} WHERE EXISTS (SELECT 1 FROM s WHERE s.issue_id=iss.id)", 12 * 3),
            'NOT EXISTS restricts the right input':
                (f"SELECT rep.id,iss.id {pair} WHERE NOT EXISTS (SELECT 1 FROM s WHERE s.issue_id=iss.id)", 12 * 17),
            'a chain of two joins behind the right input':
                (f"SELECT rep.id,iss.id,c.kind {pair} JOIN s ON s.issue_id=iss.id JOIN c ON c.note=s.note "
                 "WHERE c.kind='x'", 12 * 2),
            'a join restricts the left input':
                (f"SELECT rep.id,iss.id {pair} JOIN team ON team.id=rep.team", 6 * 20),
            'joins restrict both inputs':
                (f"SELECT rep.id,iss.id,s.note {pair} JOIN s ON s.issue_id=iss.id JOIN team ON team.id=rep.team", 6 * 3),
            'a join that reads both inputs stays above':
                (f"SELECT rep.id,iss.id,s.note {pair} JOIN s ON s.issue_id=iss.id+rep.id", 12 * 20),
            'columns of the moved join are dropped above it':
                (f"SELECT count(*),max(rep.id) {pair} JOIN s ON s.issue_id=iss.id", 12 * 3),
            'with a key condition of unknown keys':
                (f"SELECT rep.id,iss.id FROM rep JOIN iss ON jev_keys_overlap(rep.keys,iss.keys) AND {holds} "
                 "JOIN s ON s.issue_id=iss.id", None),
            'a semantic EXISTS below an ordinary join':
                ("SELECT iss.id,s.note FROM iss JOIN s ON s.issue_id=iss.id WHERE EXISTS "
                 "(SELECT 1 FROM rep WHERE jev_holds('p',rep.text,iss.text))", None),
        }
        for name, (query, expected) in cases.items():
            with self.subTest(name):
                outputs = {}
                for reduce in ('true', 'false'):
                    rows = self.sql(setup + query + ' ORDER BY ALL; SELECT pairs FROM jevdb_stats();',
                                    f'SET jevdb_reduce_inputs={reduce};')
                    outputs[reduce] = (rows[:-1], int(rows[-1]))
                self.assertEqual(outputs['true'][0], outputs['false'][0])
                self.assertLessEqual(outputs['true'][1], outputs['false'][1])
                if expected is not None:
                    self.assertEqual(outputs['true'][1], expected)

    def test_automatic_calibration_uses_materialized_distinct_pool_with_keys(self):
        rows = self.sql("""
            CREATE TABLE a AS SELECT * FROM (VALUES (1,'good',['x']),(2,'good',['x']),(3,'bad',['y'])) v(id,text,keys);
            CREATE TABLE b AS SELECT * FROM (VALUES ('right x',['x']),('right y',['y'])) v(text,keys);
            SELECT count(*) FROM a JOIN b ON jev_keys_overlap(a.keys,b.keys) AND jev_holds('p',a.text,b.text);
            SELECT pairs,calibration_items,cascade_items,index_candidate_pairs FROM jevdb_stats();
        """, f"SET jevdb_cascade_model='local-boolean'; SET jevdb_cascade_endpoint={literal(self.endpoint)}; "
             "SET jevdb_cascade_threads=1; SET jevdb_cascade_calibrate=16;")
        self.assertEqual(rows, ['2','2,2,2,0'])

    def test_custom_scorer_threshold_reuse_cascade_and_default_backend(self):
        rows = self.sql(f"""
            LOAD {literal(SCORER)};
            SELECT jev_probability('p','abcd','abxy'),jev_holds('p','abcd','abxy'),
                jev_holds('p','abcd','abxy',{{'threshold':0.6}});
            SELECT requests,pairs,cache_hits FROM jevdb_stats();
            SET jevdb_cascade_model='local-boolean';
            SET jevdb_cascade_endpoint={literal(self.endpoint)};
            SET jevdb_cascade_threads=1;
            SELECT jev_holds('p','good','goat');
            SELECT requests,cascade_items FROM jevdb_stats();
            SET jevdb_example_scorer=false;
            SET jevdb_cascade_model='';
            SELECT jev_probability('p','good','pair');
        """)
        self.assertEqual(rows, ['0.5,true,false', '0,1,2', 'true', '0,1', '0.9'])
        self.assertEqual([path for path, raw, auth, at in self.server.entries],
                         ['/v1/chat/completions', '/v1/systemone'])


if __name__ == '__main__':
    unittest.main()
