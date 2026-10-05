"""Regressions for layout-independent answers and exact request-body logging."""

import json
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from http.server import HTTPServer
from pathlib import Path

import jev_standin


class StandinTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.log_path = Path(self.directory.name) / 'requests.jsonl'
        jev_standin.LOG = self.log_path
        self.server = HTTPServer(('127.0.0.1', 0), jev_standin.H)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': 0.01})
        self.thread.start()
        self.client = HTTPConnection(*self.server.server_address, timeout=5)

    def tearDown(self):
        self.client.close()
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()
        self.directory.cleanup()

    def post(self, raw_body):
        self.client.request('POST', '/predict', body=raw_body, headers={'Content-Type': 'application/json'})
        response = self.client.getresponse()
        self.assertEqual(response.status, 200)
        return json.loads(response.read())['answers']

    def test_join_layouts_judge_the_same_records(self):
        for left, right in [('good report', 'bad issue'), ('bad report', 'good issue'), ('bad report', 'bad issue')]:
            layouts = {
                'sep': ({'left': left, 'right': right}, 'Does the condition hold?'),
                'pack': ({}, {'left': left, 'right': right, 'question': 'Does the condition hold?'}),
                'star_left': ({'left': left}, {'right': right, 'question': 'Does the condition hold?'}),
                'star_right': ({'right': right}, {'left': left, 'question': 'Does the condition hold?'}),
            }
            expected = 0.9 if (left, right) != ('bad report', 'bad issue') else 0.1
            for layout, (state, instructions) in layouts.items():
                with self.subTest(layout=layout, left=left, right=right):
                    body = {'state': {'condition': 'Match records', **state},
                            'questions': {'p0': {'type': 'noul', 'instructions': instructions}}}
                    self.assertEqual(self.post(json.dumps(body).encode()), {'p0': {'noul': expected}})

    def test_prompts_do_not_affect_record_answers(self):
        body = {
            'state': {'condition': 'good!!!!!'},
            'questions': {
                'p0': {'type': 'noul', 'instructions': {'record': 'bad', 'question': 'good!!!!!'}},
                'p1': {'type': 'score', 'instructions': {'record': 'bad!!', 'question': '!!!!!'},
                       'criteria': ['zero', 'one', 'two', 'three']},
            },
        }
        self.assertEqual(self.post(json.dumps(body).encode()), {'p0': {'noul': 0.1}, 'p1': {'score': 2.0}})

    def test_choice_and_score_read_state(self):
        body = {
            'state': {'condition': 'Judge the record', 'record': '高!!!'},
            'questions': {
                'p0': {'type': 'choice', 'instructions': 'Choose a label', 'criteria': {'高': 'high', '低': 'low'}},
                'p1': {'type': 'score', 'instructions': 'Assign a score', 'criteria': ['zero', 'one', 'two']},
            },
        }
        self.assertEqual(self.post(json.dumps(body).encode()), {'p0': {'choice': '高'}, 'p1': {'score': 2.0}})

    def test_single_table_questions_keep_separate_records(self):
        body = {
            'state': {'condition': 'Judge each record'},
            'questions': {
                'p0': {'type': 'noul', 'instructions': {'record': 'good', 'question': 'Judge'}},
                'p1': {'type': 'noul', 'instructions': {'record': 'bad', 'question': 'Judge'}},
            },
        }
        self.assertEqual(self.post(json.dumps(body).encode()), {'p0': {'noul': 0.9}, 'p1': {'noul': 0.1}})

    def test_log_preserves_original_bytes(self):
        payloads = [
            b'{ "state": {"condition":"judge", "record":"\\u597d"},\n'
            b'  "questions": {"p0":{"type":"noul","instructions":"judge"}} }\n',
            '{"questions":{"p0":{"instructions":"judge","type":"noul"}},'
            '"state":{"record":"好","condition":"judge"}}'.encode('utf-8'),
        ]
        self.assertEqual(json.loads(payloads[0]), json.loads(payloads[1]))
        for payload in payloads:
            self.post(payload)
        entries = self.log_path.read_text(encoding='utf-8').splitlines()
        self.assertEqual(len(entries), 2)
        self.assertEqual([json.loads(entry)['raw_body'].encode('utf-8') for entry in entries], payloads)


if __name__ == '__main__':
    unittest.main()
