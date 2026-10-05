"""Local Jev stand-in with deterministic answers from record fields.

Record fields come from state and each question's instructions; condition and
question text are excluded. noul is 0.9 when a record contains 'good', else 0.1.
Choice selects the first label in the record, else the last; score counts '!',
capped at levels-1. Each JSONL entry stores the original UTF-8 body in raw_body.

Usage: python3 jev_standin.py PORT LOG_PATH
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        raw_body = self.rfile.read(int(self.headers['Content-Length']))
        # JSON escaping keeps embedded newlines in one log entry without
        # changing the request's whitespace, field order or escape sequences.
        with open(LOG, 'a', encoding='utf-8') as log:
            log.write(json.dumps({'raw_body': raw_body.decode('utf-8')}) + '\n')
        body = json.loads(raw_body)
        answers = {}
        for qid, q in body['questions'].items():
            record = {name: value for name, value in body['state'].items() if name != 'condition'}
            instructions = q['instructions']
            # sep puts both records in state and uses a string for instructions.
            if isinstance(instructions, dict):
                record.update((name, value) for name, value in instructions.items() if name != 'question')
            text = json.dumps(record, ensure_ascii=False).lower()
            if q['type'] == 'noul':
                answers[qid] = {'noul': 0.9 if 'good' in text else 0.1}
            elif q['type'] == 'choice':
                labels = list(q['criteria'])
                answers[qid] = {'choice': next((l for l in labels if l.lower() in text), labels[-1])}
            else:
                answers[qid] = {'score': float(min(text.count('!'), len(q['criteria']) - 1))}
        out = json.dumps({'answers': answers, 'usage': {'input_tokens': 10}, 'model': 'standin'}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


if __name__ == '__main__':
    LOG = sys.argv[2]
    HTTPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()
