"""Media SQL and Jev wire-contract regressions against a local service."""

import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import struct
import tempfile
import threading
import unittest
import zlib

from tools import test_release

PROJECT = test_release.PROJECT
literal = test_release.literal


def png(width, height, color):
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data))
    pixels = (b'\0' + bytes(color) * width) * height
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!IIBBBBB', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b''))


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def do_POST(self):
        raw = self.rfile.read(int(self.headers['Content-Length'])).decode()
        body = json.loads(raw)
        with self.server.lock:
            self.server.entries.append((body, self.headers.get('TypeSafe-Danger-Zone-Images'), raw))
        assets = body.get('assets', {})
        answers = {}
        for qid, question in body['questions'].items():
            fields = {k: v for k, v in body['state'].items() if k != 'condition'}
            if isinstance(question['instructions'], dict):
                fields.update((k, v) for k, v in question['instructions'].items() if k != 'question')
            text = json.dumps(fields)
            colors = []
            for reference, asset in assets.items():
                colors.extend([self.server.colors.get(asset['data_base64'], 'unknown')] * text.count(reference))
            matches = len(set(colors)) == 1 if body['state']['condition'] == 'same' else 'red' in colors
            if question['type'] == 'choice':
                answers[qid] = {'choice': 'red' if matches else 'blue'}
            elif question['type'] == 'score':
                answers[qid] = {'score': 2.0 if matches else 0.0}
            else:
                answers[qid] = {'noul': .9 if matches else .1}
        payload = json.dumps({'answers': answers, 'usage': {'input_tokens': 10}}).encode()
        self.send_response(429 if self.server.retry and len(self.server.entries) == 1 else 200)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(payload)


class MediaBehavior(unittest.TestCase):
    command = test_release.ReleaseBehavior.command
    execute = test_release.ReleaseBehavior.execute
    sql = test_release.ReleaseBehavior.sql

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=PROJECT / 'build')
        self.path = Path(self.directory.name)
        self.red = self.path / 'red.png'
        self.blue = self.path / 'blue.png'
        self.red.write_bytes(png(280, 280, (255, 0, 0)))
        self.blue.write_bytes(png(280, 280, (0, 0, 255)))
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.entries = []
        self.server.lock = threading.Lock()
        self.server.retry = False
        self.server.colors = {base64.b64encode(p.read_bytes()).decode(): c
                              for p, c in ((self.red, 'red'), (self.blue, 'blue'))}
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': .01})
        self.thread.start()
        self.endpoint = f'http://127.0.0.1:{self.server.server_port}/v1/'

    def tearDown(self):
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()
        self.directory.cleanup()

    def media(self, path):
        return "jev_media('image', {'path': " + literal(path) + '})'

    def table(self):
        return (f"CREATE TABLE pictures AS SELECT * FROM (VALUES (1,{literal(self.red)}),"
                f"(2,{literal(self.blue)}),(3,{literal(self.red)}),(4,NULL)) t(id,path); ")

    def assert_assets(self):
        for body, header, raw in self.server.entries:
            self.assertEqual(header, 'true')
            self.assertIn('assets', body)
            content = json.dumps([body['state'], body['questions']])
            for reference, asset in body['assets'].items():
                self.assertIn(reference, content)
                self.assertEqual(asset['type'], 'image')
                self.assertEqual(asset['format'], 'png')
                self.assertTrue(base64.b64decode(asset['data_base64'], validate=True).startswith(b'\x89PNG'))
            self.assertNotIn(str(self.path), content)

    def test_filter_nulls_duplicates_and_query_local_cache(self):
        query = "SELECT id FROM pictures WHERE jev_holds('red', jev_media('image', {'path':path})) ORDER BY id;"
        rows = self.sql(self.table() + query + 'SELECT records,cache_hits FROM jevdb_stats();' + query)
        self.assertEqual(rows, ['1', '3', '2,1', '1', '3'])
        self.assertEqual(len(self.server.entries), 2)
        self.assert_assets()
        self.assertEqual([len(body['assets']) for body, _, _ in self.server.entries], [2, 2])

    def test_probability_choice_score_and_nested_records(self):
        photo = self.media(self.red)
        record = "{'description':'red object', 'details': {'photo': " + photo + '}}'
        query = (f"SELECT jev_probability('red', {record}), jev_choose('red', {record}, ['red','blue']), "
                 f"jev_score('red', {record}, ['low','medium','high']);")
        self.assertEqual(self.sql(query), ['0.9,red,2.0'])
        self.assert_assets()

    def test_media_type_survives_stored_columns(self):
        query = (f"CREATE TABLE pictures AS SELECT {self.media(self.red)} AS photo; "
                 "SELECT typeof(photo), jev_holds('red', photo) FROM pictures;")
        self.assertEqual(self.sql(query), ['JEV_MEDIA,true'])
        self.assert_assets()

    def test_lists_keep_media_and_unsupported_containers_fail(self):
        photo = self.media(self.red)
        self.assertEqual(self.sql(f"SELECT jev_holds('red',[{photo},{photo}]);"), ['true'])
        self.assert_assets()
        self.assertEqual(len(self.server.entries[0][0]['assets']), 1)
        self.server.entries.clear()
        query = f"SELECT jev_holds('red',map(['photo'],[{photo}]));"
        result = self.execute(query)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('STRUCTs or LISTs', result.stderr)
        result = self.execute(f"SELECT jev_score('red','record',[{photo},{photo}]);")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('choices or score levels', result.stderr)
        self.assertEqual(self.server.entries, [])

    def test_all_pair_layouts_preserve_images_and_deduplicate_uploads(self):
        setup = (f"CREATE TABLE a AS SELECT * FROM (VALUES (1,{self.media(self.red)}),"
                 f"(2,{self.media(self.blue)})) t(id,photo); CREATE TABLE b AS SELECT * FROM a;")
        query = "SELECT a.id,b.id FROM a JOIN b ON jev_holds('same',a.photo,b.photo) ORDER BY a.id,b.id;"
        for layout in ('pack', 'star', 'sep', 'auto'):
            for anchor in ('left', 'right'):
                with self.subTest(layout=layout, anchor=anchor):
                    self.server.entries.clear()
                    self.assertEqual(self.sql(setup + query, f"SET jevdb_layout='{layout}'; SET jevdb_anchor='{anchor}';"),
                                     ['1,1', '2,2'])
                    self.assert_assets()
                    if layout == 'pack':
                        self.assertEqual(len(self.server.entries[0][0]['assets']), 2)

    def test_text_image_join_and_semi_anti_joins(self):
        setup = (f"CREATE TABLE images AS SELECT * FROM (VALUES (1,{self.media(self.red)}),"
                 f"(2,{self.media(self.blue)})) t(id,photo); CREATE TABLE descriptions AS SELECT 'red object' AS text;")
        query = ("SELECT id FROM images i WHERE {}EXISTS (SELECT 1 FROM descriptions d "
                 "WHERE jev_holds('red',d.text,i.photo)) ORDER BY id;")
        self.assertEqual(self.sql(setup + query.format('') + query.format('NOT ')), ['1', '2'])
        self.assert_assets()

    def test_planning_and_filtered_rows_do_not_read_files(self):
        missing = self.media(self.path / 'missing.png')
        self.sql(f"EXPLAIN SELECT jev_holds('red',{missing}); PREPARE p AS SELECT jev_holds('red',{missing});")
        self.assertEqual(self.server.entries, [])
        query = (self.table() + f"INSERT INTO pictures VALUES (5,{literal(self.path / 'missing.png')}); "
                 "SELECT id FROM pictures WHERE id=1 AND jev_holds('red',jev_media('image',{'path':path}));")
        self.assertEqual(self.sql(query), ['1'])

    def test_ordinary_structs_are_text_and_do_not_get_image_headers(self):
        self.sql("SELECT jev_holds('red', {'type':'image','path':'missing.png'});")
        body, header, raw = self.server.entries[0]
        self.assertNotIn('assets', body)
        self.assertIsNone(header)
        self.assertEqual(body['questions']['p0']['instructions']['path'], 'missing.png')

    def test_invalid_inputs_and_unsupported_backends_fail_before_http(self):
        cases = [("jev_media('video',{'path':'clip.mp4'})", ''),
                 ("jev_media('image',{'url':'https://example.com/photo.png'})", ''),
                 ("jev_media('image',{'path':12})", ''),
                 (self.media('https://example.com/photo.png'), ''),
                 (self.media(self.path / 'missing.png'), ''),
                 (self.media(self.red), "SET jevdb_backend='llm';"),
                 (self.media(self.red), "SET jevdb_cascade_model='vision';"),
                 (self.media(self.red), 'SET enable_external_access=false;')]
        for value, settings in cases:
            with self.subTest(value=value, settings=settings):
                result = self.execute(f"SELECT jev_holds('red',{value});", settings)
                self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.server.entries, [])

    def test_file_format_and_dimensions_are_validated(self):
        bad = self.path / 'bad.png'
        for contents in (b'not an image', png(256, 256, (255, 0, 0)), png(1569, 280, (255, 0, 0))):
            bad.write_bytes(contents)
            self.assertNotEqual(self.execute(f"SELECT jev_holds('red',{self.media(bad)});").returncode, 0)
        self.assertEqual(self.server.entries, [])

    def test_image_byte_budget_splits_batches(self):
        image = self.path / 'large.png'
        image.write_bytes(png(1568, 1568, (255, 0, 0)))
        query = (f"SELECT jev_holds('red',{{'id':i,'photo':{self.media(image)}}}) FROM range(5) t(i);")
        self.sql(query, 'SET jevdb_k=100;')
        self.assertEqual([len(body['questions']) for body, _, _ in self.server.entries], [2, 2, 1])
        self.assert_assets()

    def test_retries_keep_assets_and_headers(self):
        self.server.retry = True
        self.assertEqual(self.sql(f"SELECT jev_holds('red',{self.media(self.red)});"), ['true'])
        self.assertEqual(len(self.server.entries), 2)
        self.assertEqual(self.server.entries[0][2], self.server.entries[1][2])
        self.assert_assets()


if __name__ == '__main__':
    unittest.main()
