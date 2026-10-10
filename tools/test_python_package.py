"""Test the installed wheel with the notebook's queries and local API fixture."""

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import ssl
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from unittest.mock import patch

import nbformat
import pandas as pd
from IPython.display import Code

import jevdb
from tools.test_release import Handler, literal


class PythonPackageTest(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {"JEV_API_KEY": "local-wheel-test"}))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        display = self.stack.enter_context(patch("IPython.display.display"))
        temp = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(Path, "cwd", return_value=temp))
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.addCleanup(self.server.server_close)
        self.server.lock = threading.Lock()
        self.server.entries = []
        self.server.statuses = [200]
        self.server.retry_after = None
        self.server.invalid = None
        worker = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .01})
        worker.start()
        self.addCleanup(worker.join)
        self.addCleanup(self.server.shutdown)
        self.csv = temp / "Reviews.csv"
        pd.DataFrame([
            (str(i), f"good movie {i}" if i < 5 else f"bad movie {i}", "POSITIVE")
            for i in range(201)
        ], columns=["reviewId", "reviewText", "scoreSentiment"]).to_csv(self.csv, index=False)
        notebook = Path(__file__).resolve().parents[1] / "examples/jevdb_colab.ipynb"
        nb = nbformat.read(notebook, as_version=4)
        nbformat.validate(nb)
        self.cells = {c.id: c for c in nb.cells if c.cell_type == "code"}
        self.con = jevdb.connect()
        self.addCleanup(self.con.close)
        work_dir = temp / "jevdb_colab"
        work_dir.mkdir()
        self.namespace = {
            "con": self.con, "work_dir": work_dir,
            "Code": Code, "display": display, "sql_literal": literal,
        }
        exec(self.cells["configure-model"].source, self.namespace)
        self.con.execute(f"SET jevdb_endpoint='http://127.0.0.1:{self.server.server_port}/v1/'")
        with patch("urllib.request.urlretrieve", side_effect=lambda url, dest: shutil.copyfile(self.csv, dest)):
            exec(self.cells["load-data"].source, self.namespace)

    def run_query(self, **controls):
        source = self.cells["run-query"].source
        for name, value in controls.items():
            source = "\n".join(
                f"{name} = {value!r}" if line.startswith(f"{name} = ") else line
                for line in source.splitlines()
            )
        exec(source, self.namespace)
        return self.namespace

    def test_notebook_setup_and_query(self):
        self.assertEqual(self.server.entries, [])
        self.assertEqual(self.con.execute(
            "SELECT current_setting('jevdb_k'), current_setting('jevdb_threads')"
        ).fetchone(), (100, 20))
        ca_file = ssl.get_default_verify_paths().cafile
        if ca_file:
            self.assertEqual(self.con.execute("SELECT current_setting('jevdb_ca_cert_file')").fetchone()[0], ca_file)
        os.environ["JEV_API_KEY"] = "must-not-override-the-connection-secret"
        result = self.run_query()
        self.assertEqual(result["results"].reviewId.tolist(), [str(i) for i in range(5)])
        self.assertEqual(result["f1"], 1)
        stats = result["stats"].iloc[0]
        self.assertEqual((stats["records"], stats["requests"], stats["errors"]), (100, 1, 0))
        self.assertAlmostEqual(result["estimated_cost_usd"], 10 * .042 / 1_000_000)
        for _, raw, auth, _ in self.server.entries:
            self.assertEqual(auth, "Bearer local-wheel-test")
            self.assertNotIn("scoreSentiment", raw)
            self.assertNotIn("reviewId", raw)

    def test_default_batch_size(self):
        self.run_query(num_reviews=201)
        sizes = sorted(len(json.loads(raw)["questions"]) for _, raw, _, _ in self.server.entries)
        self.assertEqual(sizes, [1, 100, 100])

    def test_custom_condition_and_empty_results(self):
        result = self.run_query(condition="The critic's review praises the acting.")
        self.assertIsNone(result["f1"])
        result = self.run_query(threshold=.95)
        self.assertTrue(result["results"].empty)
        self.assertEqual(result["f1"], 0)

    def test_wrong_duckdb_version_requests_restart(self):
        with patch.object(jevdb.duckdb, "__version__", "0.0.0"):
            with self.assertRaisesRegex(RuntimeError, "restart"):
                jevdb.connect()


if __name__ == "__main__":
    unittest.main()
