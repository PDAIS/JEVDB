# Python package

The `jevdb` wheel bundles the C++ extension and requires `duckdb==1.4.3`.
`jevdb.connect()` loads that bundled binary into a DuckDB connection; it does not
download or compile code at runtime and makes no model requests.

```sh
pip install jevdb
```

```python
import jevdb

con = jevdb.connect()
con.execute("CREATE SECRET jev (TYPE jev, PROVIDER env)")
con.execute("SELECT jev_holds('The review is positive', 'A wonderful film')").show()
```

Set `JEV_API_KEY` before creating the secret. The initial wheel release targets
Linux x86-64. Installing from a source distribution requires a C++20 compiler,
OpenSSL development headers, and network access to the pinned DuckDB source.

## Build and test locally

Before the first PyPI release, install a locally built wheel:

```sh
python -m pip install build
python -m build
python -m pip install dist/*.whl
python -m pip install pandas ipython nbformat
python -m unittest tools.test_python_package -v
```

The build uses the existing C++ extension and downloads DuckDB 1.4.3 with a fixed
SHA-256 checksum. The source distribution can rebuild the wheel without an
existing checkout or `build/` directory. Model tests use a localhost fixture.

A wheel built on a recent Linux host may require newer system libraries than
Colab provides. The `Python package` GitHub Actions workflow builds in the
`manylinux_2_28_x86_64` image, bundles OpenSSL dependencies, then tests the wheel.
`tools/repair_python_wheel.py` restores DuckDB's appended metadata after
`auditwheel` changes the ELF file and rebuilds the wheel's RECORD hashes.

## First PyPI release

The first release must be uploaded before `pip install jevdb` works from PyPI.
The workflow can build and test artifacts without publishing them.

1. In a PyPI account, add a pending Trusted Publisher for project `jevdb`, owner
   `PDAIS`, repository `JEVDB`, workflow `python-package.yml`, environment `pypi`.
2. Create the GitHub Actions environment `pypi` and run `Python package` with
   `publish` left off. Review the tested distribution artifacts.
3. Run the workflow with `publish` enabled to upload that version to PyPI.

Version `0.1.0` is defined in `pyproject.toml`, `python/jevdb/__init__.py`, and
`extension_config.cmake`; update these together for subsequent releases.

See [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
for the account setup.
