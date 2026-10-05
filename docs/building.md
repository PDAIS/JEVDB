# Building JEVDB

JEVDB 0.1.0 is built for DuckDB 1.4.3. You need a C++17 compiler, CMake, Ninja,
OpenSSL headers and libraries, Python 3, and the matching DuckDB source and CLI.

```sh
mkdir -p build/dependencies
curl -fL https://codeload.github.com/duckdb/duckdb/tar.gz/refs/tags/v1.4.3 \
  -o build/dependencies/duckdb-v1.4.3.tar.gz
tar -xzf build/dependencies/duckdb-v1.4.3.tar.gz -C build/dependencies
# Download the DuckDB 1.4.3 CLI for your platform and put it in build/dependencies/cli.
bash scripts/build.sh -DOPENSSL_ROOT_DIR=/absolute/path/to/openssl
```

Use `OPENSSL_INCLUDE_DIR`, `OPENSSL_SSL_LIBRARY` and `OPENSSL_CRYPTO_LIBRARY` when
headers and libraries are installed separately. The result is
`build/release/extension/jevdb/jevdb.duckdb_extension`.

```sql
-- Start the matching CLI with -unsigned.
LOAD '/absolute/path/to/build/release/extension/jevdb/jevdb.duckdb_extension';
```

For a binary that does not depend on the build machine's OpenSSL, build with
`-DOPENSSL_USE_STATIC_LIBS=ON` and static OpenSSL libraries.

For the example scorer and local regressions:

```sh
bash scripts/build.sh -DJEVDB_BUILD_EXAMPLE_SCORER=ON
python3 -m unittest tools.test_release -v
python3 tools/standin/test_jev_standin.py -v
```

The regression services run on localhost with fixed answers and use no external
model APIs. To try the Jev stand-in directly:

```sh
python3 tools/standin/jev_standin.py 8080 /tmp/jev-requests.jsonl
```

```sql
SET jevdb_endpoint='http://127.0.0.1:8080';
SELECT jev_holds('positive','good'); -- true
```
