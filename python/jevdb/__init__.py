"""Load the bundled JEVDB extension into its matching DuckDB Python engine."""

import ctypes
from pathlib import Path
import ssl

import duckdb

__version__ = "0.1.0"
_duckdb_library = None


def connect(database=":memory:"):
    """Return a DuckDB connection with JEVDB loaded, without making API calls.

    Set JEV_API_KEY or create a DuckDB secret before executing semantic SQL.
    The returned object is an ordinary duckdb.DuckDBPyConnection.
    """
    if duckdb.__version__ != "1.4.3":
        raise RuntimeError(
            "JEVDB requires DuckDB 1.4.3. Install the package dependencies, then "
            "restart the Python or Colab session if DuckDB was already imported."
        )
    extension = Path(__file__).with_name("jevdb.duckdb_extension")
    if not extension.is_file():
        raise RuntimeError("The JEVDB binary is missing. Install a built jevdb wheel with pip.")

    # Make the Python engine's C++ symbols available to the extension on Linux.
    global _duckdb_library
    if _duckdb_library is None:
        import _duckdb

        _duckdb_library = ctypes.CDLL(_duckdb.__file__, mode=ctypes.RTLD_GLOBAL)
    con = duckdb.connect(str(database), config={"allow_unsigned_extensions": True})
    try:
        con.execute("LOAD '" + str(extension).replace("'", "''") + "'")
        # Bundled OpenSSL must use the runtime's trust store, not the build host's.
        ca_file = ssl.get_default_verify_paths().cafile
        if ca_file:
            literal = "'" + ca_file.replace("'", "''") + "'"
            con.execute("SET jevdb_ca_cert_file = " + literal)
            con.execute("SET jevdb_cascade_ca_cert_file = " + literal)
    except Exception:
        con.close()
        raise
    return con
