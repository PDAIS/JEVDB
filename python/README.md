# JEVDB

Scalable semantic SQL on decision models.

```sh
pip install jevdb
```

```python
import jevdb

con = jevdb.connect()
con.execute("CREATE SECRET jev (TYPE jev, PROVIDER env)")  # Reads JEV_API_KEY.
con.execute("SELECT jev_holds('The review is positive', 'Loved this film!')").show()
```

The wheel includes the compiled JEVDB extension and installs the matching DuckDB
version. Creating a connection loads the extension; no compiler or source checkout
is needed when installing a wheel. The initial binary release targets Linux x86-64,
including Google Colab. Source builds require CMake, a C++20 compiler and OpenSSL
development headers.

[Try the interactive Colab demo](https://colab.research.google.com/github/PDAIS/JEVDB/blob/main/examples/jevdb_colab.ipynb).

[Source and SQL documentation](https://github.com/PDAIS/JEVDB) · [Project website](https://jevdb.org)
