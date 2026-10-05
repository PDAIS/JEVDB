#!/usr/bin/env bash
set -euo pipefail

jevdb_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cmake -S "$jevdb_root/build/dependencies/duckdb-1.4.3" -B "$jevdb_root/build/release" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_EXTENSIONS_ONLY=ON \
  -DEXTENSION_STATIC_BUILD=OFF -DBUILD_UNITTESTS=OFF -DBUILD_SHELL=OFF \
  -DDISABLE_BUILTIN_EXTENSIONS=ON \
  -DDUCKDB_EXTENSION_CONFIGS="$jevdb_root/extension_config.cmake" \
  -DOVERRIDE_GIT_DESCRIBE=v1.4.3 "$@"
jevdb_targets=(jevdb_loadable_extension)
if grep -q '^JEVDB_BUILD_EXAMPLE_SCORER:BOOL=ON$' "$jevdb_root/build/release/CMakeCache.txt"; then
  jevdb_targets+=(jevdb_example_scorer_loadable_extension)
fi
cmake --build "$jevdb_root/build/release" --target "${jevdb_targets[@]}" --parallel 4
