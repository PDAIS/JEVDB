cmake_minimum_required(VERSION 3.24)
project(JevDBPython LANGUAGES C CXX)

include(ExternalProject)
get_filename_component(JEVDB_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(DUCKDB_BUILD "${CMAKE_CURRENT_BINARY_DIR}/duckdb-build")
set(JEVDB_BINARY "${DUCKDB_BUILD}/extension/jevdb/jevdb.duckdb_extension")

ExternalProject_Add(jevdb_binary
  URL https://codeload.github.com/duckdb/duckdb/tar.gz/refs/tags/v1.4.3
  URL_HASH SHA256=b6a2afd09d9cf07e50d5cd07077df7f7697b61cca2eb00754f5adf89a1ae6c64
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/duckdb-source"
  BINARY_DIR "${DUCKDB_BUILD}"
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_STANDARD=20
    -DOPENSSL_ROOT_DIR=${OPENSSL_ROOT_DIR}
    -DBUILD_EXTENSIONS_ONLY=ON
    -DEXTENSION_STATIC_BUILD=OFF
    -DBUILD_UNITTESTS=OFF
    -DBUILD_SHELL=OFF
    -DDISABLE_BUILTIN_EXTENSIONS=ON
    -DDUCKDB_EXTENSION_CONFIGS=${JEVDB_ROOT}/extension_config.cmake
    -DOVERRIDE_GIT_DESCRIBE=v1.4.3-0-gd1dc88f950
  BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target jevdb_loadable_extension --parallel 4
  BUILD_ALWAYS TRUE
  BUILD_BYPRODUCTS "${JEVDB_BINARY}"
  INSTALL_COMMAND ""
)

install(FILES "${JEVDB_BINARY}" DESTINATION jevdb)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/duckdb-source/LICENSE"
        DESTINATION jevdb/licenses RENAME DuckDB.txt)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/duckdb-source/third_party/yyjson/LICENSE"
        DESTINATION jevdb/licenses RENAME yyjson.txt)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/duckdb-source/third_party/httplib/LICENSE"
        DESTINATION jevdb/licenses RENAME cpp-httplib.txt)
