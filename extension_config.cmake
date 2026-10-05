duckdb_extension_load(jevdb SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR} DONT_LINK EXTENSION_VERSION 0.1.0)

option(JEVDB_BUILD_EXAMPLE_SCORER "Build the shared-prefix scorer example" OFF)
if(JEVDB_BUILD_EXAMPLE_SCORER)
  duckdb_extension_load(jevdb_example_scorer SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/examples/scorer DONT_LINK EXTENSION_VERSION 0.1.0)
endif()
