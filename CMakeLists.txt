cmake_minimum_required(VERSION 3.16)
project(JevDBExtension)
find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)

include_directories(src/include ${DUCKDB_MODULE_BASE_DIR}/third_party/httplib
                    ${DUCKDB_MODULE_BASE_DIR}/third_party/yyjson/include)
set(JEVDB_SOURCES src/jevdb_extension.cpp src/jev_client.cpp src/llm_client.cpp src/semantic_join.cpp src/sbf_optimizer.cpp src/jev_stats.cpp src/scorer.cpp
                 ${DUCKDB_MODULE_BASE_DIR}/third_party/yyjson/yyjson.cpp)
build_loadable_extension(jevdb "" ${JEVDB_SOURCES})
target_compile_definitions(jevdb_loadable_extension PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
target_link_libraries(jevdb_loadable_extension OpenSSL::SSL OpenSSL::Crypto Threads::Threads)
