cmake_minimum_required(VERSION 3.16)
project(JevDBExampleScorer)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../src/include)
build_loadable_extension(jevdb_example_scorer "" scorer.cpp)
