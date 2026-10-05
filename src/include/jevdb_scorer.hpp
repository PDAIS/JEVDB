#pragma once

#include "duckdb/catalog/catalog_entry/scalar_function_catalog_entry.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include <functional>
#include <utility>

namespace duckdb {
namespace jevdb {

// A custom scorer supplies pairwise first-stage probabilities. Thresholding,
// query-local reuse and second-stage decisions stay in JEVDB.
struct FirstStageScorer {
	string configuration;
	string identity;
	std::function<vector<double>(const string &, const vector<std::pair<string, string>> &)> score_pairs;
};

using first_stage_bind_t = shared_ptr<FirstStageScorer> (*)(ClientContext &, const string &, bool);

struct FirstStageScorerAPI : ScalarFunctionInfo {
	// Return nullptr from the binder to use the configured HTTP backend.
	first_stage_bind_t bind = nullptr;
};

// Use catalog-owned function metadata: separately loaded extensions do not
// share exported symbols when DuckDB opens their libraries with RTLD_LOCAL.
inline void RegisterFirstStageScorer(ExtensionLoader &loader, first_stage_bind_t bind) {
	auto &functions = loader.GetFunction("jev_holds").functions.functions;
	if (!functions[0].function_info) {
		throw InvalidInputException("Load the matching jevdb extension before registering a scorer");
	}
	functions[0].function_info->Cast<FirstStageScorerAPI>().bind = bind;
}

shared_ptr<ScalarFunctionInfo> CreateFirstStageScorerAPI();
shared_ptr<FirstStageScorer> BindFirstStageScorer(ClientContext &context, const ScalarFunction &function,
                                                  bool pairwise);

} // namespace jevdb
} // namespace duckdb
