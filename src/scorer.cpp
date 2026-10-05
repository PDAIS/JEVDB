#include "jevdb_scorer.hpp"

namespace duckdb {
namespace jevdb {

shared_ptr<ScalarFunctionInfo> CreateFirstStageScorerAPI() {
	return make_shared_ptr<FirstStageScorerAPI>();
}

shared_ptr<FirstStageScorer> BindFirstStageScorer(ClientContext &context, const ScalarFunction &function,
                                                  bool pairwise) {
	auto &api = function.function_info->Cast<FirstStageScorerAPI>();
	return api.bind ? api.bind(context, function.name, pairwise) : nullptr;
}

} // namespace jevdb
} // namespace duckdb
