#include "jevdb_scorer.hpp"
#include "duckdb/main/config.hpp"

namespace duckdb {
namespace jevdb {

// A deliberately simple example: shared prefix length / longer string length.
static shared_ptr<FirstStageScorer> BindExample(ClientContext &context, const string &function_name, bool pairwise) {
	Value enabled;
	context.TryGetCurrentSetting("jevdb_example_scorer", enabled);
	if (!BooleanValue::Get(enabled) || !pairwise ||
	    (function_name != "jev_holds" && function_name != "jev_probability"))
		return nullptr;
	auto scorer = make_shared_ptr<FirstStageScorer>();
	scorer->configuration = "shared-prefix";
	scorer->identity = "example-1";
	scorer->score_pairs = [](const string &, const vector<std::pair<string, string>> &pairs) {
		vector<double> scores;
		for (auto &pair : pairs) {
			const auto length = MaxValue(pair.first.size(), pair.second.size());
			idx_t prefix = 0;
			while (prefix < MinValue(pair.first.size(), pair.second.size()) &&
			       pair.first[prefix] == pair.second[prefix])
				prefix++;
			scores.push_back(length ? double(prefix) / length : 1);
		}
		return scores;
	};
	return scorer;
}

static void LoadExample(ExtensionLoader &loader) {
	RegisterFirstStageScorer(loader, BindExample);
	DBConfig::GetConfig(loader.GetDatabaseInstance())
	    .AddExtensionOption("jevdb_example_scorer", "Use the shared-prefix example scorer", LogicalType::BOOLEAN,
	                        Value(true));
}

} // namespace jevdb
} // namespace duckdb

extern "C" {
DUCKDB_CPP_EXTENSION_ENTRY(jevdb_example_scorer, loader) {
	duckdb::jevdb::LoadExample(loader);
}
}
