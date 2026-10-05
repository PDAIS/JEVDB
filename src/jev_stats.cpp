#include "jevdb.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include <cmath>

namespace duckdb {
namespace jevdb {

void QueryState::QueryEnd() {
	if (used) {
		last_stats = stats;
	}
	answers.clear();
	pools.clear();
	cascades.clear();
	llm_pools.clear();
	stats = Stats();
	used = false;
}

void QueryState::WriteProfilingInformation(std::ostream &ss) {
	if (!used) {
		return;
	}
	const auto &values = stats;
	ss << "JEV: requests=" << values.requests << ", pairs=" << values.pairs << ", records=" << values.records
	   << ", cache_hits=" << values.cache_hits << ", input_tokens=" << values.input_tokens
	   << ", retries=" << values.retries << ", errors=" << values.errors
	   << ", request_ms=" << values.request_us / 1000.0 << ", send_ms=" << values.send_us / 1000.0 << "\n";
	ss << "JEV candidates: left=" << values.candidate_left_rows << ", right=" << values.candidate_right_rows
	   << ", retained_left=" << values.candidate_left_retained << ", retained_right=" << values.candidate_right_retained
	   << ", posting_visits=" << values.index_posting_visits << ", candidates=" << values.index_candidate_pairs
	   << ", cross_product_pairs=" << values.cross_product_pairs << "\n";
	if (!std::isnan(values.cascade_last_low)) {
		ss << "JEV cascade: requests=" << values.cascade_requests << ", items=" << values.cascade_items
		   << ", calibration_items=" << values.calibration_items << ", band_items=" << values.band_items
		   << ", cache_hits=" << values.cascade_cache_hits << ", errors=" << values.cascade_errors
		   << ", input_tokens=" << values.cascade_input_tokens << ", request_ms=" << values.cascade_request_us / 1000.0
		   << ", send_ms=" << values.cascade_send_us / 1000.0 << ", last_low=" << values.cascade_last_low
		   << ", last_high=" << values.cascade_last_high << "\n";
	}
}

struct StatsScan : GlobalTableFunctionState {
	Stats values;
	bool finished = false;
};

static unique_ptr<FunctionData> BindStats(ClientContext &, TableFunctionBindInput &, vector<LogicalType> &types,
                                          vector<string> &names) {
	names = {"requests",
	         "pairs",
	         "records",
	         "cache_hits",
	         "input_tokens",
	         "errors",
	         "clients_created",
	         "client_reuses",
	         "request_ms",
	         "send_ms",
	         "cascade_requests",
	         "cascade_items",
	         "cascade_cache_hits",
	         "calibration_items",
	         "band_items",
	         "cascade_errors",
	         "cascade_input_tokens",
	         "cascade_clients_created",
	         "cascade_client_reuses",
	         "cascade_request_ms",
	         "cascade_send_ms",
	         "cascade_last_low",
	         "cascade_last_high",
	         "candidate_left_rows",
	         "candidate_right_rows",
	         "candidate_left_retained",
	         "candidate_right_retained",
	         "index_posting_visits",
	         "index_candidate_pairs",
	         "cross_product_pairs",
	         "retries",
	         "cascade_retries"};
	types.assign(8, LogicalType::UBIGINT);
	types.push_back(LogicalType::DOUBLE);
	types.push_back(LogicalType::DOUBLE);
	for (idx_t i = 0; i < 9; i++)
		types.push_back(LogicalType::UBIGINT);
	for (idx_t i = 0; i < 4; i++)
		types.push_back(LogicalType::DOUBLE);
	for (idx_t i = 0; i < 9; i++)
		types.push_back(LogicalType::UBIGINT);
	return nullptr;
}

static unique_ptr<GlobalTableFunctionState> InitStats(ClientContext &context, TableFunctionInitInput &) {
	auto scan = make_uniq<StatsScan>();
	auto state = context.registered_state->GetOrCreate<QueryState>("jevdb");
	std::lock_guard<std::mutex> guard(state->mutex);
	scan->values = state->last_stats;
	return std::move(scan);
}

static void ScanStats(ClientContext &, TableFunctionInput &input, DataChunk &output) {
	auto &scan = input.global_state->Cast<StatsScan>();
	if (scan.finished) {
		return;
	}
	const auto &stats = scan.values;
	output.SetCardinality(1);
	vector<idx_t> counts = {stats.requests,     stats.pairs,  stats.records,         stats.cache_hits,
	                        stats.input_tokens, stats.errors, stats.clients_created, stats.client_reuses};
	for (idx_t i = 0; i < counts.size(); i++) {
		output.SetValue(i, 0, Value::UBIGINT(counts[i]));
	}
	output.SetValue(8, 0, Value::DOUBLE(stats.request_us / 1000.0));
	output.SetValue(9, 0, Value::DOUBLE(stats.send_us / 1000.0));
	vector<idx_t> cascade_counts = {
	    stats.cascade_requests,     stats.cascade_items,  stats.cascade_cache_hits,   stats.calibration_items,
	    stats.band_items,           stats.cascade_errors, stats.cascade_input_tokens, stats.cascade_clients_created,
	    stats.cascade_client_reuses};
	for (idx_t i = 0; i < cascade_counts.size(); i++) {
		output.SetValue(10 + i, 0, Value::UBIGINT(cascade_counts[i]));
	}
	output.SetValue(19, 0, Value::DOUBLE(stats.cascade_request_us / 1000.0));
	output.SetValue(20, 0, Value::DOUBLE(stats.cascade_send_us / 1000.0));
	output.SetValue(
	    21, 0, std::isnan(stats.cascade_last_low) ? Value(LogicalType::DOUBLE) : Value::DOUBLE(stats.cascade_last_low));
	output.SetValue(22, 0,
	                std::isnan(stats.cascade_last_high) ? Value(LogicalType::DOUBLE)
	                                                    : Value::DOUBLE(stats.cascade_last_high));
	vector<idx_t> candidate_counts = {stats.candidate_left_rows,     stats.candidate_right_rows,
	                                  stats.candidate_left_retained, stats.candidate_right_retained,
	                                  stats.index_posting_visits,    stats.index_candidate_pairs,
	                                  stats.cross_product_pairs};
	for (idx_t i = 0; i < candidate_counts.size(); i++)
		output.SetValue(23 + i, 0, Value::UBIGINT(candidate_counts[i]));
	output.SetValue(30, 0, Value::UBIGINT(stats.retries));
	output.SetValue(31, 0, Value::UBIGINT(stats.cascade_retries));
	scan.finished = true;
}

void RegisterStats(ExtensionLoader &loader) {
	TableFunction function("jevdb_stats", {}, ScanStats);
	function.bind = BindStats;
	function.init_global = InitStats;
	loader.RegisterFunction(function);
}

} // namespace jevdb
} // namespace duckdb
