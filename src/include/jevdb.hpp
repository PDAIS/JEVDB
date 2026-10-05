#pragma once

#include "duckdb.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/client_context_state.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/operator/logical_extension_operator.hpp"
#include "duckdb/planner/operator/logical_any_join.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "jevdb_scorer.hpp"
#include <atomic>
#include <map>
#include <mutex>
#include <utility>
#include <tuple>
#include <limits>
#include <set>

namespace duckdb {
class ExtensionLoader;
class Optimizer;
namespace jevdb {

using Pair = std::pair<string, string>;
using Record = vector<Pair>;
enum class DecisionMode { HOLDS, PROBABILITY, CHOICE, SCORE };
enum class QuestionKind { NOUL, CHOICE, SCORE };

struct RetryConfig {
	idx_t max_retries = 7;
	double max_delay = 60;
	double max_retry_after = 300;
	bool operator==(const RetryConfig &other) const {
		return max_retries == other.max_retries && max_delay == other.max_delay &&
		       max_retry_after == other.max_retry_after;
	}
};

struct CascadeConfig {
	RetryConfig retry;
	string endpoint = "https://api.openai.com/v1/";
	string model;
	string api_key;
	string ca_cert_file;
	double low = 0.5;
	double high = 0.9;
	idx_t calibrate = 0;
	double recall_loss = 0.05;
	double precision_loss = 0.05;
	idx_t batch_size = 16;
	idx_t threads = 8;
	bool fill_batches = true;
	double timeout = 120;
	bool Enabled() const {
		return !model.empty();
	}
	bool operator==(const CascadeConfig &other) const {
		return retry == other.retry &&
		       std::tie(endpoint, model, api_key, ca_cert_file, low, high, calibrate, recall_loss, precision_loss,
		                batch_size, threads, fill_batches,
		                timeout) == std::tie(other.endpoint, other.model, other.api_key, other.ca_cert_file, other.low,
		                                     other.high, other.calibrate, other.recall_loss, other.precision_loss,
		                                     other.batch_size, other.threads, other.fill_batches, other.timeout);
	}
};

struct Config {
	string backend;
	string on_error;
	RetryConfig retry;
	string endpoint;
	string model;
	string layout;
	string anchor;
	idx_t k;
	double threshold;
	double budget_tokens;
	double auto_min_group;
	idx_t threads = 8;
	bool prefetch = true;
	string api_key;
	string ca_cert_file;
	string dump_path;
	string raw_dump_path;
	double timeout = 120;
	CascadeConfig cascade;
};

using CacheIdentity = std::tuple<string, string, string, QuestionKind, string, string, string, string, string>;
using ConnectionIdentity = std::tuple<string, string, string, idx_t, double, idx_t, double, double>;
using JevScheduleIdentity = std::tuple<string, string, idx_t, double, double, double, idx_t, bool>;
using CascadeIdentity = std::tuple<CacheIdentity, bool, vector<string>, vector<string>, JevScheduleIdentity, string,
                                   string, double, double, idx_t, double, double, idx_t, idx_t, bool, double>;
struct AnswerCache {
	std::map<Pair, double> pair_scores;
	std::map<Record, double> record_scores;
	std::map<Record, string> record_choices;
	std::set<Pair> failed_pairs;
	std::set<Record> failed_records;
};
struct CascadeCache {
	std::map<Pair, bool> pair_answers;
	std::map<Record, bool> record_answers;
	bool calibrated = false;
	double low = 0.5;
	double high = 0.9;
};
struct Stats {
	idx_t candidate_left_rows = 0;
	idx_t candidate_right_rows = 0;
	idx_t candidate_left_retained = 0;
	idx_t candidate_right_retained = 0;
	idx_t index_posting_visits = 0;
	idx_t index_candidate_pairs = 0;
	idx_t cross_product_pairs = 0;
	idx_t retries = 0;
	idx_t cascade_retries = 0;
	idx_t requests = 0;
	idx_t pairs = 0;
	idx_t records = 0;
	idx_t cache_hits = 0;
	idx_t input_tokens = 0;
	idx_t errors = 0;
	idx_t clients_created = 0;
	idx_t client_reuses = 0;
	int64_t request_us = 0;
	int64_t send_us = 0;
	idx_t cascade_requests = 0;
	idx_t cascade_items = 0;
	idx_t cascade_cache_hits = 0;
	idx_t calibration_items = 0;
	idx_t band_items = 0;
	idx_t cascade_errors = 0;
	idx_t cascade_input_tokens = 0;
	idx_t cascade_clients_created = 0;
	idx_t cascade_client_reuses = 0;
	int64_t cascade_request_us = 0;
	int64_t cascade_send_us = 0;
	double cascade_last_low = std::numeric_limits<double>::quiet_NaN();
	double cascade_last_high = std::numeric_limits<double>::quiet_NaN();
};
class ClientPool;
class LlmClientPool;

// Per-query answers are shared by expression copies. mutex protects cache and
// counters; HTTP runs outside it. execution_mutex serializes each collect/send
// operation so concurrent consumers cannot submit the same uncached input.
struct QueryState : ClientContextState {
	std::mutex mutex;
	std::mutex execution_mutex;
	std::map<CacheIdentity, AnswerCache> answers;
	std::map<ConnectionIdentity, shared_ptr<ClientPool>> pools;
	std::map<CascadeIdentity, CascadeCache> cascades;
	std::map<ConnectionIdentity, shared_ptr<LlmClientPool>> llm_pools;
	Stats stats;
	Stats last_stats;
	bool used = false;
	// The owning connection's interrupt flag; waits between request attempts stop when it is set.
	const std::atomic<bool> *interrupted = nullptr;
	void QueryEnd() override;
	void WriteProfilingInformation(std::ostream &ss) override;
};

struct HoldsData : FunctionData {
	Config config;
	string condition;
	shared_ptr<QueryState> state;
	DecisionMode mode = DecisionMode::HOLDS;
	QuestionKind kind = QuestionKind::NOUL;
	bool pairwise = true;
	bool stable_inputs = false;
	shared_ptr<FirstStageScorer> first_stage;
	vector<string> left_fields;
	vector<string> right_fields;
	vector<LogicalType> argument_types;
	string criteria_json;
	vector<string> choices;
	idx_t level_count = 0;
	CacheIdentity Identity() const {
		return std::make_tuple(condition, config.endpoint, config.model, kind, criteria_json,
		                       first_stage ? first_stage->configuration : "", first_stage ? first_stage->identity : "",
		                       config.backend, config.on_error);
	}
	bool UsesCascade() const {
		return mode == DecisionMode::HOLDS && config.cascade.Enabled();
	}
	CascadeIdentity CascadeKey() const {
		const auto &cascade = config.cascade;
		auto jev_schedule =
		    std::make_tuple(config.layout, config.anchor, config.k, config.threshold, config.budget_tokens,
		                    config.auto_min_group, config.threads, config.prefetch);
		return std::make_tuple(Identity(), pairwise, left_fields, right_fields, jev_schedule, cascade.endpoint,
		                       cascade.model, cascade.low, cascade.high, cascade.calibrate, cascade.recall_loss,
		                       cascade.precision_loss, cascade.batch_size, cascade.threads, cascade.fill_batches,
		                       cascade.timeout);
	}
	unique_ptr<FunctionData> Copy() const override;
	bool Equals(const FunctionData &other) const override;
};

inline bool UsesCascade(const HoldsData &data) {
	return data.UsesCascade();
}

// CONSISTENT lets relational rules move the predicate. Foldability is separate:
// planning a query must never send an external decision request.
class HoldsExpression : public BoundFunctionExpression {
public:
	HoldsExpression(ScalarFunction function, vector<unique_ptr<Expression>> children, unique_ptr<FunctionData> data);
	bool IsFoldable() const override {
		return false;
	}
	unique_ptr<Expression> Copy() const override;
};

Value Setting(ClientContext &context, const string &name);
ScalarFunction HoldsFunction();
bool IsHolds(const Expression &expression);
bool ContainsHolds(const Expression &expression);
void Judge(HoldsData &data, const vector<Pair> &pairs);
void JudgeLlm(HoldsData &data, const vector<Pair> *pairs, const vector<Record> *records);
bool Matches(HoldsData &data, const Pair &pair);
void FinishCascade(HoldsData &data, const vector<Pair> &pairs);
void FinishRecordCascade(HoldsData &data, const vector<Record> &records);
bool CascadeMatches(HoldsData &data, const Pair &pair);
bool CascadeRecordMatches(HoldsData &data, const Record &record);
bool RecordMatches(HoldsData &data, const Record &record);
double Probability(HoldsData &data, const Pair &pair);
void JudgeRecords(HoldsData &data, const vector<Record> &records);
double RecordProbability(HoldsData &data, const Record &record);
string RecordChoice(HoldsData &data, const Record &record);
bool InputRecord(const Value &value, const vector<string> &names, Record &record);
bool PairText(const Value &value, const vector<string> &names, string &text);
bool CanUseKeyCandidates(LogicalAnyJoin &join);
void RegisterStats(ExtensionLoader &loader);
void PrepareSBF(Optimizer &optimizer, unique_ptr<LogicalOperator> &plan);
void OptimizeSBF(Optimizer &optimizer, unique_ptr<LogicalOperator> &plan);
void ReduceSemanticInputs(Optimizer &optimizer, unique_ptr<LogicalOperator> &plan);

class LogicalSemanticJoin : public LogicalExtensionOperator {
public:
	explicit LogicalSemanticJoin(LogicalAnyJoin &join);
	JoinType join_type;
	bool key_candidates;
	vector<idx_t> left_projection_map;
	vector<idx_t> right_projection_map;
	vector<ColumnBinding> GetColumnBindings() override;
	string GetName() const override {
		return "JEV_SEMANTIC_JOIN";
	}
	string GetExtensionName() const override {
		return "jevdb";
	}
	void ResolveColumnBindings(ColumnBindingResolver &resolver, vector<ColumnBinding> &bindings) override;
	PhysicalOperator &CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) override;

protected:
	void ResolveTypes() override;
};

class LogicalSemanticFilter : public LogicalExtensionOperator {
public:
	explicit LogicalSemanticFilter(LogicalFilter &filter);
	vector<idx_t> projection_map;
	vector<ColumnBinding> GetColumnBindings() override;
	string GetName() const override {
		return "JEV_SEMANTIC_FILTER";
	}
	PhysicalOperator &CreatePlan(ClientContext &context, PhysicalPlanGenerator &planner) override;

protected:
	void ResolveTypes() override;
};

} // namespace jevdb
} // namespace duckdb
