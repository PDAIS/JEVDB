#include "jevdb.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/execution/operator/join/physical_cross_product.hpp"
#include "duckdb/execution/operator/join/physical_join.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_cross_product.hpp"
#include <set>

namespace duckdb {
namespace jevdb {

static vector<LogicalType> InputTypes(const BoundFunctionExpression &holds) {
	vector<LogicalType> types {holds.children[1]->return_type};
	if (holds.bind_info->Cast<HoldsData>().pairwise) {
		types.push_back(holds.children[2]->return_type);
	}
	return types;
}

static bool ReadDecision(HoldsData &data, DataChunk &values, idx_t row, Pair &pair, Record &record) {
	if (data.pairwise) {
		return PairText(values.GetValue(0, row), data.left_fields, pair.first) &&
		       PairText(values.GetValue(1, row), data.right_fields, pair.second);
	}
	return InputRecord(values.GetValue(0, row), data.left_fields, record);
}

static bool DecisionMatches(HoldsData &data, const Pair &pair, const Record &record) {
	return data.pairwise ? Matches(data, pair) : RecordMatches(data, record);
}

static void FinishDecisions(HoldsData &data, const vector<Pair> &pairs, const vector<Record> &records) {
	if (data.pairwise) {
		FinishCascade(data, pairs);
	} else {
		FinishRecordCascade(data, records);
	}
}

// Buffer only eligible rows. Relational predicates and model input expressions
// are evaluated once, before calibration; replay uses these saved values.
struct DeferredInputs {
	DeferredInputs(ClientContext &context, const vector<LogicalType> &types) : rows(context, types) {
	}
	ColumnDataCollection rows;
	vector<Pair> pairs;
	vector<Record> records;
	std::set<Pair> seen_pairs;
	std::set<Record> seen_records;
	void Add(HoldsData &data, const Pair &pair, const Record &record) {
		if (data.pairwise) {
			if (seen_pairs.insert(pair).second)
				pairs.push_back(pair);
		} else if (seen_records.insert(record).second) {
			records.push_back(record);
		}
	}
};

struct ReplayState : GlobalSourceState {
	ColumnDataScanState scan;
	DataChunk rows;
	DataChunk values;
	vector<bool> matched;
	idx_t left_position = 0;
	bool initialized = false;
	ColumnDataScanState indexed_scan;
	DataChunk indexed_input;
	unique_ptr<OperatorState> indexed_state;
	idx_t indexed_position = 0;
	bool indexed_more = false;
	bool indexed_done = false;
};

static void ReadSavedValues(DataChunk &rows, DataChunk &values, idx_t offset) {
	values.Reset();
	values.SetCardinality(rows.size());
	for (idx_t column = 0; column < values.ColumnCount(); column++) {
		values.data[column].Reference(rows.data[offset + column]);
	}
}

static void SplitCondition(Expression &expression, optional_ptr<BoundFunctionExpression> &holds,
                           vector<unique_ptr<Expression>> &ordinary) {
	if (IsHolds(expression)) {
		if (holds) {
			throw NotImplementedException("Semantic operators support one jev_holds predicate each");
		}
		holds = expression.Cast<BoundFunctionExpression>();
	} else if (expression.type == ExpressionType::CONJUNCTION_AND) {
		for (auto &child : expression.Cast<BoundConjunctionExpression>().children) {
			SplitCondition(*child, holds, ordinary);
		}
	} else {
		if (ContainsHolds(expression)) {
			throw NotImplementedException("Semantic join predicates must be AND conditions");
		}
		ordinary.push_back(expression.Copy());
	}
}

// Only direct, positive AND keys with columns in the two current inputs are
// indexed. Attached-key LEFT JOINs are therefore eligible without base-table
// lineage assumptions.
static void Conjuncts(const Expression &expression, vector<const Expression *> &result) {
	if (expression.type == ExpressionType::CONJUNCTION_AND) {
		for (auto &child : expression.Cast<BoundConjunctionExpression>().children)
			Conjuncts(*child, result);
	} else {
		result.push_back(&expression);
	}
}

static bool IsKey(const Expression &expression) {
	return expression.expression_class == ExpressionClass::BOUND_FUNCTION &&
	       expression.Cast<BoundFunctionExpression>().function.name == "jev_keys_overlap";
}

bool CanUseKeyCandidates(LogicalAnyJoin &join) {
	if (join.join_type != JoinType::INNER)
		return false;
	vector<const Expression *> conjuncts;
	Conjuncts(*join.condition, conjuncts);
	idx_t holds_count = 0;
	bool key_found = false;
	auto left = join.children[0]->GetColumnBindings();
	auto right = join.children[1]->GetColumnBindings();
	auto belongs = [](const vector<ColumnBinding> &bindings, ColumnBinding binding) {
		return std::find(bindings.begin(), bindings.end(), binding) != bindings.end();
	};
	for (auto expression : conjuncts) {
		if (IsHolds(*expression)) {
			auto &data = expression->Cast<BoundFunctionExpression>().bind_info->Cast<HoldsData>();
			// Global calibration samples the materialized distinct pool. Direct
			// indexed enumeration changes its order, hence its sampled answers.
			if (data.UsesCascade() && data.config.cascade.calibrate > 0)
				return false;
			holds_count++;
		} else if (ContainsHolds(*expression)) {
			return false;
		}
		if (!IsKey(*expression))
			continue;
		auto &args = expression->Cast<BoundFunctionExpression>().children;
		if (args.size() != 2 || args[0]->expression_class != ExpressionClass::BOUND_COLUMN_REF ||
		    args[1]->expression_class != ExpressionClass::BOUND_COLUMN_REF ||
		    args[0]->return_type != LogicalType::LIST(LogicalType::VARCHAR) ||
		    args[1]->return_type != LogicalType::LIST(LogicalType::VARCHAR))
			continue;
		auto &a = args[0]->Cast<BoundColumnRefExpression>();
		auto &b = args[1]->Cast<BoundColumnRefExpression>();
		if (!a.depth && !b.depth &&
		    ((belongs(left, a.binding) && belongs(right, b.binding)) ||
		     (belongs(right, a.binding) && belongs(left, b.binding))))
			key_found = true;
	}
	return holds_count == 1 && key_found;
}

struct KeyColumns {
	idx_t left;
	idx_t right;
};

static vector<KeyColumns> BoundKeys(const Expression &condition, idx_t left_count) {
	vector<const Expression *> conjuncts;
	Conjuncts(condition, conjuncts);
	vector<KeyColumns> result;
	for (auto expression : conjuncts) {
		if (!IsKey(*expression))
			continue;
		auto &args = expression->Cast<BoundFunctionExpression>().children;
		if (args.size() != 2 || args[0]->expression_class != ExpressionClass::BOUND_REF ||
		    args[1]->expression_class != ExpressionClass::BOUND_REF)
			continue;
		auto a = args[0]->Cast<BoundReferenceExpression>().index;
		auto b = args[1]->Cast<BoundReferenceExpression>().index;
		if (a < left_count && b >= left_count)
			result.push_back({a, b - left_count});
		else if (b < left_count && a >= left_count)
			result.push_back({b, a - left_count});
	}
	return result;
}

static std::set<string> KeyTokens(const Value &key) {
	std::set<string> result;
	for (auto &token : ListValue::GetChildren(key)) {
		if (!token.IsNull())
			result.insert(StringValue::Get(token));
	}
	return result;
}

struct RowAddress {
	idx_t chunk;
	idx_t row;
};

struct KeyIndex {
	std::map<string, vector<idx_t>> postings;
	vector<idx_t> null_rows;
	vector<RowAddress> addresses;
	vector<idx_t> active_rows;
	void Build(ColumnDataCollection &rows, idx_t key_column, const vector<bool> &active) {
		DataChunk chunk;
		rows.InitializeScanChunk(chunk);
		for (idx_t c = 0; c < rows.ChunkCount(); c++) {
			rows.FetchChunk(c, chunk);
			for (idx_t row = 0; row < chunk.size(); row++) {
				auto id = addresses.size();
				addresses.push_back({c, row});
				if (!active[id])
					continue;
				active_rows.push_back(id);
				auto key = chunk.GetValue(key_column, row);
				if (key.IsNull()) {
					null_rows.push_back(id);
				} else {
					for (auto &token : KeyTokens(key))
						postings[token].push_back(id);
				}
			}
		}
	}
	vector<idx_t> Lookup(const Value &key, idx_t &visits) const {
		std::set<idx_t> matches;
		if (key.IsNull()) {
			for (auto row : active_rows) {
				visits++;
				matches.insert(row);
			}
		} else {
			for (auto row : null_rows) {
				visits++;
				matches.insert(row);
			}
			for (auto &token : KeyTokens(key)) {
				auto entry = postings.find(token);
				if (entry != postings.end()) {
					for (auto row : entry->second) {
						visits++;
						matches.insert(row);
					}
				}
			}
		}
		return vector<idx_t>(matches.begin(), matches.end());
	}
};

// Each key proves partner existence independently against the original opposite
// input. This is a conservative preselection, not joint multi-key matching.
struct PartnerKeys {
	idx_t rows = 0;
	bool has_null = false;
	std::set<string> tokens;
};

static vector<PartnerKeys> SummarizeKeys(ColumnDataCollection &rows, const vector<KeyColumns> &keys, bool left) {
	vector<PartnerKeys> result(keys.size());
	ColumnDataScanState scan;
	DataChunk chunk;
	rows.InitializeScan(scan);
	rows.InitializeScanChunk(chunk);
	while (rows.Scan(scan, chunk)) {
		for (idx_t k = 0; k < keys.size(); k++) {
			auto &summary = result[k];
			summary.rows += chunk.size();
			for (idx_t row = 0; row < chunk.size(); row++) {
				auto key = chunk.GetValue(left ? keys[k].left : keys[k].right, row);
				if (key.IsNull()) {
					summary.has_null = true;
				} else {
					for (auto &token : KeyTokens(key))
						summary.tokens.insert(token);
				}
			}
		}
	}
	return result;
}

static bool HasPartner(const Value &key, const PartnerKeys &opposite) {
	if (!opposite.rows)
		return false;
	if (key.IsNull() || opposite.has_null)
		return true;
	for (auto &token : KeyTokens(key)) {
		if (opposite.tokens.count(token))
			return true;
	}
	return false;
}

static vector<bool> SelectPartnerRows(ColumnDataCollection &rows, const vector<KeyColumns> &keys, bool left,
                                      const vector<PartnerKeys> &opposite, vector<idx_t> &non_null) {
	vector<bool> result;
	ColumnDataScanState scan;
	DataChunk chunk;
	rows.InitializeScan(scan);
	rows.InitializeScanChunk(chunk);
	while (rows.Scan(scan, chunk)) {
		for (idx_t row = 0; row < chunk.size(); row++) {
			bool keep = true;
			for (idx_t k = 0; k < keys.size(); k++) {
				if (!HasPartner(chunk.GetValue(left ? keys[k].left : keys[k].right, row), opposite[k])) {
					keep = false;
					break;
				}
			}
			if (keep) {
				for (idx_t k = 0; k < keys.size(); k++) {
					if (!chunk.GetValue(left ? keys[k].left : keys[k].right, row).IsNull())
						non_null[k]++;
				}
			}
			result.push_back(keep);
		}
	}
	return result;
}

struct JoinSink : GlobalSinkState {
	JoinSink(ClientContext &context, const vector<LogicalType> &right_types, const vector<LogicalType> &left_types,
	         const vector<LogicalType> &deferred_types, bool global, bool indexed)
	    : right(context, right_types) {
		if (global || indexed)
			left = make_uniq<ColumnDataCollection>(context, left_types);
		if (global)
			deferred = make_uniq<DeferredInputs>(context, deferred_types);
	}
	ColumnDataCollection right;
	KeyIndex index;
	vector<bool> active_left;
	bool index_ready = false;
	idx_t key_choice = 0;
	unique_ptr<ColumnDataCollection> left;
	unique_ptr<DeferredInputs> deferred;
	std::mutex probe_mutex;
};

struct JoinState : CachingOperatorState {
	JoinState(ExecutionContext &context, ColumnDataCollection &right, const BoundFunctionExpression &holds,
	          optional_ptr<const Expression> ordinary, const vector<LogicalType> &types)
	    : cross_product(right), arguments(context.client), ordinary_executor(context.client) {
		arguments.AddExpression(*holds.children[1]);
		if (holds.bind_info->Cast<HoldsData>().pairwise) {
			arguments.AddExpression(*holds.children[2]);
		}
		values.Initialize(Allocator::DefaultAllocator(), InputTypes(holds));
		product.Initialize(Allocator::DefaultAllocator(), types);
		if (ordinary) {
			ordinary_executor.AddExpression(*ordinary);
		}
	}
	bool NextIndexed(DataChunk &input, ColumnDataCollection &right, const KeyIndex &index, idx_t key_column) {
		idx_t count = 0;
		product_left.clear();
		while (index_left < input.size() && count < STANDARD_VECTOR_SIZE) {
			if (!lookup_ready) {
				index_matches = index.Lookup(input.GetValue(key_column, index_left), posting_visits);
				candidate_pairs += index_matches.size();
				lookup_ready = true;
			}
			if (index_position == index_matches.size()) {
				index_position = 0;
				lookup_ready = false;
				index_left++;
				continue;
			}
			auto address = index.addresses[index_matches[index_position++]];
			if (fetched_chunk != address.chunk) {
				if (right_chunk.ColumnCount() == 0)
					right.InitializeScanChunk(right_chunk);
				right.FetchChunk(address.chunk, right_chunk);
				fetched_chunk = address.chunk;
			}
			for (idx_t column = 0; column < input.ColumnCount(); column++)
				product.SetValue(column, count, input.GetValue(column, index_left));
			for (idx_t column = 0; column < right_chunk.ColumnCount(); column++)
				product.SetValue(input.ColumnCount() + column, count, right_chunk.GetValue(column, address.row));
			product_left.push_back(index_left);
			count++;
		}
		product.SetCardinality(count);
		if (!count) {
			index_left = 0;
			return false;
		}
		return true;
	}
	DataChunk right_chunk;
	vector<idx_t> product_left;
	vector<idx_t> index_matches;
	idx_t index_left = 0;
	idx_t index_position = 0;
	idx_t fetched_chunk = DConstants::INVALID_INDEX;
	idx_t posting_visits = 0;
	idx_t candidate_pairs = 0;
	bool lookup_ready = false;
	CrossProductExecutor cross_product;
	ExpressionExecutor arguments;
	ExpressionExecutor ordinary_executor;
	DataChunk product;
	DataChunk values;
	vector<vector<Value>> rows;
	idx_t position = 0;
	bool gathered = false;
};

class PhysicalSemanticJoin : public PhysicalJoin {
public:
	PhysicalSemanticJoin(PhysicalPlan &plan, LogicalSemanticJoin &logical, PhysicalOperator &left,
	                     PhysicalOperator &right, unique_ptr<Expression> condition)
	    : PhysicalJoin(plan, logical, PhysicalOperatorType::EXTENSION, logical.join_type,
	                   logical.estimated_cardinality),
	      condition(std::move(condition)) {
		if (logical.key_candidates)
			keys = BoundKeys(*this->condition, left.types.size());
		children.push_back(left);
		children.push_back(right);
		vector<unique_ptr<Expression>> conjuncts;
		SplitCondition(*this->condition, holds, conjuncts);
		if (!conjuncts.empty()) {
			auto conjunction = make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND);
			conjunction->children = std::move(conjuncts);
			ordinary = std::move(conjunction);
		}
		if (join_type != JoinType::INNER && join_type != JoinType::SEMI && join_type != JoinType::ANTI) {
			throw NotImplementedException("Semantic joins support INNER, SEMI and ANTI");
		}
		auto append_projection = [&](const vector<idx_t> &projection, idx_t count, idx_t offset) {
			if (projection.empty()) {
				for (idx_t i = 0; i < count; i++) {
					output_columns.push_back(offset + i);
				}
			} else {
				for (auto column : projection) {
					output_columns.push_back(offset + column);
				}
			}
		};
		append_projection(logical.left_projection_map, left.types.size(), 0);
		if (join_type == JoinType::INNER) {
			append_projection(logical.right_projection_map, right.types.size(), left.types.size());
		}
		auto &data = holds->bind_info->Cast<HoldsData>();
		global_calibration = UsesCascade(data) && data.config.cascade.calibrate > 0;
		if (global_calibration) {
			if (join_type == JoinType::INNER)
				deferred_types = types;
			auto input_types = InputTypes(*holds);
			deferred_types.insert(deferred_types.end(), input_types.begin(), input_types.end());
			deferred_types.push_back(LogicalType::UBIGINT);
		}
	}

	string GetName() const override {
		return keys.empty() ? "JEV_SEMANTIC_JOIN" : "JEV_INDEXED_SEMANTIC_JOIN";
	}
	InsertionOrderPreservingMap<string> ParamsToString() const override {
		InsertionOrderPreservingMap<string> parameters;
		parameters["Condition"] = condition->ToString();
		return parameters;
	}
	bool IsSink() const override {
		return true;
	}
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override {
		return make_uniq<JoinSink>(context, children[1].get().types, children[0].get().types, deferred_types,
		                           global_calibration, !keys.empty());
	}
	SinkResultType Sink(ExecutionContext &, DataChunk &chunk, OperatorSinkInput &input) const override {
		input.global_state.Cast<JoinSink>().right.Append(chunk);
		return SinkResultType::NEED_MORE_INPUT;
	}
	void PrepareCandidates(JoinSink &sink) const {
		auto left_keys = SummarizeKeys(*sink.left, keys, true);
		auto right_keys = SummarizeKeys(sink.right, keys, false);
		vector<idx_t> left_non_null(keys.size(), 0), right_non_null(keys.size(), 0);
		sink.active_left = SelectPartnerRows(*sink.left, keys, true, right_keys, left_non_null);
		auto active_right = SelectPartnerRows(sink.right, keys, false, left_keys, right_non_null);
		// All-NULL arrays on either retained side make this key a wildcard
		// for every pair. Choose the first key without that property; if all
		// keys are wildcards, keep the original first-key rule.
		for (idx_t k = 0; k < keys.size(); k++) {
			if (left_non_null[k] && right_non_null[k]) {
				sink.key_choice = k;
				break;
			}
		}
		// Build postings only from retained original rows. NULL probes traverse
		// this same active range, including retained empty-key rows.
		sink.index.Build(sink.right, keys[sink.key_choice].right, active_right);
		sink.index_ready = true;
		auto &query = *holds->bind_info->Cast<HoldsData>().state;
		std::lock_guard<std::mutex> guard(query.mutex);
		query.used = true;
		query.stats.candidate_left_rows += sink.left->Count();
		query.stats.candidate_right_rows += sink.right.Count();
		query.stats.candidate_left_retained += std::count(sink.active_left.begin(), sink.active_left.end(), true);
		query.stats.candidate_right_retained += sink.index.active_rows.size();
	}
	unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context) const override {
		auto all_types = children[0].get().types;
		auto &right_types = children[1].get().types;
		all_types.insert(all_types.end(), right_types.begin(), right_types.end());
		return make_uniq<JoinState>(context, sink_state->Cast<JoinSink>().right, *holds, ordinary.get(), all_types);
	}
	bool IsSource() const override {
		return global_calibration || !keys.empty();
	}
	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &) const override {
		return make_uniq<ReplayState>();
	}
	SourceResultType GetData(ExecutionContext &context, DataChunk &output, OperatorSourceInput &input) const override {
		auto &state = input.global_state.Cast<ReplayState>();
		auto &sink = sink_state->Cast<JoinSink>();
		auto &data = holds->bind_info->Cast<HoldsData>();
		if (!keys.empty() && !state.indexed_done) {
			if (!sink.index_ready) {
				PrepareCandidates(sink);
				sink.left->InitializeScan(state.indexed_scan);
				sink.left->InitializeScanChunk(state.indexed_input);
				state.indexed_state = GetOperatorState(context);
			}
			SelectionVector selection(STANDARD_VECTOR_SIZE);
			while (true) {
				if (!state.indexed_more) {
					if (!sink.left->Scan(state.indexed_scan, state.indexed_input)) {
						state.indexed_done = true;
						break;
					}
					idx_t found = 0;
					for (idx_t row = 0; row < state.indexed_input.size(); row++) {
						if (sink.active_left[state.indexed_position + row])
							selection.set_index(found++, row);
					}
					state.indexed_position += state.indexed_input.size();
					state.indexed_input.Slice(selection, found);
					if (!found)
						continue;
				}
				auto result = Probe(state.indexed_input, output, state.indexed_state->Cast<JoinState>());
				state.indexed_more = result == OperatorResultType::HAVE_MORE_OUTPUT;
				if (output.size())
					return SourceResultType::HAVE_MORE_OUTPUT;
			}
		}
		if (!global_calibration)
			return SourceResultType::FINISHED;
		auto &saved = *sink.deferred;
		const idx_t value_offset = join_type == JoinType::INNER ? types.size() : 0;
		if (!state.initialized) {
			FinishDecisions(data, saved.pairs, saved.records);
			saved.rows.InitializeScan(state.scan);
			saved.rows.InitializeScanChunk(state.rows);
			state.values.Initialize(Allocator::DefaultAllocator(), InputTypes(*holds));
			if (join_type != JoinType::INNER) {
				state.matched.resize(sink.left->Count(), false);
				while (saved.rows.Scan(state.scan, state.rows)) {
					ReadSavedValues(state.rows, state.values, value_offset);
					for (idx_t row = 0; row < state.rows.size(); row++) {
						Pair pair;
						Record record;
						ReadDecision(data, state.values, row, pair, record);
						if (DecisionMatches(data, pair, record)) {
							auto left_row = state.rows.GetValue(state.rows.ColumnCount() - 1, row).GetValue<idx_t>();
							state.matched[left_row] = true;
						}
					}
				}
				sink.left->InitializeScan(state.scan);
				state.rows.Destroy();
				sink.left->InitializeScanChunk(state.rows);
			}
			state.initialized = true;
		}
		SelectionVector selected(STANDARD_VECTOR_SIZE);
		while ((join_type == JoinType::INNER ? saved.rows : *sink.left).Scan(state.scan, state.rows)) {
			idx_t found = 0;
			if (join_type == JoinType::INNER) {
				ReadSavedValues(state.rows, state.values, value_offset);
				for (idx_t row = 0; row < state.rows.size(); row++) {
					Pair pair;
					Record record;
					ReadDecision(data, state.values, row, pair, record);
					if (DecisionMatches(data, pair, record))
						selected.set_index(found++, row);
				}
			} else {
				for (idx_t row = 0; row < state.rows.size(); row++) {
					if (state.matched[state.left_position + row] == (join_type == JoinType::SEMI)) {
						selected.set_index(found++, row);
					}
				}
				state.left_position += state.rows.size();
			}
			if (!found)
				continue;
			output.SetCardinality(state.rows.size());
			for (idx_t column = 0; column < output.ColumnCount(); column++) {
				output.data[column].Reference(
				    state.rows.data[join_type == JoinType::INNER ? column : output_columns[column]]);
			}
			output.Slice(selected, found);
			return SourceResultType::HAVE_MORE_OUTPUT;
		}
		return SourceResultType::FINISHED;
	}

protected:
	OperatorResultType ExecuteInternal(ExecutionContext &, DataChunk &input, DataChunk &output, GlobalOperatorState &,
	                                   OperatorState &operator_state) const override {
		if (!keys.empty()) {
			auto &sink = sink_state->Cast<JoinSink>();
			std::lock_guard<std::mutex> guard(sink.probe_mutex);
			sink.left->Append(input);
			return OperatorResultType::NEED_MORE_INPUT;
		}
		return Probe(input, output, operator_state.Cast<JoinState>());
	}

	OperatorResultType Probe(DataChunk &input, DataChunk &output, JoinState &state) const {
		auto &data = holds->bind_info->Cast<HoldsData>();
		auto &sink = sink_state->Cast<JoinSink>();
		// UNION branches may have distinct probe pipelines even when each
		// pipeline has only one worker. Their shared append and row offsets
		// must be one operation.
		std::unique_lock<std::mutex> probe(sink.probe_mutex, std::defer_lock);
		if (global_calibration)
			probe.lock();
		idx_t left_base = 0;
		if (global_calibration && keys.empty()) {
			left_base = sink.left->Count();
			sink.left->Append(input);
		}
		if (!state.gathered) {
			vector<Pair> pairs;
			vector<Record> records;
			vector<vector<Value>> candidates;
			vector<idx_t> left_indices;
			SelectionVector selection(STANDARD_VECTOR_SIZE);
			idx_t cross_pairs = 0;
			if (sink.right.Count()) {
				while (true) {
					state.product.Reset();
					if (!keys.empty()) {
						if (!state.NextIndexed(input, sink.right, sink.index, keys[sink.key_choice].left))
							break;
					} else {
						if (state.cross_product.Execute(input, state.product) == OperatorResultType::NEED_MORE_INPUT)
							break;
						cross_pairs += state.product.size();
					}
					idx_t count = state.product.size();
					if (ordinary) {
						count = state.ordinary_executor.SelectExpression(state.product, selection);
					} else {
						for (idx_t i = 0; i < count; i++) {
							selection.set_index(i, i);
						}
					}
					// Evaluate ordinary and key predicates before collecting model inputs.
					// Selection indices also identify the original probe-side row.
					for (idx_t i = 0; i < count; i++) {
						left_indices.push_back(!keys.empty() ? state.product_left[selection.get_index(i)]
						                       : state.cross_product.ScanLHS() ? state.cross_product.PositionInChunk()
						                                                       : selection.get_index(i));
					}
					state.product.Slice(selection, count);
					state.values.Reset();
					state.arguments.Execute(state.product, state.values);
					DataChunk deferred_rows;
					idx_t deferred_count = 0;
					if (global_calibration)
						deferred_rows.Initialize(Allocator::DefaultAllocator(), deferred_types);
					vector<Pair> chunk_pairs;
					vector<Record> chunk_records;
					for (idx_t row = 0; row < count; row++) {
						Pair pair;
						Record record;
						if (!ReadDecision(data, state.values, row, pair, record)) {
							left_indices[left_indices.size() - count + row] = DConstants::INVALID_INDEX;
						} else {
							if (global_calibration) {
								auto &saved = *sink.deferred;
								saved.Add(data, pair, record);
								idx_t column = 0;
								if (join_type == JoinType::INNER) {
									for (auto projected : output_columns) {
										deferred_rows.SetValue(column++, deferred_count,
										                       state.product.GetValue(projected, row));
									}
								}
								for (idx_t argument = 0; argument < state.values.ColumnCount(); argument++) {
									deferred_rows.SetValue(column++, deferred_count,
									                       state.values.GetValue(argument, row));
								}
								deferred_rows.SetValue(
								    column, deferred_count++,
								    Value::UBIGINT(left_base + left_indices[left_indices.size() - count + row]));
							}
							if (!data.config.prefetch) {
								if (data.pairwise)
									chunk_pairs.push_back(pair);
								else
									chunk_records.push_back(record);
							}
						}
						pairs.push_back(std::move(pair));
						records.push_back(std::move(record));
						vector<Value> values;
						if (join_type == JoinType::INNER && !global_calibration) {
							// Retain projected output values only; model inputs already
							// live in pairs/records. SEMI/ANTI need no pair copies.
							for (auto column : output_columns) {
								values.push_back(state.product.GetValue(column, row));
							}
						}
						candidates.push_back(std::move(values));
					}
					if (global_calibration && deferred_count) {
						deferred_rows.SetCardinality(deferred_count);
						sink.deferred->rows.Append(deferred_rows);
					}
					if (!data.config.prefetch) {
						if (data.pairwise) {
							Judge(data, chunk_pairs);
						} else {
							JudgeRecords(data, chunk_records);
						}
					}
				}
			}
			{
				std::lock_guard<std::mutex> guard(data.state->mutex);
				data.state->used = true;
				auto &stats = data.state->stats;
				if (!keys.empty()) {
					stats.index_posting_visits += state.posting_visits;
					stats.index_candidate_pairs += state.candidate_pairs;
					state.posting_visits = state.candidate_pairs = 0;
				}
				stats.cross_product_pairs += cross_pairs;
			}
			vector<Pair> requested;
			vector<Record> requested_records;
			for (idx_t i = 0; i < pairs.size(); i++) {
				if (left_indices[i] != DConstants::INVALID_INDEX) {
					if (data.pairwise) {
						requested.push_back(pairs[i]);
					} else {
						requested_records.push_back(records[i]);
					}
				}
			}
			if (data.config.prefetch) {
				if (data.pairwise) {
					Judge(data, requested);
				} else {
					JudgeRecords(data, requested_records);
				}
			}
			if (global_calibration)
				return OperatorResultType::NEED_MORE_INPUT;
			FinishDecisions(data, requested, requested_records);
			vector<bool> matched(input.size(), false);
			for (idx_t i = 0; i < pairs.size(); i++) {
				if (left_indices[i] != DConstants::INVALID_INDEX && DecisionMatches(data, pairs[i], records[i])) {
					matched[left_indices[i]] = true;
					if (join_type == JoinType::INNER) {
						state.rows.push_back(std::move(candidates[i]));
					}
				}
			}
			if (join_type == JoinType::SEMI || join_type == JoinType::ANTI) {
				for (idx_t row = 0; row < input.size(); row++) {
					if (matched[row] == (join_type == JoinType::SEMI)) {
						vector<Value> values;
						for (auto column : output_columns) {
							values.push_back(input.GetValue(column, row));
						}
						state.rows.push_back(std::move(values));
					}
				}
			}
			state.gathered = true;
		}
		auto count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, state.rows.size() - state.position);
		output.SetCardinality(count);
		for (idx_t row = 0; row < count; row++) {
			for (idx_t column = 0; column < output_columns.size(); column++) {
				output.SetValue(column, row, state.rows[state.position + row][column]);
			}
		}
		state.position += count;
		if (state.position < state.rows.size()) {
			return OperatorResultType::HAVE_MORE_OUTPUT;
		}
		state.rows.clear();
		state.position = 0;
		state.gathered = false;
		return OperatorResultType::NEED_MORE_INPUT;
	}

private:
	unique_ptr<Expression> condition;
	unique_ptr<Expression> ordinary;
	optional_ptr<BoundFunctionExpression> holds;
	vector<KeyColumns> keys;
	vector<idx_t> output_columns;
	vector<LogicalType> deferred_types;
	bool global_calibration = false;
};

LogicalSemanticJoin::LogicalSemanticJoin(LogicalAnyJoin &join)
    : join_type(join.join_type), key_candidates(CanUseKeyCandidates(join)),
      left_projection_map(join.left_projection_map), right_projection_map(join.right_projection_map) {
	children = std::move(join.children);
	expressions.push_back(std::move(join.condition));
	estimated_cardinality = join.estimated_cardinality;
}

vector<ColumnBinding> LogicalSemanticJoin::GetColumnBindings() {
	auto result = MapBindings(children[0]->GetColumnBindings(), left_projection_map);
	if (join_type == JoinType::INNER) {
		auto right = MapBindings(children[1]->GetColumnBindings(), right_projection_map);
		result.insert(result.end(), right.begin(), right.end());
	}
	return result;
}

void LogicalSemanticJoin::ResolveTypes() {
	types = MapTypes(children[0]->types, left_projection_map);
	if (join_type == JoinType::INNER) {
		auto right = MapTypes(children[1]->types, right_projection_map);
		types.insert(types.end(), right.begin(), right.end());
	}
}

void LogicalSemanticJoin::ResolveColumnBindings(ColumnBindingResolver &resolver, vector<ColumnBinding> &bindings) {
	// Resolve the combined input through DuckDB's own cross-product binding
	// rule, which sets both bindings and types before visiting the condition.
	LogicalCrossProduct product(std::move(children[0]), std::move(children[1]));
	product.ResolveOperatorTypes();
	resolver.VisitOperator(product);
	resolver.VisitExpression(&expressions[0]);
	children = std::move(product.children);
	bindings = GetColumnBindings();
}

PhysicalOperator &LogicalSemanticJoin::CreatePlan(ClientContext &, PhysicalPlanGenerator &planner) {
	auto &left = planner.CreatePlan(*children[0]);
	auto &right = planner.CreatePlan(*children[1]);
	return planner.Make<PhysicalSemanticJoin>(*this, left, right, std::move(expressions[0]));
}

struct FilterState : OperatorState {
	FilterState(ClientContext &context, const BoundFunctionExpression &holds, optional_ptr<const Expression> ordinary,
	            const vector<LogicalType> &types)
	    : arguments(context), ordinary_executor(context) {
		arguments.AddExpression(*holds.children[1]);
		if (holds.bind_info->Cast<HoldsData>().pairwise) {
			arguments.AddExpression(*holds.children[2]);
		}
		if (ordinary) {
			ordinary_executor.AddExpression(*ordinary);
		}
		eligible.Initialize(Allocator::DefaultAllocator(), types);
		values.Initialize(Allocator::DefaultAllocator(), InputTypes(holds));
	}
	ExpressionExecutor arguments;
	ExpressionExecutor ordinary_executor;
	DataChunk eligible;
	DataChunk values;
};

struct FilterSink : GlobalSinkState {
	FilterSink(ClientContext &context, const vector<LogicalType> &types) : deferred(context, types) {
	}
	DeferredInputs deferred;
};

struct FilterLocalSink : LocalSinkState {
	FilterLocalSink(ClientContext &context, const BoundFunctionExpression &holds,
	                optional_ptr<const Expression> ordinary, const vector<LogicalType> &types)
	    : state(context, holds, ordinary, types) {
	}
	FilterState state;
};

class PhysicalSemanticFilter : public PhysicalOperator {
public:
	PhysicalSemanticFilter(PhysicalPlan &plan, LogicalSemanticFilter &logical, PhysicalOperator &child,
	                       unique_ptr<Expression> condition)
	    : PhysicalOperator(plan, PhysicalOperatorType::EXTENSION, logical.types, logical.estimated_cardinality),
	      condition(std::move(condition)), projection_map(logical.projection_map) {
		children.push_back(child);
		vector<unique_ptr<Expression>> conjuncts;
		SplitCondition(*this->condition, holds, conjuncts);
		if (!conjuncts.empty()) {
			auto conjunction = make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND);
			conjunction->children = std::move(conjuncts);
			ordinary = std::move(conjunction);
		}
		auto &data = holds->bind_info->Cast<HoldsData>();
		global_calibration = UsesCascade(data) && data.config.cascade.calibrate > 0;
		deferred_types = child.types;
		auto input_types = InputTypes(*holds);
		deferred_types.insert(deferred_types.end(), input_types.begin(), input_types.end());
	}
	string GetName() const override {
		return "JEV_SEMANTIC_FILTER";
	}
	InsertionOrderPreservingMap<string> ParamsToString() const override {
		InsertionOrderPreservingMap<string> parameters;
		parameters["Condition"] = condition->ToString();
		return parameters;
	}
	unique_ptr<OperatorState> GetOperatorState(ExecutionContext &context) const override {
		return make_uniq<FilterState>(context.client, *holds, ordinary.get(), children[0].get().types);
	}
	bool IsSink() const override {
		return global_calibration;
	}
	bool IsSource() const override {
		return global_calibration;
	}
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override {
		return make_uniq<FilterSink>(context, deferred_types);
	}
	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override {
		return make_uniq<FilterLocalSink>(context.client, *holds, ordinary.get(), children[0].get().types);
	}
	SinkResultType Sink(ExecutionContext &, DataChunk &input, OperatorSinkInput &sink_input) const override {
		auto &state = sink_input.local_state.Cast<FilterLocalSink>().state;
		auto &saved = sink_input.global_state.Cast<FilterSink>().deferred;
		auto &data = holds->bind_info->Cast<HoldsData>();
		SelectionVector selection(STANDARD_VECTOR_SIZE);
		auto count = input.size();
		if (ordinary) {
			count = state.ordinary_executor.SelectExpression(input, selection);
		} else {
			for (idx_t row = 0; row < count; row++)
				selection.set_index(row, row);
		}
		state.eligible.Reference(input);
		state.eligible.Slice(selection, count);
		state.values.Reset();
		state.arguments.Execute(state.eligible, state.values);
		vector<Pair> pairs;
		vector<Record> records;
		DataChunk deferred_rows;
		deferred_rows.Initialize(Allocator::DefaultAllocator(), deferred_types);
		idx_t valid = 0;
		for (idx_t row = 0; row < count; row++) {
			Pair pair;
			Record record;
			if (!ReadDecision(data, state.values, row, pair, record))
				continue;
			saved.Add(data, pair, record);
			pairs.push_back(std::move(pair));
			records.push_back(std::move(record));
			for (idx_t column = 0; column < input.ColumnCount(); column++) {
				deferred_rows.SetValue(column, valid, state.eligible.GetValue(column, row));
			}
			for (idx_t column = 0; column < state.values.ColumnCount(); column++) {
				deferred_rows.SetValue(input.ColumnCount() + column, valid, state.values.GetValue(column, row));
			}
			valid++;
		}
		if (valid) {
			deferred_rows.SetCardinality(valid);
			saved.rows.Append(deferred_rows);
		}
		if (data.pairwise)
			Judge(data, pairs);
		else
			JudgeRecords(data, records);
		return SinkResultType::NEED_MORE_INPUT;
	}
	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &) const override {
		return make_uniq<ReplayState>();
	}
	SourceResultType GetData(ExecutionContext &, DataChunk &output, OperatorSourceInput &input) const override {
		auto &state = input.global_state.Cast<ReplayState>();
		auto &saved = sink_state->Cast<FilterSink>().deferred;
		auto &data = holds->bind_info->Cast<HoldsData>();
		if (!state.initialized) {
			FinishDecisions(data, saved.pairs, saved.records);
			saved.rows.InitializeScan(state.scan);
			saved.rows.InitializeScanChunk(state.rows);
			state.values.Initialize(Allocator::DefaultAllocator(), InputTypes(*holds));
			state.initialized = true;
		}
		SelectionVector selected(STANDARD_VECTOR_SIZE);
		while (saved.rows.Scan(state.scan, state.rows)) {
			ReadSavedValues(state.rows, state.values, children[0].get().types.size());
			idx_t found = 0;
			for (idx_t row = 0; row < state.rows.size(); row++) {
				Pair pair;
				Record record;
				ReadDecision(data, state.values, row, pair, record);
				if (DecisionMatches(data, pair, record))
					selected.set_index(found++, row);
			}
			if (!found)
				continue;
			output.SetCardinality(state.rows.size());
			for (idx_t column = 0; column < output.ColumnCount(); column++) {
				output.data[column].Reference(
				    state.rows.data[projection_map.empty() ? column : projection_map[column]]);
			}
			output.Slice(selected, found);
			return SourceResultType::HAVE_MORE_OUTPUT;
		}
		return SourceResultType::FINISHED;
	}
	OperatorResultType Execute(ExecutionContext &, DataChunk &input, DataChunk &output, GlobalOperatorState &,
	                           OperatorState &operator_state) const override {
		auto &state = operator_state.Cast<FilterState>();
		auto &data = holds->bind_info->Cast<HoldsData>();
		SelectionVector selection(STANDARD_VECTOR_SIZE), matches(STANDARD_VECTOR_SIZE);
		auto count = input.size();
		if (ordinary) {
			count = state.ordinary_executor.SelectExpression(input, selection);
		} else {
			for (idx_t row = 0; row < count; row++) {
				selection.set_index(row, row);
			}
		}
		state.eligible.Reference(input);
		state.eligible.Slice(selection, count);
		state.values.Reset();
		state.arguments.Execute(state.eligible, state.values);
		vector<Pair> pairs;
		vector<Record> records;
		vector<idx_t> rows;
		for (idx_t row = 0; row < count; row++) {
			Pair pair;
			Record record;
			if (ReadDecision(data, state.values, row, pair, record)) {
				pairs.push_back(std::move(pair));
				records.push_back(std::move(record));
				rows.push_back(selection.get_index(row));
			}
		}
		if (data.pairwise) {
			Judge(data, pairs);
		} else {
			JudgeRecords(data, records);
		}
		FinishDecisions(data, pairs, records);
		idx_t found = 0;
		for (idx_t i = 0; i < pairs.size(); i++) {
			if (DecisionMatches(data, pairs[i], records[i])) {
				matches.set_index(found++, rows[i]);
			}
		}
		output.SetCardinality(input.size());
		for (idx_t column = 0; column < output.ColumnCount(); column++) {
			output.data[column].Reference(input.data[projection_map.empty() ? column : projection_map[column]]);
		}
		output.Slice(matches, found);
		return OperatorResultType::NEED_MORE_INPUT;
	}

private:
	unique_ptr<Expression> condition;
	unique_ptr<Expression> ordinary;
	optional_ptr<BoundFunctionExpression> holds;
	vector<idx_t> projection_map;
	vector<LogicalType> deferred_types;
	bool global_calibration = false;
};

LogicalSemanticFilter::LogicalSemanticFilter(LogicalFilter &filter) : projection_map(filter.projection_map) {
	children = std::move(filter.children);
	auto conjunction = make_uniq<BoundConjunctionExpression>(ExpressionType::CONJUNCTION_AND);
	conjunction->children = std::move(filter.expressions);
	expressions.push_back(std::move(conjunction));
	estimated_cardinality = filter.estimated_cardinality;
}

vector<ColumnBinding> LogicalSemanticFilter::GetColumnBindings() {
	return MapBindings(children[0]->GetColumnBindings(), projection_map);
}

void LogicalSemanticFilter::ResolveTypes() {
	types = MapTypes(children[0]->types, projection_map);
}

PhysicalOperator &LogicalSemanticFilter::CreatePlan(ClientContext &, PhysicalPlanGenerator &planner) {
	auto &child = planner.CreatePlan(*children[0]);
	return planner.Make<PhysicalSemanticFilter>(*this, child, std::move(expressions[0]));
}

} // namespace jevdb
} // namespace duckdb
