#include "jevdb.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/optimizer/join_order/join_order_optimizer.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_cross_product.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_top_n.hpp"
#include "duckdb/main/secret/secret_manager.hpp"
#include "yyjson.hpp"
#include <cmath>
#include <cstdlib>
#include <set>

namespace duckdb {
namespace jevdb {

using namespace duckdb_yyjson;

static unique_ptr<BaseSecret> CreateApiSecret(ClientContext &, CreateSecretInput &input) {
	auto secret = make_uniq<KeyValueSecret>(input.scope, input.type, input.provider, input.name);
	const char *env_name = input.type == "jev" ? "JEV_API_KEY" : "OPENAI_API_KEY";
	const char *label = input.type == "jev" ? "Jev" : "OpenAI";
	if (input.provider == "env") {
		auto key = std::getenv(env_name);
		if (!key || !*key) {
			throw InvalidInputException("%s is required for the %s env secret provider", env_name, label);
		}
		secret->secret_map["api_key"] = Value(key);
	} else {
		secret->TrySetValue("api_key", input);
		auto key = secret->TryGetValue("api_key");
		if (key.IsNull() || StringValue::Get(key).empty()) {
			throw InvalidInputException("%s config secrets require API_KEY", label);
		}
	}
	secret->redact_keys.insert("api_key");
	return std::move(secret);
}

static void RegisterApiSecrets(ExtensionLoader &loader) {
	for (auto name : {"jev", "openai"}) {
		SecretType type;
		type.name = name;
		type.default_provider = "config";
		type.deserializer = KeyValueSecret::Deserialize<KeyValueSecret>;
		loader.RegisterSecretType(type);
		for (auto provider : {"config", "env"}) {
			CreateSecretFunction function;
			function.secret_type = name;
			function.provider = provider;
			function.function = CreateApiSecret;
			if (function.provider == "config") {
				function.named_parameters["api_key"] = LogicalType::VARCHAR;
			}
			loader.RegisterFunction(function);
		}
	}
}

static string ApiKey(ClientContext &context, const string &endpoint, const string &type, const char *env_name) {
	auto &manager = SecretManager::Get(context);
	auto transaction = CatalogTransaction::GetSystemCatalogTransaction(context);
	auto match = manager.LookupSecret(transaction, endpoint, type);
	if (match.HasMatch()) {
		auto &secret = dynamic_cast<const KeyValueSecret &>(match.GetSecret());
		return StringValue::Get(secret.TryGetValue("api_key", true));
	}
	auto key = std::getenv(env_name);
	return key ? key : "";
}

Value Setting(ClientContext &context, const string &name) {
	Value value;
	context.TryGetCurrentSetting(name, value);
	return value;
}

unique_ptr<FunctionData> HoldsData::Copy() const {
	auto copy = make_uniq<HoldsData>();
	copy->config = config;
	copy->condition = condition;
	copy->state = state;
	copy->mode = mode;
	copy->kind = kind;
	copy->pairwise = pairwise;
	copy->stable_inputs = stable_inputs;
	copy->first_stage = first_stage;
	copy->left_fields = left_fields;
	copy->right_fields = right_fields;
	copy->argument_types = argument_types;
	copy->criteria_json = criteria_json;
	copy->choices = choices;
	copy->level_count = level_count;
	return std::move(copy);
}

bool HoldsData::Equals(const FunctionData &other_data) const {
	auto &other = other_data.Cast<HoldsData>();
	return Identity() == other.Identity() && config.layout == other.config.layout &&
	       config.anchor == other.config.anchor && config.k == other.config.k &&
	       config.threshold == other.config.threshold && config.budget_tokens == other.config.budget_tokens &&
	       config.auto_min_group == other.config.auto_min_group && config.threads == other.config.threads &&
	       config.backend == other.config.backend && config.on_error == other.config.on_error &&
	       config.retry == other.config.retry && config.prefetch == other.config.prefetch &&
	       config.api_key == other.config.api_key && config.ca_cert_file == other.config.ca_cert_file &&
	       config.timeout == other.config.timeout && config.dump_path == other.config.dump_path &&
	       config.raw_dump_path == other.config.raw_dump_path && config.cascade == other.config.cascade &&
	       mode == other.mode && kind == other.kind && pairwise == other.pairwise &&
	       stable_inputs == other.stable_inputs && left_fields == other.left_fields &&
	       right_fields == other.right_fields && criteria_json == other.criteria_json &&
	       argument_types == other.argument_types;
}

HoldsExpression::HoldsExpression(ScalarFunction function, vector<unique_ptr<Expression>> children,
                                 unique_ptr<FunctionData> data)
    : BoundFunctionExpression(LogicalType::BOOLEAN, std::move(function), std::move(children), std::move(data)) {
	return_type = this->function.return_type;
}

unique_ptr<Expression> HoldsExpression::Copy() const {
	vector<unique_ptr<Expression>> arguments;
	for (auto &child : children) {
		arguments.push_back(child->Copy());
	}
	auto copy = make_uniq<HoldsExpression>(function, std::move(arguments), bind_info->Copy());
	copy->CopyProperties(*this);
	return std::move(copy);
}

static LogicalType ReturnType(DecisionMode mode) {
	return mode == DecisionMode::HOLDS    ? LogicalType::BOOLEAN
	       : mode == DecisionMode::CHOICE ? LogicalType::VARCHAR
	                                      : LogicalType::DOUBLE;
}

static const char *FunctionName(DecisionMode mode) {
	switch (mode) {
	case DecisionMode::HOLDS:
		return "jev_holds";
	case DecisionMode::PROBABILITY:
		return "jev_probability";
	case DecisionMode::CHOICE:
		return "jev_choose";
	case DecisionMode::SCORE:
		return "jev_score";
	}
	throw InternalException("Unknown Jev decision mode");
}

static string Quote(const string &value) {
	auto doc = yyjson_mut_doc_new(nullptr);
	yyjson_mut_doc_set_root(doc, yyjson_mut_strncpy(doc, value.data(), value.size()));
	auto raw = yyjson_mut_write(doc, 0, nullptr);
	string result(raw);
	free(raw);
	yyjson_mut_doc_free(doc);
	return result;
}

static string CriteriaJSON(const Value &value) {
	if (value.IsNull()) {
		throw InvalidInputException("Jev criteria must not contain NULL");
	}
	if (value.type().id() == LogicalTypeId::VARCHAR) {
		return Quote(StringValue::Get(value));
	}
	string result;
	if (value.type().id() == LogicalTypeId::LIST) {
		result = "[";
		for (auto &child : ListValue::GetChildren(value)) {
			result += (result.size() > 1 ? "," : "") + CriteriaJSON(child);
		}
		return result + "]";
	}
	if (value.type().id() == LogicalTypeId::STRUCT) {
		result = "{";
		auto &children = StructValue::GetChildren(value);
		for (idx_t i = 0; i < children.size(); i++) {
			result +=
			    (i ? "," : "") + Quote(StructType::GetChildName(value.type(), i)) + ":" + CriteriaJSON(children[i]);
		}
		return result + "}";
	}
	throw InvalidInputException("Jev score levels must be strings, STRUCTs or lists of strings");
}

static void BindCriteria(ClientContext &context, HoldsData &data, Expression &expression) {
	if (!expression.IsFoldable()) {
		throw InvalidInputException("Jev choices and score levels must be constant");
	}
	auto criteria = ExpressionExecutor::EvaluateScalar(context, expression);
	if (criteria.IsNull()) {
		throw InvalidInputException("Jev choices and score levels must not be NULL");
	}
	if (data.mode == DecisionMode::SCORE) {
		if (criteria.type().id() != LogicalTypeId::LIST) {
			throw InvalidInputException("jev_score levels must be an ordered list");
		}
		data.level_count = ListValue::GetChildren(criteria).size();
		if (data.level_count < 2 || data.level_count > 10) {
			throw InvalidInputException("jev_score requires 2 to 10 ordered levels");
		}
		data.criteria_json = CriteriaJSON(criteria);
		return;
	}
	vector<Pair> choices;
	if (criteria.type().id() == LogicalTypeId::MAP) {
		for (auto &entry : MapValue::GetChildren(criteria)) {
			auto &kv = StructValue::GetChildren(entry);
			if (kv[0].IsNull() || kv[1].IsNull() || kv[0].type() != LogicalType::VARCHAR ||
			    kv[1].type() != LogicalType::VARCHAR) {
				throw InvalidInputException("jev_choose MAP keys and descriptions must be non-NULL strings");
			}
			choices.emplace_back(StringValue::Get(kv[0]), StringValue::Get(kv[1]));
		}
	} else if (criteria.type().id() == LogicalTypeId::LIST) {
		for (auto &entry : ListValue::GetChildren(criteria)) {
			if (!entry.IsNull() && entry.type().id() == LogicalTypeId::VARCHAR) {
				choices.emplace_back(StringValue::Get(entry), StringValue::Get(entry));
			} else if (!entry.IsNull() && entry.type().id() == LogicalTypeId::LIST) {
				auto &pair = ListValue::GetChildren(entry);
				if (pair.size() != 2 || pair[0].IsNull() || pair[1].IsNull() ||
				    pair[0].type() != LogicalType::VARCHAR || pair[1].type() != LogicalType::VARCHAR) {
					throw InvalidInputException("jev_choose choice pairs must contain a label and description");
				}
				choices.emplace_back(StringValue::Get(pair[0]), StringValue::Get(pair[1]));
			} else {
				throw InvalidInputException("jev_choose choices must be strings or label-description pairs");
			}
		}
	} else {
		throw InvalidInputException("jev_choose choices must be a MAP or list");
	}
	if (choices.size() < 2 || choices.size() > 255) {
		throw InvalidInputException("jev_choose requires 2 to 255 choices");
	}
	std::set<string> labels;
	data.criteria_json = "{";
	for (auto &choice : choices) {
		if (!labels.insert(choice.first).second) {
			throw InvalidInputException("jev_choose labels must be unique");
		}
		data.criteria_json += (data.choices.empty() ? "" : ",") + Quote(choice.first) + ":" + Quote(choice.second);
		data.choices.push_back(choice.first);
	}
	data.criteria_json += "}";
}

static bool OptionName(const string &name) {
	static const std::set<string> names = {"backend",
	                                       "max_retries",
	                                       "retry_max_delay",
	                                       "retry_after_max",
	                                       "on_error",
	                                       "endpoint",
	                                       "model",
	                                       "layout",
	                                       "anchor",
	                                       "k",
	                                       "threads",
	                                       "threshold",
	                                       "budget_tokens",
	                                       "auto_min_group",
	                                       "prefetch",
	                                       "timeout",
	                                       "ca_cert_file",
	                                       "dump_path",
	                                       "raw_dump_path",
	                                       "cascade_endpoint",
	                                       "cascade_model",
	                                       "cascade_low",
	                                       "cascade_high",
	                                       "cascade_calibrate",
	                                       "cascade_recall_loss",
	                                       "cascade_precision_loss",
	                                       "cascade_batch_size",
	                                       "cascade_fill_batches",
	                                       "cascade_threads",
	                                       "cascade_timeout",
	                                       "cascade_ca_cert_file"};
	return names.count(StringUtil::Lower(name)) != 0;
}

static bool IsOptions(const Expression &expression) {
	if (expression.return_type.id() != LogicalTypeId::STRUCT) {
		return false;
	}
	for (auto &field : StructType::GetChildTypes(expression.return_type)) {
		if (OptionName(field.first)) {
			return true;
		}
	}
	return false;
}

static void SetOption(ClientContext &context, Config &config, const string &name_p, const Value &value) {
	auto name = StringUtil::Lower(name_p);
	if (value.IsNull()) {
		throw InvalidInputException("Jev option '%s' must not be NULL", name);
	}
	if (name == "backend")
		config.backend = StringValue::Get(value.CastAs(context, LogicalType::VARCHAR));
	else if (name == "on_error")
		config.on_error = StringValue::Get(value.CastAs(context, LogicalType::VARCHAR));
	else if (name == "max_retries") {
		config.retry.max_retries = UBigIntValue::Get(value.CastAs(context, LogicalType::UBIGINT));
		config.cascade.retry.max_retries = config.retry.max_retries;
	} else if (name == "retry_max_delay") {
		config.retry.max_delay = DoubleValue::Get(value.CastAs(context, LogicalType::DOUBLE));
		config.cascade.retry.max_delay = config.retry.max_delay;
	} else if (name == "retry_after_max") {
		config.retry.max_retry_after = DoubleValue::Get(value.CastAs(context, LogicalType::DOUBLE));
		config.cascade.retry.max_retry_after = config.retry.max_retry_after;
	} else if (name == "endpoint")
		config.endpoint = StringValue::Get(value.CastAs(context, LogicalType::VARCHAR));
	else if (name == "model")
		config.model = StringValue::Get(value.CastAs(context, LogicalType::VARCHAR));
	else if (name == "layout")
		config.layout = StringValue::Get(value.CastAs(context, LogicalType::VARCHAR));
	else if (name == "anchor")
		config.anchor = StringValue::Get(value.CastAs(context, LogicalType::VARCHAR));
	else if (name == "k")
		config.k = UBigIntValue::Get(value.CastAs(context, LogicalType::UBIGINT));
	else if (name == "threads")
		config.threads = UBigIntValue::Get(value.CastAs(context, LogicalType::UBIGINT));
	else if (name == "threshold")
		config.threshold = DoubleValue::Get(value.CastAs(context, LogicalType::DOUBLE));
	else if (name == "budget_tokens")
		config.budget_tokens = DoubleValue::Get(value.CastAs(context, LogicalType::DOUBLE));
	else if (name == "auto_min_group")
		config.auto_min_group = DoubleValue::Get(value.CastAs(context, LogicalType::DOUBLE));
	else if (name == "prefetch")
		config.prefetch = BooleanValue::Get(value.CastAs(context, LogicalType::BOOLEAN));
	else if (name == "timeout")
		config.timeout = DoubleValue::Get(value.CastAs(context, LogicalType::DOUBLE));
	else if (name == "ca_cert_file")
		config.ca_cert_file = StringValue::Get(value.CastAs(context, LogicalType::VARCHAR));
	else if (name == "dump_path")
		config.dump_path = StringValue::Get(value.CastAs(context, LogicalType::VARCHAR));
	else if (name == "raw_dump_path")
		config.raw_dump_path = StringValue::Get(value.CastAs(context, LogicalType::VARCHAR));
	else if (name == "cascade_endpoint")
		config.cascade.endpoint = StringValue::Get(value.CastAs(context, LogicalType::VARCHAR));
	else if (name == "cascade_model")
		config.cascade.model = StringValue::Get(value.CastAs(context, LogicalType::VARCHAR));
	else if (name == "cascade_low")
		config.cascade.low = DoubleValue::Get(value.CastAs(context, LogicalType::DOUBLE));
	else if (name == "cascade_high")
		config.cascade.high = DoubleValue::Get(value.CastAs(context, LogicalType::DOUBLE));
	else if (name == "cascade_calibrate")
		config.cascade.calibrate = UBigIntValue::Get(value.CastAs(context, LogicalType::UBIGINT));
	else if (name == "cascade_recall_loss")
		config.cascade.recall_loss = DoubleValue::Get(value.CastAs(context, LogicalType::DOUBLE));
	else if (name == "cascade_precision_loss")
		config.cascade.precision_loss = DoubleValue::Get(value.CastAs(context, LogicalType::DOUBLE));
	else if (name == "cascade_batch_size")
		config.cascade.batch_size = UBigIntValue::Get(value.CastAs(context, LogicalType::UBIGINT));
	else if (name == "cascade_threads")
		config.cascade.threads = UBigIntValue::Get(value.CastAs(context, LogicalType::UBIGINT));
	else if (name == "cascade_fill_batches")
		config.cascade.fill_batches = BooleanValue::Get(value.CastAs(context, LogicalType::BOOLEAN));
	else if (name == "cascade_timeout")
		config.cascade.timeout = DoubleValue::Get(value.CastAs(context, LogicalType::DOUBLE));
	else if (name == "cascade_ca_cert_file")
		config.cascade.ca_cert_file = StringValue::Get(value.CastAs(context, LogicalType::VARCHAR));
	else
		throw InvalidInputException("Unknown Jev option '%s'", name);
}

static vector<string> FieldNames(const Expression &expression) {
	vector<string> names;
	if (expression.return_type.id() == LogicalTypeId::STRUCT) {
		for (auto &field : StructType::GetChildTypes(expression.return_type)) {
			if (field.first == "question" || field.first == "condition") {
				throw InvalidInputException("Jev record fields 'question' and 'condition' are reserved");
			}
			names.push_back(field.first);
		}
	} else {
		string name = "document";
		if (expression.expression_class == ExpressionClass::BOUND_COLUMN_REF) {
			name = expression.GetName();
		} else if (expression.HasAlias()) {
			name = expression.GetAlias();
		}
		if (name.empty() || name == "question" || name == "condition") {
			name = "document";
		}
		names.push_back(name);
	}
	return names;
}

bool InputRecord(const Value &value, const vector<string> &names, Record &record) {
	if (value.IsNull()) {
		return false;
	}
	if (value.type().id() == LogicalTypeId::STRUCT) {
		auto &children = StructValue::GetChildren(value);
		for (idx_t i = 0; i < children.size(); i++) {
			if (children[i].IsNull()) {
				return false;
			}
			record.emplace_back(names[i], children[i].ToString());
		}
	} else {
		record.emplace_back(names[0], value.ToString());
	}
	return true;
}

bool PairText(const Value &value, const vector<string> &names, string &text) {
	Record fields;
	if (!InputRecord(value, names, fields)) {
		return false;
	}
	for (idx_t i = 0; i < fields.size(); i++) {
		if (i)
			text += "; ";
		if (fields.size() > 1)
			text += fields[i].first + ": ";
		text += fields[i].second;
	}
	return true;
}

static unique_ptr<FunctionData> BindDecision(ClientContext &context, ScalarFunction &function,
                                             vector<unique_ptr<Expression>> &arguments) {
	auto data = make_uniq<HoldsData>();
	if (function.name == "jev_probability")
		data->mode = DecisionMode::PROBABILITY;
	else if (function.name == "jev_choose")
		data->mode = DecisionMode::CHOICE;
	else if (function.name == "jev_score")
		data->mode = DecisionMode::SCORE;
	data->kind = data->mode == DecisionMode::CHOICE  ? QuestionKind::CHOICE
	             : data->mode == DecisionMode::SCORE ? QuestionKind::SCORE
	                                                 : QuestionKind::NOUL;
	bool criteria = data->kind != QuestionKind::NOUL;
	if (arguments.size() < (criteria ? 3U : 2U) || arguments.size() > 4) {
		throw InvalidInputException(
		    "%s expects a condition, input, optional second input/criteria and constant options", function.name);
	}
	if (!arguments[0]->IsFoldable()) {
		throw InvalidInputException("Jev condition must be a constant string");
	}
	auto condition = ExpressionExecutor::EvaluateScalar(context, *arguments[0]);
	if (condition.IsNull()) {
		throw InvalidInputException("Jev condition must not be NULL");
	}
	if (condition.type() != LogicalType::VARCHAR) {
		throw InvalidInputException("Jev condition must be a string");
	}
	data->condition = StringValue::Get(condition);
	for (auto name : {"backend",
	                  "max_retries",
	                  "retry_max_delay",
	                  "retry_after_max",
	                  "on_error",
	                  "endpoint",
	                  "model",
	                  "layout",
	                  "anchor",
	                  "k",
	                  "threshold",
	                  "budget_tokens",
	                  "auto_min_group",
	                  "threads",
	                  "prefetch",
	                  "timeout",
	                  "ca_cert_file",
	                  "dump_path",
	                  "raw_dump_path",
	                  "cascade_endpoint",
	                  "cascade_model",
	                  "cascade_low",
	                  "cascade_high",
	                  "cascade_calibrate",
	                  "cascade_recall_loss",
	                  "cascade_precision_loss",
	                  "cascade_batch_size",
	                  "cascade_fill_batches",
	                  "cascade_threads",
	                  "cascade_timeout",
	                  "cascade_ca_cert_file"}) {
		SetOption(context, data->config, name, Setting(context, string("jevdb_") + name));
	}
	idx_t option_idx = DConstants::INVALID_INDEX;
	data->pairwise = !criteria && arguments.size() >= 3;
	if (criteria) {
		BindCriteria(context, *data, *arguments[2]);
		if (arguments.size() == 4)
			option_idx = 3;
	} else if (arguments.size() == 4) {
		option_idx = 3;
	} else if (arguments.size() == 3 && IsOptions(*arguments[2])) {
		data->pairwise = false;
		option_idx = 2;
	}
	if (option_idx != DConstants::INVALID_INDEX) {
		auto &expression = *arguments[option_idx];
		if (expression.return_type.id() != LogicalTypeId::STRUCT || !expression.IsFoldable()) {
			throw InvalidInputException("Jev options must be a constant STRUCT");
		}
		auto options = ExpressionExecutor::EvaluateScalar(context, expression);
		if (options.IsNull())
			throw InvalidInputException("Jev options must not be NULL");
		auto &values = StructValue::GetChildren(options);
		for (idx_t i = 0; i < values.size(); i++) {
			SetOption(context, data->config, StructType::GetChildName(options.type(), i), values[i]);
		}
	}
	if (data->config.backend != "jev" && data->config.backend != "llm")
		throw InvalidInputException("jevdb_backend must be jev or llm");
	if (data->config.on_error != "error" && data->config.on_error != "reject")
		throw InvalidInputException("jevdb_on_error must be error or reject");
	if (!std::isfinite(data->config.retry.max_delay) || data->config.retry.max_delay < 0)
		throw InvalidInputException("jevdb_retry_max_delay must be nonnegative");
	if (!std::isfinite(data->config.retry.max_retry_after) || data->config.retry.max_retry_after < 0)
		throw InvalidInputException("jevdb_retry_after_max must be nonnegative");
	if (data->config.layout != "auto" && data->config.layout != "star" && data->config.layout != "pack" &&
	    data->config.layout != "sep") {
		throw InvalidInputException("jevdb_layout must be auto, star, pack or sep");
	}
	if (data->config.anchor != "auto" && data->config.anchor != "left" && data->config.anchor != "right") {
		throw InvalidInputException("jevdb_anchor must be auto, left or right");
	}
	if (data->config.k == 0 || data->config.threads == 0) {
		throw InvalidInputException("Jev k and threads must be positive");
	}
	if (!std::isfinite(data->config.threshold) || data->config.threshold < 0 || data->config.threshold > 1) {
		throw InvalidInputException("Jev threshold must be between 0 and 1");
	}
	if (!std::isfinite(data->config.budget_tokens) || data->config.budget_tokens <= 0 ||
	    data->config.budget_tokens > 50000) {
		throw InvalidInputException("Jev budget_tokens must be positive and at most 50000");
	}
	if (!std::isfinite(data->config.auto_min_group) || data->config.auto_min_group <= 0 ||
	    !std::isfinite(data->config.timeout) || data->config.timeout <= 0) {
		throw InvalidInputException("Jev auto_min_group and timeout must be positive");
	}
	if (UsesCascade(*data)) {
		const auto &cascade = data->config.cascade;
		if (!(cascade.low <= cascade.high)) {
			throw InvalidInputException("Jev cascade_low must not exceed cascade_high");
		}
		if (cascade.batch_size == 0 || cascade.threads == 0) {
			throw InvalidInputException("Jev cascade_batch_size and cascade_threads must be positive");
		}
		if (!std::isfinite(cascade.timeout) || cascade.timeout <= 0) {
			throw InvalidInputException("Jev cascade_timeout must be positive");
		}
		if (!std::isfinite(cascade.recall_loss) || cascade.recall_loss < 0 || cascade.recall_loss > 1 ||
		    !std::isfinite(cascade.precision_loss) || cascade.precision_loss < 0 || cascade.precision_loss > 1) {
			throw InvalidInputException("Jev cascade_recall_loss and cascade_precision_loss must be between 0 and 1");
		}
		data->config.cascade.api_key = ApiKey(context, cascade.endpoint, "openai", "OPENAI_API_KEY");
	}
	data->config.api_key = data->config.backend == "llm"
	                           ? ApiKey(context, data->config.endpoint, "openai", "OPENAI_API_KEY")
	                           : ApiKey(context, data->config.endpoint, "jev", "JEV_API_KEY");
	data->left_fields = FieldNames(*arguments[1]);
	if (data->pairwise)
		data->right_fields = FieldNames(*arguments[2]);
	function.arguments.clear();
	for (auto &argument : arguments) {
		function.arguments.push_back(argument->return_type);
	}
	function.varargs = LogicalType::INVALID;
	data->argument_types = function.arguments;
	data->first_stage = BindFirstStageScorer(context, function, data->pairwise);
	data->state = context.registered_state->GetOrCreate<QueryState>("jevdb");
	data->state->interrupted = &context.interrupted;
	return std::move(data);
}

static void ExecuteDecision(DataChunk &input, ExpressionState &expression_state, Vector &result) {
	auto &function = expression_state.expr.Cast<BoundFunctionExpression>();
	auto &data = function.bind_info->Cast<HoldsData>();
	if (UsesCascade(data) && data.config.cascade.calibrate > 0) {
		throw InvalidInputException("Jev global cascade calibration requires a semantic JOIN or WHERE filter");
	}
	vector<Pair> pairs;
	vector<Record> records;
	vector<idx_t> rows;
	for (idx_t row = 0; row < input.size(); row++) {
		bool present;
		if (data.pairwise) {
			Pair pair;
			present = PairText(input.GetValue(1, row), data.left_fields, pair.first) &&
			          PairText(input.GetValue(2, row), data.right_fields, pair.second);
			if (present)
				pairs.push_back(std::move(pair));
		} else {
			Record record;
			present = InputRecord(input.GetValue(1, row), data.left_fields, record);
			if (present)
				records.push_back(std::move(record));
		}
		if (present) {
			rows.push_back(row);
		} else {
			result.SetValue(row, Value(ReturnType(data.mode)));
		}
	}
	if (data.pairwise)
		Judge(data, pairs);
	else
		JudgeRecords(data, records);
	if (UsesCascade(data)) {
		if (data.pairwise)
			FinishCascade(data, pairs);
		else
			FinishRecordCascade(data, records);
	}
	for (idx_t i = 0; i < rows.size(); i++) {
		if (data.mode == DecisionMode::CHOICE) {
			result.SetValue(rows[i], Value(RecordChoice(data, records[i])));
			continue;
		}
		if (data.mode == DecisionMode::HOLDS) {
			result.SetValue(rows[i], Value(data.pairwise ? Matches(data, pairs[i]) : RecordMatches(data, records[i])));
			continue;
		}
		auto probability = data.pairwise ? Probability(data, pairs[i]) : RecordProbability(data, records[i]);
		result.SetValue(rows[i], std::isnan(probability) ? Value(LogicalType::DOUBLE) : Value::DOUBLE(probability));
	}
}

static ScalarFunction DecisionFunction(DecisionMode mode);

static unique_ptr<Expression> BindDecisionExpression(FunctionBindExpressionInput &input) {
	vector<unique_ptr<Expression>> children;
	for (auto &child : input.children) {
		children.push_back(child->Copy());
	}
	auto &data = input.bind_data->Cast<HoldsData>();
	auto function = DecisionFunction(data.mode);
	function.arguments = data.argument_types;
	function.varargs = LogicalType::INVALID;
	return make_uniq<HoldsExpression>(std::move(function), std::move(children), input.bind_data->Copy());
}

static ScalarFunction DecisionFunction(DecisionMode mode) {
	ScalarFunction function(FunctionName(mode), {LogicalType::VARCHAR}, ReturnType(mode), ExecuteDecision,
	                        BindDecision);
	function.varargs = LogicalType::ANY;
	function.bind_expression = BindDecisionExpression;
	function.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	function.errors = FunctionErrors::CAN_THROW_RUNTIME_ERROR;
	return function;
}

ScalarFunction HoldsFunction() {
	return DecisionFunction(DecisionMode::HOLDS);
}

bool IsHolds(const Expression &expression) {
	return expression.expression_class == ExpressionClass::BOUND_FUNCTION &&
	       expression.Cast<BoundFunctionExpression>().function.name == "jev_holds";
}

bool ContainsHolds(const Expression &expression) {
	bool found = IsHolds(expression);
	ExpressionIterator::EnumerateChildren(expression, [&](const Expression &child) { found |= ContainsHolds(child); });
	return found;
}

static void KeysOverlap(DataChunk &input, ExpressionState &, Vector &result) {
	UnifiedVectorFormat left_lists, right_lists, left_keys, right_keys;
	input.data[0].ToUnifiedFormat(input.size(), left_lists);
	input.data[1].ToUnifiedFormat(input.size(), right_lists);
	ListVector::GetEntry(input.data[0]).ToUnifiedFormat(ListVector::GetListSize(input.data[0]), left_keys);
	ListVector::GetEntry(input.data[1]).ToUnifiedFormat(ListVector::GetListSize(input.data[1]), right_keys);
	auto left_entries = UnifiedVectorFormat::GetData<list_entry_t>(left_lists);
	auto right_entries = UnifiedVectorFormat::GetData<list_entry_t>(right_lists);
	auto left_strings = UnifiedVectorFormat::GetData<string_t>(left_keys);
	auto right_strings = UnifiedVectorFormat::GetData<string_t>(right_keys);
	result.SetVectorType(VectorType::FLAT_VECTOR);
	FlatVector::Validity(result).SetAllValid(input.size());
	auto overlaps = FlatVector::GetData<bool>(result);
	for (idx_t row = 0; row < input.size(); row++) {
		auto left_row = left_lists.sel->get_index(row);
		auto right_row = right_lists.sel->get_index(row);
		bool overlap = !left_lists.validity.RowIsValid(left_row) || !right_lists.validity.RowIsValid(right_row);
		if (!overlap) {
			auto &left = left_entries[left_row];
			auto &right = right_entries[right_row];
			for (idx_t key = 0; key < left.length; key++) {
				auto left_key = left_keys.sel->get_index(left.offset + key);
				if (left_keys.validity.RowIsValid(left_key)) {
					for (idx_t other = 0; other < right.length; other++) {
						auto right_key = right_keys.sel->get_index(right.offset + other);
						if (right_keys.validity.RowIsValid(right_key) &&
						    left_strings[left_key] == right_strings[right_key]) {
							overlap = true;
							break;
						}
					}
				}
				if (overlap)
					break;
			}
		}
		overlaps[row] = overlap;
	}
}

static bool WrappableCondition(const Expression &expression, idx_t &count) {
	if (IsHolds(expression)) {
		count++;
		return true;
	}
	if (expression.type == ExpressionType::CONJUNCTION_AND) {
		for (auto &child : expression.Cast<BoundConjunctionExpression>().children) {
			if (!WrappableCondition(*child, count))
				return false;
		}
		return true;
	}
	return !ContainsHolds(expression);
}

static void WrapJoins(unique_ptr<LogicalOperator> &plan) {
	for (auto &child : plan->children) {
		WrapJoins(child);
	}
	if (plan->type == LogicalOperatorType::LOGICAL_ANY_JOIN) {
		auto &join = plan->Cast<LogicalAnyJoin>();
		idx_t count = 0;
		if ((join.join_type == JoinType::INNER || join.join_type == JoinType::SEMI ||
		     join.join_type == JoinType::ANTI) &&
		    WrappableCondition(*join.condition, count) && count == 1) {
			plan = make_uniq<LogicalSemanticJoin>(join);
		}
	} else if (plan->type == LogicalOperatorType::LOGICAL_FILTER) {
		auto &filter = plan->Cast<LogicalFilter>();
		idx_t count = 0;
		bool supported = true;
		for (auto &expression : filter.expressions) {
			supported &= WrappableCondition(*expression, count);
		}
		if (supported && count == 1) {
			plan = make_uniq<LogicalSemanticFilter>(filter);
		}
	}
}

static bool IsDecision(const Expression &expression) {
	if (expression.GetExpressionClass() != ExpressionClass::BOUND_FUNCTION)
		return false;
	auto &name = expression.Cast<BoundFunctionExpression>().function.name;
	return name == "jev_holds" || name == "jev_probability" || name == "jev_choose" || name == "jev_score";
}

// Arguments read only the current row: no nested decision, no subquery, no outer column.
static bool RowArguments(const Expression &expression) {
	bool supported = true;
	ExpressionIterator::EnumerateChildren(expression, [&](const Expression &child) {
		if (IsDecision(child) || child.GetExpressionClass() == ExpressionClass::BOUND_SUBQUERY ||
		    (child.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF &&
		     child.Cast<BoundColumnRefExpression>().depth != 0))
			supported = false;
		else
			supported &= RowArguments(child);
	});
	return supported;
}

// A decision in the select list of ORDER BY ... LIMIT k is needed only for the k returned rows. When the projection
// below a zero-offset Top-N holds one decision beside plain columns and the ordering reads plain columns, the
// projection moves above the Top-N, so the remaining rows are never sent to the model.
static void DeferDecisionAfterTopN(unique_ptr<LogicalOperator> &op) {
	if (op->type == LogicalOperatorType::LOGICAL_TOP_N &&
	    op->children[0]->type == LogicalOperatorType::LOGICAL_PROJECTION) {
		auto &topn = op->Cast<LogicalTopN>();
		auto &projection = topn.children[0]->Cast<LogicalProjection>();
		idx_t decisions = 0;
		bool supported = topn.offset == 0;
		for (auto &expression : projection.expressions) {
			if (expression->GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
				supported &= expression->Cast<BoundColumnRefExpression>().depth == 0;
			} else {
				supported &= IsDecision(*expression) && RowArguments(*expression);
				decisions++;
			}
		}
		for (auto &order : topn.orders) {
			if (!supported || order.expression->GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
				supported = false;
				break;
			}
			auto &column = order.expression->Cast<BoundColumnRefExpression>();
			supported = column.depth == 0 && column.binding.table_index == projection.table_index &&
			            column.binding.column_index < projection.expressions.size() &&
			            projection.expressions[column.binding.column_index]->GetExpressionClass() ==
			                ExpressionClass::BOUND_COLUMN_REF;
		}
		if (supported && decisions == 1) {
			for (auto &order : topn.orders) {
				auto index = order.expression->Cast<BoundColumnRefExpression>().binding.column_index;
				order.expression = projection.expressions[index]->Copy();
			}
			auto moved = std::move(topn.children[0]);
			topn.children[0] = std::move(moved->children[0]);
			moved->SetEstimatedCardinality(topn.estimated_cardinality);
			moved->children[0] = std::move(op);
			op = std::move(moved);
		}
	}
	for (auto &child : op->children) {
		DeferDecisionAfterTopN(child);
	}
}

static void PreOptimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
	if (StringValue::Get(Setting(input.context, "jevdb_optimizer_mode")) == "early") {
		WrapJoins(plan);
	} else {
		PrepareSBF(input.optimizer, plan);
	}
}

static void Optimize(OptimizerExtensionInput &input, unique_ptr<LogicalOperator> &plan) {
	auto strategy = StringValue::Get(Setting(input.context, "jevdb_optimizer_mode"));
	if (strategy != "early") {
		if (BooleanValue::Get(Setting(input.context, "jevdb_reduce_inputs")))
			ReduceSemanticInputs(input.optimizer, plan);
		OptimizeSBF(input.optimizer, plan);
		WrapJoins(plan);
	}
	if (strategy == "late") {
		JoinOrderOptimizer order(input.context);
		plan = order.Optimize(std::move(plan));
	}
	DeferDecisionAfterTopN(plan);
}

static void Load(ExtensionLoader &loader) {
	RegisterApiSecrets(loader);
	auto scorer_api = CreateFirstStageScorerAPI();
	for (auto mode : {DecisionMode::HOLDS, DecisionMode::PROBABILITY, DecisionMode::CHOICE, DecisionMode::SCORE}) {
		auto function = DecisionFunction(mode);
		function.function_info = scorer_api;
		loader.RegisterFunction(function);
	}
	RegisterStats(loader);
	ScalarFunction overlap("jev_keys_overlap",
	                       {LogicalType::LIST(LogicalType::VARCHAR), LogicalType::LIST(LogicalType::VARCHAR)},
	                       LogicalType::BOOLEAN, KeysOverlap);
	overlap.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	loader.RegisterFunction(overlap);
	auto &config = DBConfig::GetConfig(loader.GetDatabaseInstance());
	config.AddExtensionOption("jevdb_backend", "First-stage backend: jev or llm", LogicalType::VARCHAR, Value("jev"));
	config.AddExtensionOption("jevdb_max_retries", "Retries after transient HTTP failures", LogicalType::UBIGINT,
	                          Value::UBIGINT(7));
	config.AddExtensionOption("jevdb_retry_max_delay", "Maximum backoff in seconds", LogicalType::DOUBLE,
	                          Value::DOUBLE(60));
	config.AddExtensionOption("jevdb_retry_after_max", "Longest wait in seconds that a Retry-After header can impose",
	                          LogicalType::DOUBLE, Value::DOUBLE(300));
	config.AddExtensionOption("jevdb_on_error", "Exhausted requests: error or reject", LogicalType::VARCHAR,
	                          Value("error"));
	config.AddExtensionOption("jevdb_endpoint", "First-stage HTTP or HTTPS endpoint", LogicalType::VARCHAR,
	                          Value("https://api.typesafe.ai/v1/"));
	config.AddExtensionOption("jevdb_model", "First-stage model", LogicalType::VARCHAR, Value("jev-1.13.0"));
	config.AddExtensionOption("jevdb_layout", "Request layout", LogicalType::VARCHAR, Value("auto"));
	config.AddExtensionOption("jevdb_anchor", "Star anchor", LogicalType::VARCHAR, Value("auto"));
	config.AddExtensionOption("jevdb_k", "Maximum judgments per request", LogicalType::UBIGINT, Value::UBIGINT(100));
	config.AddExtensionOption("jevdb_threshold", "Decision probability threshold", LogicalType::DOUBLE,
	                          Value::DOUBLE(0.5));
	config.AddExtensionOption("jevdb_budget_tokens", "Estimated request token limit", LogicalType::DOUBLE,
	                          Value::DOUBLE(50000));
	config.AddExtensionOption("jevdb_auto_min_group", "Minimum average pairs per star anchor", LogicalType::DOUBLE,
	                          Value::DOUBLE(8));
	config.AddExtensionOption("jevdb_threads", "Concurrent first-stage request workers", LogicalType::UBIGINT,
	                          Value::UBIGINT(8));
	config.AddExtensionOption("jevdb_prefetch", "Collect join pairs before requesting decisions", LogicalType::BOOLEAN,
	                          Value(true));
	config.AddExtensionOption("jevdb_timeout", "HTTP read/write timeout in seconds (connection timeout is 30 seconds)",
	                          LogicalType::DOUBLE, Value::DOUBLE(120));
	config.AddExtensionOption("jevdb_ca_cert_file", "CA certificate file for HTTPS", LogicalType::VARCHAR, Value(""));
	config.AddExtensionOption("jevdb_dump_path", "Request/probability log path", LogicalType::VARCHAR, Value(""));
	config.AddExtensionOption("jevdb_raw_dump_path", "Original request/response log path", LogicalType::VARCHAR,
	                          Value(""));
	config.AddExtensionOption("jevdb_cascade_model", "Second-stage model (empty disables cascade)",
	                          LogicalType::VARCHAR, Value(""));
	config.AddExtensionOption("jevdb_cascade_endpoint", "OpenAI-compatible second-stage endpoint", LogicalType::VARCHAR,
	                          Value("https://api.openai.com/v1/"));
	config.AddExtensionOption("jevdb_cascade_low", "Reject first-stage probabilities below this bound",
	                          LogicalType::DOUBLE, Value::DOUBLE(0.5));
	config.AddExtensionOption("jevdb_cascade_high", "Accept first-stage probabilities at or above this bound",
	                          LogicalType::DOUBLE, Value::DOUBLE(0.9));
	config.AddExtensionOption("jevdb_cascade_calibrate", "Global sample size for automatic interval calibration",
	                          LogicalType::UBIGINT, Value::UBIGINT(0));
	config.AddExtensionOption("jevdb_cascade_recall_loss", "Maximum share of calibration positives below the interval",
	                          LogicalType::DOUBLE, Value::DOUBLE(0.05));
	config.AddExtensionOption("jevdb_cascade_precision_loss",
	                          "Maximum share of calibration negatives among items above the interval",
	                          LogicalType::DOUBLE, Value::DOUBLE(0.05));
	config.AddExtensionOption("jevdb_cascade_batch_size", "Maximum second-stage judgments per request",
	                          LogicalType::UBIGINT, Value::UBIGINT(16));
	config.AddExtensionOption("jevdb_cascade_fill_batches", "Fill second-stage batches before dispatch",
	                          LogicalType::BOOLEAN, Value(true));
	config.AddExtensionOption("jevdb_cascade_threads", "Concurrent second-stage request workers", LogicalType::UBIGINT,
	                          Value::UBIGINT(8));
	config.AddExtensionOption("jevdb_cascade_timeout", "Second-stage HTTP read/write timeout in seconds",
	                          LogicalType::DOUBLE, Value::DOUBLE(120));
	config.AddExtensionOption("jevdb_cascade_ca_cert_file", "CA certificate file for second-stage HTTPS",
	                          LogicalType::VARCHAR, Value(""));
	config.AddExtensionOption("jevdb_reduce_inputs",
	                          "Evaluate ordinary joins that restrict one semantic input before it",
	                          LogicalType::BOOLEAN, Value(true));
	config.AddExtensionOption("jevdb_optimizer_mode", "Optimizer placement: late, early, explicit",
	                          LogicalType::VARCHAR, Value("explicit"));
	OptimizerExtension optimizer;
	optimizer.pre_optimize_function = PreOptimize;
	optimizer.optimize_function = Optimize;
	config.optimizer_extensions.push_back(std::move(optimizer));
}

} // namespace jevdb
} // namespace duckdb

extern "C" {
DUCKDB_CPP_EXTENSION_ENTRY(jevdb, loader) {
	duckdb::jevdb::Load(loader);
}
}
