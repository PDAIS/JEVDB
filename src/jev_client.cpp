#include "jevdb.hpp"
#include "http_retry.hpp"
#include "yyjson.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <set>
#include <thread>

namespace duckdb {
namespace jevdb {

using namespace duckdb_yyjson;
static const char *QUESTION = "Does `condition` hold between `left` and `right`?";
static const char *RECORD_QUESTION = "Does `condition` hold for this record?";
static const char *CHOICE_QUESTION = "Follow `condition` and choose exactly one category for this record.";
static const char *SCORE_QUESTION = "Follow `condition` and rate this record against the supplied ordered levels.";
using Clock = std::chrono::steady_clock;

static int64_t Elapsed(Clock::time_point start) {
	return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count();
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

struct Request {
	string body;
	// Keep indices into the distinct input vector instead of copying both texts
	// into every request. This changes storage, not ordering or batching rules.
	vector<idx_t> indices;
};

static vector<Request> BuildRequests(const Config &config, const string &condition, const vector<Pair> &pairs) {
	std::map<string, idx_t> left_seen, right_seen;
	double left_chars = 0, right_chars = 0;
	for (auto &pair : pairs) {
		left_seen[pair.first]++;
		right_seen[pair.second]++;
		left_chars += pair.first.size();
		right_chars += pair.second.size();
	}
	auto layout = config.layout;
	if (layout == "auto") {
		auto groups = MinValue(left_seen.size(), right_seen.size());
		layout = pairs.size() >= config.auto_min_group * groups ? "star" : "pack";
	}
	bool anchor_left = config.anchor == "left";
	if (config.anchor == "auto") {
		anchor_left =
		    left_seen.size() != right_seen.size() ? left_seen.size() < right_seen.size() : left_chars >= right_chars;
	}
	vector<Request> requests;
	auto emit = [&](const string &state, const vector<idx_t> &indices) {
		Request request;
		request.indices = indices;
		request.body = "{\"model\":" + Quote(config.model) + ",\"state\":" + state + ",\"questions\":{";
		for (idx_t slot = 0; slot < indices.size(); slot++) {
			if (slot) {
				request.body += ',';
			}
			auto &pair = pairs[indices[slot]];
			string instructions;
			if (layout == "sep") {
				instructions = Quote(QUESTION);
			} else if (layout == "pack") {
				instructions = "{\"left\":" + Quote(pair.first) + ",\"right\":" + Quote(pair.second) +
				               ",\"question\":" + Quote(QUESTION) + "}";
			} else {
				instructions = string("{\"") + (anchor_left ? "right" : "left") +
				               "\":" + Quote(anchor_left ? pair.second : pair.first) +
				               ",\"question\":" + Quote(QUESTION) + "}";
			}
			request.body +=
			    Quote("p" + std::to_string(slot)) + ":{\"type\":\"noul\",\"instructions\":" + instructions + "}";
		}
		request.body += "}}";
		requests.push_back(std::move(request));
	};
	auto state = "{\"condition\":" + Quote(condition);
	if (layout == "sep") {
		for (idx_t i = 0; i < pairs.size(); i++) {
			emit(state + ",\"left\":" + Quote(pairs[i].first) + ",\"right\":" + Quote(pairs[i].second) + "}", {i});
		}
		return requests;
	}
	std::map<string, vector<idx_t>> groups;
	if (layout == "pack") {
		for (idx_t i = 0; i < pairs.size(); i++) {
			groups[""].push_back(i);
		}
	} else {
		for (idx_t i = 0; i < pairs.size(); i++) {
			groups[anchor_left ? pairs[i].first : pairs[i].second].push_back(i);
		}
	}
	for (auto &group : groups) {
		auto group_state = layout == "pack"
		                       ? state + "}"
		                       : state + ",\"" + (anchor_left ? "left" : "right") + "\":" + Quote(group.first) + "}";
		double room = config.budget_tokens - (layout == "pack" ? 0 : group.first.size() / 3.5);
		double used = 0;
		vector<idx_t> batch;
		for (auto index : group.second) {
			auto &pair = pairs[index];
			double size = layout == "pack" ? pair.first.size() / 3.5 + pair.second.size() / 3.5
			                               : (anchor_left ? pair.second.size() : pair.first.size()) / 3.5;
			if (!batch.empty() && (batch.size() == config.k || used + size > room)) {
				emit(group_state, batch);
				batch.clear();
				used = 0;
			}
			batch.push_back(index);
			used += size;
		}
		if (!batch.empty()) {
			emit(group_state, batch);
		}
	}
	return requests;
}

static vector<Request> BuildRecordRequests(const HoldsData &data, const vector<Record> &records) {
	vector<Request> requests;
	const auto per_request = data.config.layout == "sep" ? idx_t(1) : data.config.k;
	const auto criteria_size = data.kind == QuestionKind::NOUL ? 0.0 : data.criteria_json.size() / 3.5;
	double used = 0;
	for (idx_t index = 0; index < records.size(); index++) {
		auto &record = records[index];
		double size = criteria_size;
		for (auto &field : record) {
			size += (field.first.size() + field.second.size()) / 3.5;
		}
		if (requests.empty() || requests.back().indices.size() == per_request ||
		    used + size > data.config.budget_tokens) {
			if (!requests.empty()) {
				requests.back().body += "}}";
			}
			Request request;
			request.body = "{\"model\":" + Quote(data.config.model) +
			               ",\"state\":{\"condition\":" + Quote(data.condition) + "},\"questions\":{";
			requests.push_back(std::move(request));
			used = 0;
		}
		auto &request = requests.back();
		const auto slot = request.indices.size();
		if (slot) {
			request.body += ',';
		}
		const char *type = data.kind == QuestionKind::CHOICE  ? "choice"
		                   : data.kind == QuestionKind::SCORE ? "score"
		                                                      : "noul";
		const char *question = data.kind == QuestionKind::CHOICE  ? CHOICE_QUESTION
		                       : data.kind == QuestionKind::SCORE ? SCORE_QUESTION
		                                                          : RECORD_QUESTION;
		request.body += Quote("p" + std::to_string(slot)) + ":{\"type\":" + Quote(type) + ",\"instructions\":{";
		for (auto &field : record) {
			request.body += Quote(field.first) + ':' + Quote(field.second) + ',';
		}
		request.body += "\"question\":" + Quote(question) + '}';
		if (data.kind != QuestionKind::NOUL) {
			request.body += ",\"criteria\":" + data.criteria_json;
		}
		request.body += '}';
		request.indices.push_back(index);
		used += size;
	}
	if (!requests.empty()) {
		requests.back().body += "}}";
	}
	return requests;
}

struct Response {
	idx_t retries = 0;
	int status = -1;
	string body;
	string error;
	int64_t elapsed_us = 0;
};

class ClientPool {
public:
	explicit ClientPool(const Config &config) : config(config) {
		const auto scheme_end = config.endpoint.find("://");
		const auto path_start = config.endpoint.find('/', scheme_end == string::npos ? 0 : scheme_end + 3);
		host = config.endpoint.substr(0, path_start);
		path = path_start == string::npos ? "/" : config.endpoint.substr(path_start);
		if (path.back() != '/') {
			path += '/';
		}
		path += "systemone";
	}

	vector<Response> Send(const vector<Request> &requests, QueryState &state) {
		const auto start = Clock::now();
		const auto workers = MinValue<idx_t>(config.threads, requests.size());
		idx_t created = 0;
		while (clients.size() < workers) {
			auto client = make_uniq<duckdb_httplib_openssl::Client>(host);
			client->set_keep_alive(true);
			client->set_bearer_token_auth(config.api_key);
			string ca = config.ca_cert_file;
			if (ca.empty()) {
				const char *certificate_file = std::getenv("SSL_CERT_FILE");
				ca = certificate_file                            ? certificate_file
				     : std::ifstream("/etc/ssl/cert.pem").good() ? "/etc/ssl/cert.pem"
				                                                 : "";
			}
			if (!ca.empty()) {
				client->set_ca_cert_path(ca);
			}
			client->enable_server_certificate_verification(true);
			client->set_connection_timeout(30, 0);
			client->set_read_timeout(std::chrono::duration<double>(config.timeout));
			client->set_write_timeout(std::chrono::duration<double>(config.timeout));
			clients.push_back(std::move(client));
			started.push_back(false);
			created++;
		}
		vector<Response> responses(requests.size());
		vector<idx_t> reuses(workers, 0);
		std::atomic<idx_t> next {0};
		vector<std::thread> threads;
		for (idx_t worker = 0; worker < workers; worker++) {
			threads.emplace_back([&, worker]() {
				for (auto index = next++; index < requests.size(); index = next++) {
					if (started[worker]) {
						reuses[worker]++;
					}
					started[worker] = true;
					const auto request_start = Clock::now();
					auto &response = responses[index];
					auto result = PostWithRetries(*clients[worker], path, requests[index].body, config.retry,
					                              response.retries, state.interrupted);
					response.elapsed_us = Elapsed(request_start);
					if (result) {
						response.status = result->status;
						response.body = result->body;
					} else {
						response.error = duckdb_httplib_openssl::to_string(result.error());
					}
				}
			});
		}
		for (auto &thread : threads) {
			thread.join();
		}
		if (state.interrupted && state.interrupted->load())
			throw InterruptException();
		std::lock_guard<std::mutex> guard(state.mutex);
		state.stats.requests += requests.size();
		state.stats.clients_created += created;
		state.stats.send_us += Elapsed(start);
		for (auto reuse : reuses) {
			state.stats.client_reuses += reuse;
		}
		for (auto &response : responses) {
			state.stats.retries += response.retries;
			state.stats.request_us += response.elapsed_us;
		}
		return responses;
	}

private:
	Config config;
	string host;
	string path;
	vector<unique_ptr<duckdb_httplib_openssl::Client>> clients;
	vector<uint8_t> started;
};

static shared_ptr<ClientPool> Pool(HoldsData &data) {
	const auto &config = data.config;
	auto identity =
	    std::make_tuple(config.endpoint, config.api_key, config.ca_cert_file, config.threads, config.timeout,
	                    config.retry.max_retries, config.retry.max_delay, config.retry.max_retry_after);
	std::lock_guard<std::mutex> guard(data.state->mutex);
	auto &pool = data.state->pools[identity];
	if (!pool) {
		pool = make_shared_ptr<ClientPool>(config);
	}
	return pool;
}

static void AddError(QueryState &state) {
	std::lock_guard<std::mutex> guard(state.mutex);
	state.stats.errors++;
}

static string JsonValue(yyjson_val *value) {
	if (!value) {
		return "null";
	}
	auto raw = yyjson_val_write(value, 0, nullptr);
	string result(raw);
	free(raw);
	return result;
}

static string Failure(const Response &response) {
	return "{\"error\":" + Quote(response.body.empty() ? response.error : response.body) +
	       ",\"code\":" + std::to_string(response.status) + '}';
}

static string RecordText(const Record &record) {
	string text;
	for (auto &field : record) {
		text += field.first + ": " + field.second + '\n';
	}
	return text;
}

// One writer protects complete JSONL lines when independent queries use the
// same dump path. Authorization headers are never part of either dump.
static std::mutex dump_mutex;

static void Dump(const HoldsData &data, const Request &request, const Response &response, yyjson_doc *document,
                 const vector<Pair> *pairs, const vector<Record> *records, std::ofstream &raw, std::ofstream &dump) {
	if (data.config.raw_dump_path.empty() && data.config.dump_path.empty()) {
		return;
	}
	std::lock_guard<std::mutex> guard(dump_mutex);
	if (!data.config.raw_dump_path.empty()) {
		raw << "{\"request\":" << request.body << ",\"raw_body\":" << Quote(request.body) << ",\"response\":"
		    << (document && response.status == 200 ? JsonValue(yyjson_doc_get_root(document)) : Failure(response))
		    << "}\n";
		raw.flush();
		if (!raw) {
			throw IOException("JEVDB: cannot write raw request dump '%s'", data.config.raw_dump_path);
		}
	}
	if (!data.config.dump_path.empty()) {
		auto answers =
		    document && response.status == 200 ? yyjson_obj_get(yyjson_doc_get_root(document), "answers") : nullptr;
		const char *field = data.kind == QuestionKind::CHOICE  ? "choice"
		                    : data.kind == QuestionKind::SCORE ? "score"
		                                                       : "noul";
		for (idx_t slot = 0; slot < request.indices.size(); slot++) {
			const auto index = request.indices[slot];
			if (records) {
				dump << "{\"record\":" << Quote(RecordText((*records)[index]));
			} else {
				auto &pair = (*pairs)[index];
				dump << "{\"left\":" << Quote(pair.first) << ",\"right\":" << Quote(pair.second);
			}
			auto answer = yyjson_obj_get(answers, ("p" + std::to_string(slot)).c_str());
			dump << ",\"" << (data.kind == QuestionKind::NOUL ? "p" : field)
			     << "\":" << JsonValue(yyjson_obj_get(answer, field)) << "}\n";
		}
		dump.flush();
		if (!dump) {
			throw IOException("JEVDB: cannot write probability dump '%s'", data.config.dump_path);
		}
	}
}

// A stable SBF input keeps its first failure for the rest of this query, just
// as it keeps its first probability. Missing probabilities still reach stage two.
static void RememberFailedInput(const HoldsData &data, AnswerCache &cache, idx_t index, const vector<Pair> *pairs,
                                const vector<Record> *records) {
	if (!data.stable_inputs)
		return;
	if (pairs)
		cache.failed_pairs.insert((*pairs)[index]);
	else
		cache.failed_records.insert((*records)[index]);
}

static void Run(HoldsData &data, const vector<Request> &requests, const vector<Pair> *pairs,
                const vector<Record> *records) {
	const auto identity = data.Identity();
	auto responses = Pool(data)->Send(requests, *data.state);
	// Opening these files per response dominated the measured logging cost.
	std::ofstream raw, dump;
	if (!data.config.raw_dump_path.empty())
		raw.open(data.config.raw_dump_path, std::ios::app);
	if (!data.config.dump_path.empty())
		dump.open(data.config.dump_path, std::ios::app);
	// Preserve all received batches before the default fail-query policy can
	// throw on one of them. Parse each response once and retain it for validation.
	std::vector<std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)>> documents;
	documents.reserve(responses.size());
	for (idx_t index = 0; index < requests.size(); index++) {
		auto &response = responses[index];
		documents.emplace_back(yyjson_read(response.body.data(), response.body.size(), 0), yyjson_doc_free);
		Dump(data, requests[index], response, documents.back().get(), pairs, records, raw, dump);
	}
	for (idx_t index = 0; index < requests.size(); index++) {
		auto &request = requests[index];
		auto &response = responses[index];
		auto &document = documents[index];
		auto root = document ? yyjson_doc_get_root(document.get()) : nullptr;
		auto input_tokens = yyjson_obj_get(yyjson_obj_get(root, "usage"), "input_tokens");
		if (yyjson_is_uint(input_tokens)) {
			std::lock_guard<std::mutex> guard(data.state->mutex);
			data.state->stats.input_tokens += yyjson_get_uint(input_tokens);
		}
		auto answers = yyjson_obj_get(root, "answers");
		if (response.status != 200 || !yyjson_is_obj(answers)) {
			AddError(*data.state);
			if (data.config.on_error == "error" || data.kind != QuestionKind::NOUL) {
				throw IOException("JEVDB: request failed or returned no answers (HTTP %d): %s", response.status,
				                  response.body.empty() ? response.error : response.body);
			}
			std::lock_guard<std::mutex> guard(data.state->mutex);
			auto &cache = data.state->answers[identity];
			for (auto index : request.indices)
				RememberFailedInput(data, cache, index, pairs, records);
			continue;
		}
		std::lock_guard<std::mutex> guard(data.state->mutex);
		auto &cache = data.state->answers[identity];
		for (idx_t slot = 0; slot < request.indices.size(); slot++) {
			const auto id = "p" + std::to_string(slot);
			auto answer = yyjson_obj_get(answers, id.c_str());
			if (data.kind == QuestionKind::CHOICE) {
				auto choice = yyjson_obj_get(answer, "choice");
				if (!yyjson_is_str(choice)) {
					data.state->stats.errors++;
					throw IOException("JEVDB: missing or invalid choice for question '%s'", id);
				}
				string label(yyjson_get_str(choice), yyjson_get_len(choice));
				if (std::find(data.choices.begin(), data.choices.end(), label) == data.choices.end()) {
					data.state->stats.errors++;
					throw IOException("JEVDB: missing or invalid choice for question '%s'", id);
				}
				cache.record_choices[(*records)[request.indices[slot]]] = std::move(label);
				continue;
			}
			auto number = yyjson_obj_get(answer, data.kind == QuestionKind::SCORE ? "score" : "noul");
			if (!number && data.kind == QuestionKind::NOUL) {
				data.state->stats.errors++;
				if (data.config.on_error == "error")
					throw IOException("JEVDB: missing probability for question '%s'", id);
				RememberFailedInput(data, cache, request.indices[slot], pairs, records);
				continue;
			}
			if (!yyjson_is_num(number) || (data.kind == QuestionKind::SCORE &&
			                               (!std::isfinite(yyjson_get_num(number)) || yyjson_get_num(number) < 0 ||
			                                yyjson_get_num(number) > double(data.level_count - 1)))) {
				data.state->stats.errors++;
				if (data.kind == QuestionKind::SCORE) {
					throw IOException("JEVDB: missing score, or not between 0 and %g, for question '%s'",
					                  double(data.level_count - 1), id);
				}
				throw IOException("JEVDB: invalid numeric noul answer for question '%s'", id);
			}
			if (pairs) {
				cache.pair_scores[(*pairs)[request.indices[slot]]] = yyjson_get_num(number);
			} else {
				cache.record_scores[(*records)[request.indices[slot]]] = yyjson_get_num(number);
			}
		}
	}
}

void Judge(HoldsData &data, const vector<Pair> &pairs) {
	std::lock_guard<std::mutex> execution(data.state->execution_mutex);
	vector<Pair> todo;
	{
		std::lock_guard<std::mutex> guard(data.state->mutex);
		data.state->used = true;
		auto &cache = data.state->answers[data.Identity()];
		std::set<Pair> queued;
		for (auto &pair : pairs) {
			if (!cache.pair_scores.count(pair) && !cache.failed_pairs.count(pair) && queued.insert(pair).second) {
				todo.push_back(pair);
			} else {
				data.state->stats.cache_hits++;
			}
		}
		data.state->stats.pairs += todo.size();
	}
	if (!todo.empty()) {
		if (data.first_stage) {
			auto scores = data.first_stage->score_pairs(data.condition, todo);
			if (scores.size() != todo.size()) {
				throw InvalidInputException("JEVDB first-stage scorer must return one probability per input pair");
			}
			std::lock_guard<std::mutex> guard(data.state->mutex);
			auto &cache = data.state->answers[data.Identity()].pair_scores;
			for (idx_t index = 0; index < todo.size(); index++) {
				if (!std::isfinite(scores[index]) || scores[index] < 0 || scores[index] > 1)
					throw InvalidInputException("JEVDB first-stage scorer must return probabilities between 0 and 1");
				cache[todo[index]] = scores[index];
			}
		} else if (data.config.backend == "llm") {
			JudgeLlm(data, &todo, nullptr);
		} else {
			Run(data, BuildRequests(data.config, data.condition, todo), &todo, nullptr);
		}
	}
}

void JudgeRecords(HoldsData &data, const vector<Record> &records) {
	std::lock_guard<std::mutex> execution(data.state->execution_mutex);
	vector<Record> todo;
	{
		std::lock_guard<std::mutex> guard(data.state->mutex);
		data.state->used = true;
		auto &cache = data.state->answers[data.Identity()];
		std::set<Record> queued;
		for (auto &record : records) {
			const auto cached = data.kind == QuestionKind::CHOICE
			                        ? cache.record_choices.count(record)
			                        : cache.record_scores.count(record) ||
			                              (data.kind == QuestionKind::NOUL && cache.failed_records.count(record));
			if (!cached && queued.insert(record).second) {
				todo.push_back(record);
			} else {
				data.state->stats.cache_hits++;
			}
		}
		data.state->stats.records += todo.size();
	}
	if (!todo.empty()) {
		if (data.config.backend == "llm")
			JudgeLlm(data, nullptr, &todo);
		else
			Run(data, BuildRecordRequests(data, todo), nullptr, &todo);
	}
}

double Probability(HoldsData &data, const Pair &pair) {
	std::lock_guard<std::mutex> guard(data.state->mutex);
	auto &cache = data.state->answers[data.Identity()].pair_scores;
	auto found = cache.find(pair);
	return found == cache.end() ? std::numeric_limits<double>::quiet_NaN() : found->second;
}

bool Matches(HoldsData &data, const Pair &pair) {
	if (UsesCascade(data))
		return CascadeMatches(data, pair);
	return Probability(data, pair) >= data.config.threshold;
}

double RecordProbability(HoldsData &data, const Record &record) {
	std::lock_guard<std::mutex> guard(data.state->mutex);
	auto &cache = data.state->answers[data.Identity()].record_scores;
	auto found = cache.find(record);
	return found == cache.end() ? std::numeric_limits<double>::quiet_NaN() : found->second;
}

bool RecordMatches(HoldsData &data, const Record &record) {
	if (UsesCascade(data))
		return CascadeRecordMatches(data, record);
	return RecordProbability(data, record) >= data.config.threshold;
}

string RecordChoice(HoldsData &data, const Record &record) {
	std::lock_guard<std::mutex> guard(data.state->mutex);
	return data.state->answers[data.Identity()].record_choices.at(record);
}

} // namespace jevdb
} // namespace duckdb
