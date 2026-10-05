#pragma once

#include "jevdb.hpp"
#include "httplib.hpp"
#include <atomic>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <random>
#include <sstream>
#include <thread>

namespace duckdb {
namespace jevdb {

// Retry only transient transport failures and rate/server responses. Keep the
// exact body on every attempt; permanent HTTP and certificate errors fail once.
// Backoff is jittered so that concurrent workers do not retry in step. A
// Retry-After header extends the wait up to max_retry_after. The wait ends
// early when the query is interrupted; the caller then raises the interrupt.
inline duckdb_httplib_openssl::Result PostWithRetries(duckdb_httplib_openssl::Client &client, const string &path,
                                                      const string &body, const RetryConfig &config, idx_t &retries,
                                                      const std::atomic<bool> *interrupted) {
	thread_local std::mt19937_64 random {std::random_device {}()};
	double backoff = 1;
	while (true) {
		auto result = client.Post(path, body, "application/json");
		bool transient;
		if (result) {
			transient = result->status == 429 || (result->status >= 500 && result->status < 600);
		} else {
			using Error = duckdb_httplib_openssl::Error;
			auto error = result.error();
			transient = error == Error::Connection || error == Error::ConnectionTimeout || error == Error::Read ||
			            error == Error::Write || error == Error::SSLConnection || error == Error::ProxyConnection;
		}
		if (!transient || retries == config.max_retries || (interrupted && interrupted->load()))
			return result;
		// Between half of the capped backoff and all of it.
		double delay = MinValue(backoff, config.max_delay) * std::uniform_real_distribution<double>(0.5, 1.0)(random);
		if (result && result->has_header("Retry-After")) {
			auto header = result->get_header_value("Retry-After");
			double requested = 0;
			char *end;
			auto seconds = std::strtod(header.c_str(), &end);
			if (end != header.c_str() && *end == '\0' && std::isfinite(seconds)) {
				requested = seconds;
			} else {
				std::tm date = {};
				std::istringstream stream(header);
				stream >> std::get_time(&date, "%a, %d %b %Y %H:%M:%S GMT");
				if (!stream.fail())
					requested = double(timegm(&date) - std::time(nullptr));
			}
			delay = MaxValue(delay, MinValue(requested, config.max_retry_after));
		}
		const auto deadline =
		    std::chrono::steady_clock::now() + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
		                                           std::chrono::duration<double>(MaxValue(delay, 0.0)));
		while (std::chrono::steady_clock::now() < deadline) {
			if (interrupted && interrupted->load())
				return result;
			std::this_thread::sleep_for(MinValue<std::chrono::steady_clock::duration>(
			    deadline - std::chrono::steady_clock::now(), std::chrono::milliseconds(50)));
		}
		retries++;
		backoff = MinValue(backoff * 2, config.max_delay);
	}
}

} // namespace jevdb
} // namespace duckdb
