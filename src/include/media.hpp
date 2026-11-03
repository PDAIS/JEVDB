#pragma once

#include "duckdb.hpp"
#include <map>

namespace duckdb {
class ExtensionLoader;
namespace jevdb {

struct QueryState;

struct MediaAsset {
	string path;
	string format;
	string base64;
	idx_t expanded_bytes = 0;
};

// Leave room for JSON framing and the service's image re-encoding.
static constexpr idx_t MEDIA_REQUEST_BYTES = 24 * 1024 * 1024;

bool IsMedia(const LogicalType &type);
bool ContainsMedia(const LogicalType &type);
void RegisterMedia(ExtensionLoader &loader);
bool MediaText(QueryState &state, const Value &value, string &text);
idx_t MediaBytes(QueryState &state, const string &text);
bool AttachMedia(QueryState &state, string &body);

} // namespace jevdb
} // namespace duckdb
