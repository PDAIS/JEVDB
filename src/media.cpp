#include "jevdb.hpp"
#include "duckdb/common/file_system.hpp"
#include "duckdb/common/type_visitor.hpp"
#include "duckdb/common/types/blob.hpp"
#include "duckdb/common/types/uuid.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {
namespace jevdb {

static const string MEDIA_PREFIX = "jevdb-media-";

static LogicalType MediaType() {
	auto type = LogicalType::STRUCT({{"type", LogicalType::VARCHAR}, {"path", LogicalType::VARCHAR}});
	type.SetAlias("JEV_MEDIA");
	return type;
}

bool IsMedia(const LogicalType &type) {
	return type.id() == LogicalTypeId::STRUCT && type.GetAlias() == "JEV_MEDIA";
}

bool ContainsMedia(const LogicalType &type) {
	return TypeVisitor::Contains(type, IsMedia);
}

static unique_ptr<FunctionData> BindMedia(ClientContext &context, ScalarFunction &function,
                                          vector<unique_ptr<Expression>> &arguments) {
	if (!arguments[0]->IsFoldable())
		throw InvalidInputException("jev_media type must be a constant string");
	auto type = ExpressionExecutor::EvaluateScalar(context, *arguments[0]);
	if (type.IsNull() || type.type() != LogicalType::VARCHAR || StringValue::Get(type) != "image")
		throw InvalidInputException("jev_media currently supports only type 'image'");
	auto &source = arguments[1]->return_type;
	if (source.id() != LogicalTypeId::STRUCT || IsMedia(source))
		throw InvalidInputException("jev_media source must be a STRUCT with one VARCHAR field named 'path'");
	auto &fields = StructType::GetChildTypes(source);
	if (fields.size() != 1 || fields[0].first != "path" ||
	    (fields[0].second != LogicalType::VARCHAR && fields[0].second.id() != LogicalTypeId::SQLNULL))
		throw InvalidInputException("jev_media source must be a STRUCT with one VARCHAR field named 'path'");
	function.arguments[1] = source;
	return nullptr;
}

static void ExecuteMedia(DataChunk &input, ExpressionState &, Vector &result) {
	for (idx_t row = 0; row < input.size(); row++) {
		auto source = input.GetValue(1, row);
		if (source.IsNull() || StructValue::GetChildren(source)[0].IsNull()) {
			result.SetValue(row, Value(result.GetType()));
			continue;
		}
		result.SetValue(row,
		                Value::STRUCT(result.GetType(), {input.GetValue(0, row), StructValue::GetChildren(source)[0]}));
	}
}

void RegisterMedia(ExtensionLoader &loader) {
	loader.RegisterType("JEV_MEDIA", MediaType());
	ScalarFunction function("jev_media", {LogicalType::VARCHAR, LogicalType::ANY}, MediaType(), ExecuteMedia,
	                        BindMedia);
	function.null_handling = FunctionNullHandling::SPECIAL_HANDLING;
	loader.RegisterFunction(function);
}

bool MediaText(QueryState &state, const Value &value, string &text) {
	if (value.IsNull())
		return false;
	if (IsMedia(value.type())) {
		auto &fields = StructValue::GetChildren(value);
		if (fields[0].IsNull() || fields[1].IsNull())
			return false;
		if (StringValue::Get(fields[0]) != "image")
			throw InvalidInputException("jev_media currently supports only type 'image'");
		auto &path = StringValue::Get(fields[1]);
		if (path.empty() || path.find('\0') != string::npos || path.find("://") != string::npos ||
		    FileSystem::IsRemoteFile(path))
			throw InvalidInputException("jev_media path must name a local image file");
		std::lock_guard<std::mutex> guard(state.mutex);
		auto &reference = state.media_paths[path];
		if (reference.empty()) {
			reference = MEDIA_PREFIX + UUID::ToString(UUIDv4::GenerateRandomUUID());
			state.media[reference].path = path;
		}
		text = reference;
	} else if (ContainsMedia(value.type())) {
		bool record = value.type().id() == LogicalTypeId::STRUCT;
		if (!record && value.type().id() != LogicalTypeId::LIST)
			throw InvalidInputException("Jev media can be nested in STRUCTs or LISTs only");
		auto &children = record ? StructValue::GetChildren(value) : ListValue::GetChildren(value);
		text = record ? "{" : "[";
		for (idx_t i = 0; i < children.size(); i++) {
			string child;
			if (!MediaText(state, children[i], child))
				return false;
			if (i)
				text += ", ";
			if (record)
				text += StructType::GetChildName(value.type(), i) + ": ";
			text += child;
		}
		text += record ? "}" : "]";
	} else {
		text = value.ToString();
	}
	return true;
}

static idx_t BigEndian(const string &bytes, idx_t start, idx_t count) {
	idx_t value = 0;
	for (idx_t i = 0; i < count; i++)
		value = (value << 8) | uint8_t(bytes[start + i]);
	return value;
}

static void LoadImage(QueryState &state, MediaAsset &asset) {
	if (!asset.base64.empty())
		return;
	auto &fs = FileSystem::GetFileSystem(*state.context);
	auto file = fs.OpenFile(asset.path, FileFlags::FILE_FLAGS_READ);
	if (!file->OnDiskFile())
		throw InvalidInputException("jev_media requires a regular local file: %s", asset.path);
	auto size = file->GetFileSize();
	if (!size || size > 16 * 1024 * 1024)
		throw InvalidInputException("Jev images must be nonempty and at most 16 MiB: %s", asset.path);
	string bytes(size, '\0');
	file->Read(&bytes[0], size, 0);
	idx_t width = 0, height = 0;
	if (size >= 24 && bytes.compare(0, 8, "\x89PNG\r\n\x1a\n", 8) == 0 && bytes.compare(12, 4, "IHDR") == 0) {
		asset.format = "png";
		width = BigEndian(bytes, 16, 4);
		height = BigEndian(bytes, 20, 4);
	} else if (size >= 4 && uint8_t(bytes[0]) == 0xff && uint8_t(bytes[1]) == 0xd8) {
		asset.format = "jpg";
		idx_t pos = 2;
		while (pos + 4 <= size && uint8_t(bytes[pos++]) == 0xff) {
			while (pos < size && uint8_t(bytes[pos]) == 0xff)
				pos++;
			if (pos + 3 > size)
				break;
			auto marker = uint8_t(bytes[pos++]);
			if (marker == 0xda || marker == 0xd9)
				break;
			auto length = BigEndian(bytes, pos, 2);
			if (length < 2 || length > size - pos)
				break;
			if (marker >= 0xc0 && marker <= 0xcf && marker != 0xc4 && marker != 0xc8 && marker != 0xcc && length >= 8) {
				height = BigEndian(bytes, pos + 3, 2);
				width = BigEndian(bytes, pos + 5, 2);
				break;
			}
			pos += length;
		}
	}
	if (!width || !height)
		throw InvalidInputException("jev_media requires a PNG or JPEG image: %s", asset.path);
	if (width < 256 || height < 256 || width > 1568 || height > 1568 || width * height < 78400)
		throw InvalidInputException(
		    "Jev images require sides of 256 to 1568 pixels and at least 78400 pixels; resize '%s'", asset.path);
	asset.base64 = Blob::ToBase64(string_t(bytes));
	// Budget for decoded pixels as well as encoded bytes, including repeated references.
	asset.expanded_bytes = MaxValue<idx_t>(width * height * 4 + 65536, asset.base64.size() + 256);
}

static vector<string> References(const string &text, const std::map<string, MediaAsset> &media) {
	vector<string> references;
	idx_t pos = 0;
	while ((pos = text.find(MEDIA_PREFIX, pos)) != string::npos) {
		auto reference = text.substr(pos, MEDIA_PREFIX.size() + UUID::STRING_SIZE);
		if (media.count(reference))
			references.push_back(std::move(reference));
		pos += MEDIA_PREFIX.size() + UUID::STRING_SIZE;
	}
	return references;
}

idx_t MediaBytes(QueryState &state, const string &text) {
	std::lock_guard<std::mutex> guard(state.mutex);
	idx_t bytes = text.size();
	if (state.media.empty())
		return bytes;
	for (auto &reference : References(text, state.media)) {
		auto &asset = state.media.at(reference);
		LoadImage(state, asset);
		bytes += asset.expanded_bytes;
	}
	return bytes;
}

bool AttachMedia(QueryState &state, string &body) {
	std::lock_guard<std::mutex> guard(state.mutex);
	if (state.media.empty())
		return false;
	auto references = References(body, state.media);
	if (references.empty())
		return false;
	std::set<string> seen;
	body.pop_back();
	body += ",\"assets\":{";
	for (auto &reference : references) {
		if (!seen.insert(reference).second)
			continue;
		auto &asset = state.media.at(reference);
		LoadImage(state, asset);
		if (seen.size() > 1)
			body += ',';
		body += '"' + reference + "\":{\"type\":\"image\",\"format\":\"" + asset.format + "\",\"data_base64\":\"" +
		        asset.base64 + "\"}";
	}
	body += "}}";
	if (body.size() > 32 * 1024 * 1024)
		throw InvalidInputException("Jev image request exceeds 32 MiB; reduce jevdb_k or resize the images");
	return true;
}

} // namespace jevdb
} // namespace duckdb
