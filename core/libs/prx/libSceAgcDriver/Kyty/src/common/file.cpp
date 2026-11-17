#include "common/file.h"

#include <cstdarg>
#include <cstdio>
#include <system_error>
#include <vector>

// AnyPS5: Common::File on top of the C stdio streams (KytyPS5's archive-aware file layer is not
// needed by the GPU stack, which only writes caches and dumps).

namespace Common {

struct File::FilePrivate {
	std::FILE* stream = nullptr;
};

File::File(): m_p(std::make_unique<FilePrivate>()) {}

File::File(const std::filesystem::path& name): File() {
	Open(name, Mode::Read);
}

File::File(const std::filesystem::path& name, Mode mode): File() {
	Open(name, mode);
}

File::~File() {
	Close();
}

bool File::Create(const std::filesystem::path& name) {
	Close();
	m_file_name = name;
#if defined(_WIN32)
	m_p->stream = _wfopen(name.c_str(), L"w+b");
#else
	m_p->stream = std::fopen(name.c_str(), "w+b");
#endif
	return m_p->stream != nullptr;
}

bool File::Open(const std::filesystem::path& name, Mode mode) {
	Close();
	m_file_name = name;
#if defined(_WIN32)
	const wchar_t* flags = mode == Mode::Read ? L"rb" : mode == Mode::Write ? L"wb" : L"r+b";
	m_p->stream          = _wfopen(name.c_str(), flags);
#else
	const char* flags = mode == Mode::Read ? "rb" : mode == Mode::Write ? "wb" : "r+b";
	m_p->stream       = std::fopen(name.c_str(), flags);
#endif
	return m_p->stream != nullptr;
}

void File::Close() {
	if (m_p != nullptr && m_p->stream != nullptr) {
		std::fclose(m_p->stream);
		m_p->stream = nullptr;
	}
}

bool File::Flush() {
	return m_p->stream != nullptr && std::fflush(m_p->stream) == 0;
}

uint64_t File::Size() const {
	if (m_p->stream == nullptr) {
		return 0;
	}
	const auto position = _ftelli64(m_p->stream);
	_fseeki64(m_p->stream, 0, SEEK_END);
	const auto size = _ftelli64(m_p->stream);
	_fseeki64(m_p->stream, position, SEEK_SET);
	return size < 0 ? 0 : static_cast<uint64_t>(size);
}

uint64_t File::Remaining() const {
	const auto size     = Size();
	const auto position = Tell();
	return position < size ? size - position : 0;
}

bool File::Seek(uint64_t offset) {
	return m_p->stream != nullptr &&
	       _fseeki64(m_p->stream, static_cast<int64_t>(offset), SEEK_SET) == 0;
}

uint64_t File::Tell() const {
	if (m_p->stream == nullptr) {
		return 0;
	}
	const auto position = _ftelli64(m_p->stream);
	return position < 0 ? 0 : static_cast<uint64_t>(position);
}

bool File::Truncate(uint64_t /*size*/) {
	return false;
}

bool File::Unlink() {
	Close();
	std::error_code error;
	return std::filesystem::remove(m_file_name, error);
}

bool File::IsInvalid() const {
	return m_p->stream == nullptr;
}

std::optional<File::Times> File::GetTimes() const {
	return std::nullopt;
}

void File::Read(void* data, uint32_t size, uint32_t* bytes_read) {
	const auto done =
	    m_p->stream != nullptr ? std::fread(data, 1, size, m_p->stream) : std::size_t {0};
	if (bytes_read != nullptr) {
		*bytes_read = static_cast<uint32_t>(done);
	}
}

void File::Write(const void* data, uint32_t size, uint32_t* bytes_written) {
	const auto done =
	    m_p->stream != nullptr ? std::fwrite(data, 1, size, m_p->stream) : std::size_t {0};
	if (bytes_written != nullptr) {
		*bytes_written = static_cast<uint32_t>(done);
	}
}

void File::Printf(const char* format, ...) {
	if (m_p->stream == nullptr) {
		return;
	}
	va_list args;
	va_start(args, format);
	std::vfprintf(m_p->stream, format, args);
	va_end(args);
}

std::vector<std::byte> File::ReadWholeBuffer() {
	std::vector<std::byte> bytes(static_cast<std::size_t>(Remaining()));
	uint32_t               done = 0;
	if (!bytes.empty()) {
		Read(bytes.data(), static_cast<uint32_t>(bytes.size()), &done);
	}
	bytes.resize(done);
	return bytes;
}

uint64_t File::Size(const std::filesystem::path& name) {
	std::error_code error;
	const auto      size = std::filesystem::file_size(name, error);
	return error ? 0 : size;
}

std::optional<File::Info> File::GetInfo(const std::filesystem::path& name) {
	std::error_code error;
	const auto      status = std::filesystem::status(name, error);
	if (error || !std::filesystem::exists(status)) {
		return std::nullopt;
	}
	const bool is_file = std::filesystem::is_regular_file(status);
	return Info {is_file, is_file ? Size(name) : 0};
}

std::optional<File::Times> File::GetTimes(const std::filesystem::path& /*name*/) {
	return std::nullopt;
}

bool File::IsDirectoryExisting(const std::filesystem::path& path) {
	std::error_code error;
	return std::filesystem::is_directory(path, error);
}

bool File::IsFileExisting(const std::filesystem::path& name) {
	std::error_code error;
	return std::filesystem::is_regular_file(name, error);
}

bool File::CreateDirectory(const std::filesystem::path& path) {
	std::error_code error;
	std::filesystem::create_directory(path, error);
	return !error;
}

bool File::CreateDirectories(const std::filesystem::path& path) {
	std::error_code error;
	std::filesystem::create_directories(path, error);
	return !error;
}

bool File::DeleteFile(const std::filesystem::path& name) {
	std::error_code error;
	return std::filesystem::remove(name, error);
}

std::vector<File::DirEntry> File::GetDirEntries(const std::filesystem::path& path) {
	std::vector<DirEntry> entries;
	std::error_code       error;
	for (const auto& entry: std::filesystem::directory_iterator(path, error)) {
		entries.push_back({entry.path().filename().string(), entry.is_regular_file()});
	}
	return entries;
}

bool File::RenameFile(const std::filesystem::path& src, const std::filesystem::path& dst) {
	std::error_code error;
	std::filesystem::rename(src, dst, error);
	return !error;
}

} // namespace Common
