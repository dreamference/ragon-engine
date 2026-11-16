#include "core/safetensors.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace ragon {

int64_t TensorView::numel() const {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

#ifdef _WIN32
// Windows: the whole file mapped read-only (MapViewOfFile), the same view mmap gives on Linux. The path is UTF-8.
MappedFile::MappedFile(const std::string& path) {
  const int wn = ::MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
  std::wstring wpath(wn > 0 ? wn - 1 : 0, L'\0');
  if (wn <= 0 || ::MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wn) != wn)
    throw std::runtime_error("cannot open " + path);
  HANDLE f = ::CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
  if (f == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open " + path);
  file_ = f;
  LARGE_INTEGER sz{};
  if (!::GetFileSizeEx(f, &sz) || sz.QuadPart < 0) throw std::runtime_error("cannot stat " + path);
  size_ = static_cast<size_t>(sz.QuadPart);
  if (size_ == 0) return;  // nothing to map (CreateFileMapping refuses an empty file); the caller sees size 0
  mapping_ = ::CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (!mapping_) throw std::runtime_error("cannot map " + path);
  const void* p = ::MapViewOfFile(static_cast<HANDLE>(mapping_), FILE_MAP_READ, 0, 0, 0);
  if (!p) throw std::runtime_error("cannot map " + path);
  data_ = static_cast<const uint8_t*>(p);
}

MappedFile::~MappedFile() {
  if (data_) ::UnmapViewOfFile(data_);
  if (mapping_) ::CloseHandle(static_cast<HANDLE>(mapping_));
  if (file_) ::CloseHandle(static_cast<HANDLE>(file_));
}
#else
MappedFile::MappedFile(const std::string& path) {
  fd_ = ::open(path.c_str(), O_RDONLY);
  if (fd_ < 0) throw std::runtime_error("cannot open " + path);
  struct stat st {};
  if (::fstat(fd_, &st) != 0) throw std::runtime_error("cannot stat " + path);
  size_ = static_cast<size_t>(st.st_size);
  void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (p == MAP_FAILED) throw std::runtime_error("cannot map " + path);
  data_ = static_cast<const uint8_t*>(p);
}

MappedFile::~MappedFile() {
  if (data_) ::munmap(const_cast<uint8_t*>(data_), size_);
  if (fd_ >= 0) ::close(fd_);
}
#endif

Checkpoint::Checkpoint(const std::string& dir) : dir_(dir) {
  // A sharded checkpoint has an index; a small one (the drafter) is a single model.safetensors.
  std::set<std::string> shards;
  std::ifstream index_file(dir + "/model.safetensors.index.json");
  if (index_file) {
    nlohmann::json index = nlohmann::json::parse(index_file);
    for (auto& [name, shard] : index["weight_map"].items()) shards.insert(shard.get<std::string>());
  } else if (std::ifstream(dir + "/model.safetensors")) {
    shards.insert("model.safetensors");
  } else {
    throw std::runtime_error("no model.safetensors.index.json or model.safetensors in " + dir);
  }

  for (const std::string& shard : shards) add_shard(shard);
}

Checkpoint::Checkpoint(const std::string& dir, const std::string& file) : dir_(dir) { add_shard(file); }

void Checkpoint::add_shard(const std::string& shard) {
  auto file = std::make_unique<MappedFile>(dir_ + "/" + shard);
  if (file->size() < 8) throw std::runtime_error("truncated shard " + shard);
  uint64_t header_len = 0;
  std::memcpy(&header_len, file->data(), 8);
  if (header_len > file->size() - 8) throw std::runtime_error("bad header in " + shard);
  nlohmann::json header = nlohmann::json::parse(
      std::string(reinterpret_cast<const char*>(file->data() + 8), header_len));
  const uint8_t* base = file->data() + 8 + header_len;
  for (auto& [name, info] : header.items()) {
    if (name == "__metadata__") continue;
    TensorView view;
    view.dtype = info["dtype"].get<std::string>();
    view.shape = info["shape"].get<std::vector<int64_t>>();
    auto offsets = info["data_offsets"].get<std::vector<size_t>>();
    if (offsets.size() != 2 || offsets[1] < offsets[0] ||
        offsets[1] > file->size() - 8 - header_len) {
      throw std::runtime_error("bad offsets for " + name);
    }
    view.data = base + offsets[0];
    view.bytes = offsets[1] - offsets[0];
    tensors_[name] = view;
  }
  files_.push_back(std::move(file));
}

const TensorView& Checkpoint::get(const std::string& name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) throw std::runtime_error("tensor not in checkpoint: " + name);
  return it->second;
}

}  // namespace ragon
