// Host-side checks of the safetensors loader (no GPU): a single model.safetensors, a sharded checkpoint with its
// index, a named adapter file, and the refusals (no checkpoint, a truncated shard, a header longer than the file,
// offsets past the data). The files are written to a fresh temporary directory, so the platform's mapping (mmap,
// or MapViewOfFile on Windows) is what is exercised.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "core/safetensors.hpp"

namespace fs = std::filesystem;

static int failures = 0;
#define EXPECT(cond)                                              \
  do {                                                            \
    if (!(cond)) {                                                \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                 \
    }                                                             \
  } while (0)

// A safetensors file: the little-endian header length, the JSON header, then `data`.
static void write_shard(const fs::path& path, const std::string& header, const std::vector<uint8_t>& data) {
  std::ofstream f(path, std::ios::binary);
  const uint64_t n = header.size();
  uint8_t len[8];
  for (int i = 0; i < 8; ++i) len[i] = static_cast<uint8_t>(n >> (8 * i));
  f.write(reinterpret_cast<const char*>(len), 8);
  f.write(header.data(), static_cast<std::streamsize>(header.size()));
  f.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
}

static void write_text(const fs::path& path, const std::string& text) {
  std::ofstream(path, std::ios::binary) << text;
}

static std::vector<uint8_t> iota_bytes(size_t n, uint8_t first) {
  std::vector<uint8_t> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = static_cast<uint8_t>(first + i);
  return v;
}

static bool throws(const fs::path& dir) {
  try {
    ragon::Checkpoint ck(dir.string());
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

static void single_file(const fs::path& dir) {
  fs::create_directories(dir);
  const std::vector<uint8_t> data = iota_bytes(4 * 2 + 3, 10);  // a 2x2 BF16 tensor, then 3 U8
  write_shard(dir / "model.safetensors",
              R"({"__metadata__":{"format":"pt"},)"
              R"("w":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},)"
              R"("b":{"dtype":"U8","shape":[3],"data_offsets":[8,11]}})",
              data);
  const ragon::Checkpoint ck(dir.string());
  EXPECT(ck.count() == 2);
  EXPECT(ck.has("w") && ck.has("b") && !ck.has("__metadata__"));
  const ragon::TensorView& w = ck.get("w");
  EXPECT(w.dtype == "BF16");
  EXPECT((w.shape == std::vector<int64_t>{2, 2}));
  EXPECT(w.numel() == 4);
  EXPECT(w.bytes == 8);
  EXPECT(std::memcmp(w.data, data.data(), 8) == 0);
  const ragon::TensorView& b = ck.get("b");
  EXPECT(b.dtype == "U8" && b.bytes == 3);
  EXPECT(b.data[0] == 18 && b.data[2] == 20);
  bool missing = false;
  try {
    (void)ck.get("absent");
  } catch (const std::exception&) {
    missing = true;
  }
  EXPECT(missing);
}

static void sharded(const fs::path& dir) {
  fs::create_directories(dir);
  const std::vector<uint8_t> a = iota_bytes(16, 0), b = iota_bytes(6, 100);
  write_shard(dir / "model-00001-of-00002.safetensors",
              R"({"x":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}})", a);
  write_shard(dir / "model-00002-of-00002.safetensors",
              R"({"y":{"dtype":"F8_E4M3","shape":[2,3],"data_offsets":[0,6]},)"
              R"("z":{"dtype":"U8","shape":[0],"data_offsets":[6,6]}})",
              b);
  write_text(dir / "model.safetensors.index.json",
             R"({"metadata":{"total_size":22},"weight_map":{"x":"model-00001-of-00002.safetensors",)"
             R"("y":"model-00002-of-00002.safetensors","z":"model-00002-of-00002.safetensors"}})");
  // A stray model.safetensors beside an index is ignored: the index names the shards.
  write_shard(dir / "model.safetensors", R"({"stray":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}})", {1});
  const ragon::Checkpoint ck(dir.string());
  EXPECT(ck.count() == 3);
  EXPECT(!ck.has("stray"));
  EXPECT(ck.get("x").dtype == "F32" && ck.get("x").bytes == 16);
  EXPECT(std::memcmp(ck.get("x").data, a.data(), 16) == 0);
  EXPECT((ck.get("y").shape == std::vector<int64_t>{2, 3}));
  EXPECT(std::memcmp(ck.get("y").data, b.data(), 6) == 0);
  EXPECT(ck.get("z").bytes == 0 && ck.get("z").numel() == 0);
  // The tensors come back in name order, whatever shard holds them.
  std::string order;
  for (const auto& [name, t] : ck.tensors()) order += name;
  EXPECT(order == "xyz");
}

static void named_file(const fs::path& dir) {
  fs::create_directories(dir);
  write_shard(dir / "adapter_model.safetensors",
              R"({"lora_A":{"dtype":"BF16","shape":[1,2],"data_offsets":[0,4]}})", iota_bytes(4, 7));
  const ragon::Checkpoint ck(dir.string(), "adapter_model.safetensors");
  EXPECT(ck.count() == 1);
  EXPECT(ck.get("lora_A").data[3] == 10);
  EXPECT(throws(dir));  // without an index or a model.safetensors the directory is no checkpoint
}

static void refusals(const fs::path& dir) {
  fs::create_directories(dir / "none");
  EXPECT(throws(dir / "none"));
  EXPECT(throws(dir / "absent"));

  fs::create_directories(dir / "truncated");
  write_text(dir / "truncated" / "model.safetensors", "abc");
  EXPECT(throws(dir / "truncated"));

  fs::create_directories(dir / "empty");
  write_text(dir / "empty" / "model.safetensors", "");
  EXPECT(throws(dir / "empty"));

  fs::create_directories(dir / "long_header");
  {
    std::ofstream f(dir / "long_header" / "model.safetensors", std::ios::binary);
    const uint8_t len[8] = {0xff, 0xff, 0, 0, 0, 0, 0, 0};
    f.write(reinterpret_cast<const char*>(len), 8);
    f << "{}";
  }
  EXPECT(throws(dir / "long_header"));

  fs::create_directories(dir / "bad_offsets");
  write_shard(dir / "bad_offsets" / "model.safetensors",
              R"({"w":{"dtype":"U8","shape":[8],"data_offsets":[0,8]}})", iota_bytes(4, 0));
  EXPECT(throws(dir / "bad_offsets"));

  fs::create_directories(dir / "reversed");
  write_shard(dir / "reversed" / "model.safetensors",
              R"({"w":{"dtype":"U8","shape":[2],"data_offsets":[3,1]}})", iota_bytes(4, 0));
  EXPECT(throws(dir / "reversed"));

  fs::create_directories(dir / "missing_shard");
  write_text(dir / "missing_shard" / "model.safetensors.index.json",
             R"({"weight_map":{"w":"model-00001-of-00001.safetensors"}})");
  EXPECT(throws(dir / "missing_shard"));
}

int main() {
  const fs::path root =
      fs::temp_directory_path() / ("ragon-safetensors-tests-" + std::to_string(std::random_device{}()));
  try {
    fs::remove_all(root);
    single_file(root / "single");
    sharded(root / "sharded");
    named_file(root / "adapter");
    refusals(root / "refusals");
  } catch (const std::exception& e) {
    std::printf("FAIL unexpected exception: %s\n", e.what());
    ++failures;
  }
  std::error_code ec;
  fs::remove_all(root, ec);
  if (failures) {
    std::printf("%d failure(s)\n", failures);
    return 1;
  }
  std::printf("safetensors tests passed\n");
  return 0;
}
