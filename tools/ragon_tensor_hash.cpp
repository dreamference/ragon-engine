// ragon-tensor-hash DIR: every tensor of the checkpoint in DIR, read through the engine's loader (core/safetensors:
// mmap on Linux, MapViewOfFile on Windows), as one line each in name order:
//   name <TAB> dtype <TAB> shape (comma-separated) <TAB> bytes <TAB> SHA-256 of the tensor's bytes
// followed by "tensors N". Two copies of a checkpoint (the model host's and a node's) compare line for line; a
// Python reader of the same files (hashlib) gives the same lines, so the loader's views are checked too.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>

#include "core/safetensors.hpp"

namespace {

// FIPS 180-4 SHA-256.
class Sha256 {
 public:
  void update(const uint8_t* p, size_t n) {
    len_ += n;
    if (fill_) {
      const size_t take = n < 64 - fill_ ? n : 64 - fill_;
      std::memcpy(buf_ + fill_, p, take);
      fill_ += take, p += take, n -= take;
      if (fill_ < 64) return;
      block(buf_);
      fill_ = 0;
    }
    for (; n >= 64; p += 64, n -= 64) block(p);
    std::memcpy(buf_, p, n);
    fill_ = n;
  }
  std::string hex() {
    const uint64_t bits = len_ * 8;
    const uint8_t one = 0x80, zero = 0;
    update(&one, 1);
    while (fill_ != 56) update(&zero, 1);
    uint8_t L[8];
    for (int i = 0; i < 8; ++i) L[i] = static_cast<uint8_t>(bits >> (56 - 8 * i));
    update(L, 8);
    char out[65];
    for (int i = 0; i < 8; ++i) std::snprintf(out + 8 * i, 9, "%08x", h_[i]);
    return std::string(out, 64);
  }

 private:
  static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
  void block(const uint8_t* p) {
    static constexpr uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = uint32_t(p[4 * i]) << 24 | uint32_t(p[4 * i + 1]) << 16 | uint32_t(p[4 * i + 2]) << 8 | p[4 * i + 3];
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t t1 = h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
      const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
      h = g, g = f, f = e, e = d + t1, d = c, c = b, b = a, a = t1 + t2;
    }
    h_[0] += a, h_[1] += b, h_[2] += c, h_[3] += d, h_[4] += e, h_[5] += f, h_[6] += g, h_[7] += h;
  }
  uint32_t h_[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  uint8_t buf_[64] = {};
  size_t fill_ = 0;
  uint64_t len_ = 0;
};

std::string sha256(const void* p, size_t n) {
  Sha256 s;
  s.update(static_cast<const uint8_t*>(p), n);
  return s.hex();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: ragon-tensor-hash CHECKPOINT_DIR\n");
    return 2;
  }
  // The FIPS 180-4 examples first: a wrong hash here would make every comparison meaningless.
  const std::string abc = sha256("abc", 3), empty = sha256("", 0);
  const char* two = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
  if (abc != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" ||
      empty != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" ||
      sha256(two, std::strlen(two)) != "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1") {
    std::fprintf(stderr, "ragon-tensor-hash: SHA-256 self-check failed\n");
    return 1;
  }
  try {
    const ragon::Checkpoint ck(argv[1]);
    for (const auto& [name, t] : ck.tensors()) {
      std::string shape;
      for (size_t i = 0; i < t.shape.size(); ++i) shape += (i ? "," : "") + std::to_string(t.shape[i]);
      std::printf("%s\t%s\t%s\t%llu\t%s\n", name.c_str(), t.dtype.c_str(), shape.c_str(),
                  static_cast<unsigned long long>(t.bytes), sha256(t.data, t.bytes).c_str());
    }
    std::printf("tensors %zu\n", ck.count());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ragon-tensor-hash: %s\n", e.what());
    return 1;
  }
  return 0;
}
