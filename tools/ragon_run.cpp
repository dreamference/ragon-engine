// ragon-run: loads the model and generates from a prompt; prints tokens, timings and (with --ids) the
// token ids, for checking against a reference server.
//
//   ragon-run --model DIR [--text "..." | --chat "..." | --ids 1,2,3] [--max-tokens N] [--temperature T]
//            [--prompts-file F] [--ids-out] [--prompt-ids-out] [--max-context N]
//            [--draft DIR [--block N] [--spec] [--spec-check] [--lookup 0|1|2] [--lookup-min N] [--draft-off-below X]]
//            [--graphs] [--pdl] [--attn-bulk] [--attn-prefetch N] [--prefill-bf16] [--graph-check] [--seed N]
//            [--prefix-check [--sessions N]] [--chunk-check N] [--interleave-check [--prefill-chunk-contended N]]
//            [--batch-check] [--side-prefill N] [--draft-bf16] [--kv-pages [--kv-pool-mb N] | --no-kv-pages]
//
// --lora LIBRARY/NAME loads that adapter (LORA §3) into one GPU slot and runs every session on it, so each check
// below holds with an adapter as well as on the base.
// --graphs runs each speculative step's draft, verify and commit as captured CUDA graphs (M3); --pdl launches
// the rows path's kernels with programmatic dependent launch (M3); --attn-bulk loads the rows path's attention
// tiles with bulk copies (M3); --attn-prefetch N sets how many KV tiles ahead its attention prefetches into L2
// (default 2, 0 off; M3); --prefill-bf16 keeps the prefill's DeltaNet intermediates in BF16 as production does
// (B34; not exact, off by default: MILESTONES §14); --draft-bf16 reads the drafter's fc and conv kernel projections
// in BF16 instead of the NVFP4 copies rounded at load (BACKLOG 33, the default; RAGON_DRAFT_NVFP4=0/1 overrides).
// --kv-pages keeps the full-attention KV in 64-position pages behind a page table per session (BATCHING §5.3, P4; off
// by default, the flat slots, --no-kv-pages says so explicitly; exact: only addresses change; RAGON_KV_PAGES=0/1
// overrides). --kv-pool-mb N gives the page pool N MB instead of
// --sessions x --max-context positions (P4b), so the checks run with sessions that map pages as they go, give back the
// ones past a new prompt and, in --prefix-check, evict idle sessions (least recently used first) when the pool is short,
// as ragon-serve's scheduler does.
//
// --graph-check decodes each prompt speculatively twice from the same prefill, without and with step graphs,
// greedily and then sampling (--temperature, default 1, top-k 20, top-p 0.95, seeded by --seed or 1), and
// checks that the tokens, the recurrent state (bit for bit) and the steps and accepted drafts (which would show a
// wrong drafter context KV) are the same both ways. --seed N seeds sampling.
//
// --spec decodes with the DFlash2 drafter. --spec-check (greedy) decodes each prompt plainly and then
// speculatively, and checks that the tokens are identical and that the recurrent state and the drafter's context
// after the speculative run equal those of a plain run over the same tokens, bit for bit.
//
// --draft-off-below X (BACKLOG 10; default 1.5, 0 off; RAGON_DRAFT_OFF_BELOW overrides it) sets the drafter-off guard
// (core/draft_guard.hpp) as ragon-serve's flag does. With --spec-check it also runs the guard's check: the speculative
// run three times, with the guard off, at X (a high X such as 100 trips it after its 16 warm-up steps; then 16 plain
// steps per probe), and at X with every probe switching the drafter back on and its next step off again; each run's
// tokens must equal plain decoding's and its state and drafter context a plain replay's (the drafter's context after
// a mix of plain and speculative steps is what it would have been). Use --max-tokens 512 or more so it trips.
//
// --prefix-check prefills each prompt after the one before it (so it resumes from a kept state: the end of
// the previous prompt or a prefix checkpoint), then again from scratch, and checks that the DeltaNet and
// conv state and the next token are the same both ways (M2: prefix checkpoints). With --draft it also
// decodes 64 tokens speculatively (greedy) after each prefill and checks that the tokens, the state, the
// steps and the accepted drafts are the same both ways (the drafter's context ring after a resume, M4 §1).
// --sessions N (M4 §1) keeps N resident sessions and binds each prompt to one as ragon-serve does (the
// session it extends, else a free or the least recently used slot), so each prompt resumes from its own
// session's state, from the shared checkpoint store, or from another session's KV.
//
// --chunk-check N (M4 §3) prefills each prompt from a cold engine twice, in full chunks and in chunks of N
// tokens (what ragon-serve runs while another request decodes), and checks that the state and the next token
// are the same both ways; with --draft also the drafter's context ring after the prefill (bit for bit) and the
// 64-token speculative continuation's tokens, state, steps and accepted drafts.
//
// --interleave-check (M4 §3, stage 2; needs --draft and --sessions 2) takes the prompts in pairs (A, B). Each
// is first decoded alone (cold engine, full chunks, --max-tokens speculatively); then A is prefilled and
// decodes while B prefills in chunks of --prefill-chunk-contended tokens (default 256) between A's steps, and
// the two then take steps in turn, as ragon-serve interleaves two requests. Per sequence the tokens, the state,
// the drafter's context, the steps and the accepted drafts must equal the run alone. Greedy and sampled
// (--temperature, default 1, seeded by --seed or 1), and with the pair swapped. Tree verification (BACKLOG 26)
// stays on, as in ragon-serve: this is the check of the sampled tree beside a prefill (--batch-check, of the tree in a batch).
// A drafter context that differs only in its last bits leaves greedy tokens alone but moves a sampled draft's q,
// so the sampled runs see it first; the drafter's context is compared for the greedy ones.
//
// --side-prefill N (with --interleave-check): B's chunks run on the engine's side stream (EngineOptions::side_prefill:
// a green context of N SMs, or a plain second stream for -1), each a full chunk launched while A goes on taking steps
// on the main stream, finished when its pass is done; the same comparison with the runs alone.
//
// --vision-prompts F (images, VISION §20.8): one chat-completions request body (JSON) per line, image parts as data:
// URLs or file paths, rendered and prepared as ragon-serve does; the prompts take part in every check above with their
// images (the vision tower on). --vision-check DIR compares the tower with reference dumps (bench/vision/reference.py):
// for each DIR/NAME.json ({"image": path, "grid": [t, h, w], ...}) the preprocessing against production's pixel values
// (NAME.pixel_values.bf16) and the tower's output from those pixel values against transformers' (NAME.hf_fp32.f32,
// NAME.hf_bf16.bf16, and production's NAME.sgl.bf16 when present): relative L2 error and the worst token's cosine
// against the FP32 tower, beside the reference's own. --vision-bench times the tower per image.
//
// --fanout-check (BATCHING P4d; needs --sessions 2 or more; with or without --draft) decodes one prompt per session
// plainly alone (step, sample), then all together in one pass per step (step_batch), and passes only if each sequence's
// tokens, every step's logits bit for bit, its state and its drafter context are the same. Greedy, sampled with a seed
// on the GPU's top K, and sampled with no top_k (the host sees every logit); the sessions in order and reversed.
// --batch-check (M4 §4, stage 3; needs --draft and --sessions 2) takes the prompts in pairs (A, B) and decodes
// each pair as a batch of two (Engine::speculate_batch: one verify pass over both sequences' rows), --max-tokens
// each, greedily and sampled (--temperature, default 1, seeded by --seed or 1). Four runs per pair and mode: both
// prefilled, then batched with the rows in the order [A, B] and in the order [B, A]; and one sequence prefilling
// in chunks of --prefill-chunk-contended tokens while the other decodes alone, then the two batched, each way
// round. Per sequence the tokens, the state, the drafter's context, the steps and the accepted drafts must equal the
// prompt decoded alone (speculatively, from a cold engine, under the same seed). With --graphs the runs alone and the
// batch both replay step graphs (the batch path's own since stage 4). Tree verification (BACKLOG 26) stays on: a
// batched sequence takes the tree it takes alone, step for step, and the check prints how many it took.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/engine.hpp"
#include "serve/admission.hpp"
#include "serve/api.hpp"
#include "serve/image_input.hpp"
#include "tokenizer/chat_template.hpp"
#include "tokenizer/tokenizer.hpp"

namespace {

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path);
  return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

float bf16f(uint16_t b) {
  uint32_t u = uint32_t(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// relative L2 error of a against the truth t, the worst row's cosine (rows of `width`), and the rows' cosines in *cos
void compare(const std::vector<float>& a, const std::vector<float>& t, int width, double& rel, double& worst_cos,
             std::vector<double>* cos = nullptr) {
  double num = 0, den = 0;
  worst_cos = 1;
  if (cos) cos->clear();
  for (size_t r = 0; r * width < t.size(); ++r) {
    double ab = 0, aa = 0, tt = 0;
    for (int c = 0; c < width; ++c) {
      const double x = a[r * width + c], y = t[r * width + c];
      num += (x - y) * (x - y);
      den += y * y;
      ab += x * y;
      aa += x * x;
      tt += y * y;
    }
    const double c = ab / std::sqrt(aa * tt + 1e-30);
    worst_cos = std::min(worst_cos, c);
    if (cos) cos->push_back(c);
  }
  rel = std::sqrt(num / (den + 1e-30));
}

// The rows' cosines summarised: the 1st percentile, the median, and the share below 0.99.
std::string cos_summary(std::vector<double> c) {
  if (c.empty()) return "";
  std::sort(c.begin(), c.end());
  const double p1 = c[c.size() / 100], med = c[c.size() / 2];
  const double below = double(std::lower_bound(c.begin(), c.end(), 0.99) - c.begin()) / c.size();
  char b[128];
  std::snprintf(b, sizeof b, "cosine p1 %.5f, median %.6f, %.2f%% of tokens below 0.99", p1, med, 100 * below);
  return b;
}

// --vision-check DIR (VISION §20.5 items 2 and 3).
int vision_check_run(ragon::Engine& engine, const std::string& dir, bool bench, bool dump) {
  ragon::VisionTower* tower = engine.vision();
  if (!tower) throw std::runtime_error("no vision tower");
  const int OUT = tower->config().out, PD = tower->config().patch_dim();
  int failures = 0;
  std::vector<std::string> names;
  for (const auto& e : std::filesystem::directory_iterator(dir))
    if (e.path().extension() == ".json") names.push_back(e.path().stem().string());
  std::sort(names.begin(), names.end());
  __nv_bfloat16* dout = nullptr;
  cudaMalloc(&dout, size_t(tower->max_patches() / 4) * OUT * 2);
  for (const std::string& name : names) {
    const std::string base = dir + "/" + name;
    const nlohmann::json meta = nlohmann::json::parse(read_file(base + ".json"));
    const int gh = meta["grid"][1], gw = meta["grid"][2], n = gh * gw;
    std::string image = meta["image"];
    if (!image.empty() && image[0] != '/') image = dir + "/" + image;
    std::printf("%s: %dx%d patches, %d tokens\n", name.c_str(), gh, gw, n / 4);
    // 1. Decoding, resizing, patches against production's.
    const ragon::serve::DecodedImage ours = ragon::serve::decode_image(read_file(image));
    if (meta.contains("decoded")) {
      const std::string ref = read_file(base + ".decoded.u8");
      long diff = 0;
      int worst = 0;
      if (ref.size() != ours.rgb.size()) {
        std::printf("  decode: size %zu against production's %zu FAIL\n", ours.rgb.size(), ref.size());
        ++failures;
      } else {
        for (size_t i = 0; i < ref.size(); ++i) {
          const int d = std::abs(int(uint8_t(ref[i])) - int(ours.rgb[i]));
          diff += d != 0;
          worst = std::max(worst, d);
        }
        std::printf("  decode: %ld of %zu values differ from production's (max %d)\n", diff, ref.size(), worst);
      }
      if (meta.contains("resized")) {
        const std::string rin = ref;
        const int h = meta["decoded"][0], w = meta["decoded"][1], nh = meta["resized"][0], nw = meta["resized"][1];
        const std::vector<uint8_t> mine = ragon::serve::resize_bicubic_aa(reinterpret_cast<const uint8_t*>(rin.data()), h, w, nh, nw);
        const std::string rref = read_file(base + ".resized.u8");
        long rd = 0;
        int rw = 0;
        for (size_t i = 0; i < mine.size() && i < rref.size(); ++i) {
          const int d = std::abs(int(uint8_t(rref[i])) - int(mine[i]));
          rd += d != 0;
          rw = std::max(rw, d);
        }
        const bool ok = rref.size() == mine.size() && rw <= 1 && rd * 1000 <= long(mine.size());
        failures += !ok;
        std::printf("  resize (from production's pixels): %ld of %zu values differ (max %d) %s\n", rd, mine.size(), rw,
                    ok ? "ok" : "FAIL");
      }
    }
    const ragon::serve::PreparedImage prep = ragon::serve::prepare_image(ours);
    const std::string pv = read_file(base + ".pixel_values.bf16");
    std::vector<uint16_t> ref_patches(pv.size() / 2);
    std::memcpy(ref_patches.data(), pv.data(), pv.size());
    if (prep.grid_h != gh || prep.grid_w != gw) {
      std::printf("  grid %dx%d against production's %dx%d FAIL\n", prep.grid_h, prep.grid_w, gh, gw);
      ++failures;
    } else {
      long pd = 0;
      float pw = 0;
      for (size_t i = 0; i < ref_patches.size(); ++i)
        if (ref_patches[i] != prep.patches[i]) {
          ++pd;
          pw = std::max(pw, std::abs(bf16f(ref_patches[i]) - bf16f(prep.patches[i])));
        }
      std::printf("  patches: %ld of %zu BF16 values differ from production's pixel_values (max %.4f)\n", pd,
                  ref_patches.size(), pw);
    }
    // 2. The tower on production's pixel values against the references.
    if (ref_patches.size() != size_t(n) * PD) throw std::runtime_error(name + ": pixel_values has the wrong size");
    tower->forward(ref_patches.data(), gh, gw, dout, nullptr);
    cudaDeviceSynchronize();
    std::vector<uint16_t> ob(size_t(n / 4) * OUT);
    cudaMemcpy(ob.data(), dout, ob.size() * 2, cudaMemcpyDeviceToHost);
    std::vector<float> eng(ob.size());
    for (size_t i = 0; i < ob.size(); ++i) eng[i] = bf16f(ob[i]);
    auto load_f32 = [&](const std::string& path) {
      const std::string b = read_file(path);
      std::vector<float> v(b.size() / 4);
      std::memcpy(v.data(), b.data(), b.size());
      return v;
    };
    auto load_bf16 = [&](const std::string& path) {
      const std::string b = read_file(path);
      std::vector<float> v(b.size() / 2);
      for (size_t i = 0; i < v.size(); ++i) {
        uint16_t u;
        std::memcpy(&u, b.data() + 2 * i, 2);
        v[i] = bf16f(u);
      }
      return v;
    };
    const std::vector<float> truth = load_f32(base + ".hf_fp32.f32");
    if (truth.size() != eng.size()) throw std::runtime_error(name + ": hf_fp32 has the wrong size");
    if (dump) {
      std::ofstream f(base + ".engine.bf16", std::ios::binary);
      f.write(reinterpret_cast<const char*>(ob.data()), ob.size() * 2);
    }
    double er, ec;
    std::vector<double> ecos;
    compare(eng, truth, OUT, er, ec, &ecos);
    double floor_rel = 0, floor_cos = 1;
    std::string floor_name = "transformers BF16";
    {
      const std::vector<float> hb = load_bf16(base + ".hf_bf16.bf16");
      double r, c;
      std::vector<double> cs;
      compare(hb, truth, OUT, r, c, &cs);
      std::printf("  transformers BF16 against FP32: rel L2 %.3e, worst cosine %.6f; %s\n", r, c, cos_summary(cs).c_str());
      floor_rel = r;
      floor_cos = c;
    }
    if (std::filesystem::exists(base + ".hfprodpos.bf16")) {
      const std::vector<float> pb = load_bf16(base + ".hfprodpos.bf16");
      double r, c;
      std::vector<double> cs;
      compare(pb, truth, OUT, r, c, &cs);
      std::printf("  transformers BF16 with production's position embedding against FP32: rel L2 %.3e, worst cosine %.6f; %s\n",
                  r, c, cos_summary(cs).c_str());
      floor_rel = std::max(floor_rel, r);
      floor_cos = std::min(floor_cos, c);
    }
    if (std::filesystem::exists(base + ".sgl.bf16")) {
      const std::vector<float> sb = load_bf16(base + ".sgl.bf16");
      double r, c;
      std::vector<double> cs;
      compare(sb, truth, OUT, r, c, &cs);
      std::printf("  production (SGLang) against FP32: rel L2 %.3e, worst cosine %.6f; %s\n", r, c, cos_summary(cs).c_str());
      floor_rel = std::max(floor_rel, r);
      floor_cos = std::min(floor_cos, c);
      floor_name = "the references";
    }
    const bool ok = er <= floor_rel * 1.10 && ec >= floor_cos - 1e-4;
    failures += !ok;
    std::printf("  engine against FP32: rel L2 %.3e, worst cosine %.6f; %s (%s: %.3e, %.6f) %s\n", er, ec,
                cos_summary(ecos).c_str(), floor_name.c_str(), floor_rel, floor_cos, ok ? "ok" : "FAIL");
    if (bench) {
      cudaEvent_t a, b;
      cudaEventCreate(&a);
      cudaEventCreate(&b);
      std::vector<float> ms;
      for (int k = 0; k < 5; ++k) {
        cudaEventRecord(a);
        tower->forward(ref_patches.data(), gh, gw, dout, nullptr);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float t = 0;
        cudaEventElapsedTime(&t, a, b);
        ms.push_back(t);
      }
      std::sort(ms.begin(), ms.end());
      std::printf("  tower: %.1f ms (median of 5), %.0f patches/s\n", ms[2], n / (ms[2] / 1e3));
      cudaEventDestroy(a);
      cudaEventDestroy(b);
    }
  }
  cudaFree(dout);
  std::printf(failures ? "VISION CHECK FAILED (%d)\n" : "vision check passed\n", failures);
  return failures ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model, text, chat, ids_arg, prompts_file;
  int max_tokens = 64, max_context = 32768;
  float temperature = 0.f;
  bool ids_out = false, prompt_ids_out = false, spec = false, spec_check = false, prefix_check = false;
  bool graphs = false, pdl = false, attn_bulk = false, graph_check = false, prefill_bf16 = false;
  bool kv_pages = false;  // the flat slots by default (P4d kept them); --kv-pages the page pool
  long kv_pool_mb = -1;
  uint64_t seed = 0;
  int attn_prefetch = 2;
  std::string draft, lora;
  int block = 12, lookup = 1, lookup_min = 8, sessions = 1, chunk_check = 0, contended = 256, side_prefill = 0;
  bool fanout_check = false;
  bool fork_check = false;
  bool interleave_check = false, batch_check = false, batch_bench = false, draft_bf16 = false;
  double draft_off_below = -1;  // the drafter-off guard (BACKLOG 10): the engine's default unless given
  std::string draft_head = "sparse", sparse_head_dir;  // the drafter's head (BACKLOG 19)
  int sparse_head_dynamic = 16384;
  std::string vision_prompts, vision_check, force_file, greedy_out;  // images (VISION §20.8)
  bool vision_bench = false, vision_dump = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::cerr << a << " needs a value\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--model") model = next();
    else if (a == "--text") text = next();
    else if (a == "--chat") chat = next();
    else if (a == "--ids") ids_arg = next();
    else if (a == "--prompts-file") prompts_file = next();
    else if (a == "--max-tokens") max_tokens = std::stoi(next());
    else if (a == "--max-context") max_context = std::stoi(next());
    else if (a == "--temperature") temperature = std::stof(next());
    else if (a == "--ids-out") ids_out = true;
    else if (a == "--prompt-ids-out") prompt_ids_out = true;
    else if (a == "--draft") draft = next();
    else if (a == "--lora") lora = next();
    else if (a == "--block") block = std::stoi(next());
    else if (a == "--spec") spec = true;
    else if (a == "--lookup") lookup = std::stoi(next());
    else if (a == "--lookup-min") lookup_min = std::stoi(next());
    else if (a == "--draft-off-below") draft_off_below = std::stod(next());
    else if (a == "--spec-check") spec_check = true;
    else if (a == "--prefix-check") prefix_check = true;
    else if (a == "--graphs") graphs = true;
    else if (a == "--graph-check") graph_check = true;
    else if (a == "--seed") seed = std::stoull(next());
    else if (a == "--pdl") pdl = true;
    else if (a == "--attn-bulk") attn_bulk = true;
    else if (a == "--attn-prefetch") attn_prefetch = std::stoi(next());
    else if (a == "--prefill-bf16") prefill_bf16 = true;
    else if (a == "--kv-pages") kv_pages = true;
    else if (a == "--no-kv-pages") kv_pages = false;
    else if (a == "--kv-pool-mb") kv_pool_mb = std::stol(next());
    else if (a == "--draft-bf16") draft_bf16 = true;
    else if (a == "--draft-head") draft_head = next();  // sparse (default) or full (BACKLOG 19)
    else if (a == "--sparse-head-dir") sparse_head_dir = next();
    else if (a == "--sparse-head-dynamic") sparse_head_dynamic = std::stoi(next());
    else if (a == "--sessions") sessions = std::stoi(next());
    else if (a == "--chunk-check") chunk_check = std::stoi(next());
    else if (a == "--interleave-check") interleave_check = true;
    else if (a == "--batch-check") batch_check = true;
    else if (a == "--fanout-check") fanout_check = true;
    else if (a == "--fork-check") fork_check = true;
    else if (a == "--batch-bench") batch_check = batch_bench = true;
    else if (a == "--prefill-chunk-contended") contended = std::stoi(next());
    else if (a == "--side-prefill") side_prefill = std::stoi(next());
    else if (a == "--vision-prompts") vision_prompts = next();
    else if (a == "--vision-check") vision_check = next();
    else if (a == "--vision-bench") vision_bench = true;
    else if (a == "--vision-dump") vision_dump = true;
    else if (a == "--force-file") force_file = next();
    else if (a == "--greedy-out") greedy_out = next();
    else {
      std::cerr << "unknown argument " << a << "\n";
      return 2;
    }
  }
  if (model.empty()) {
    std::cerr << "usage: ragon-run --model DIR (--text T | --chat T | --ids 1,2,3) [--max-tokens N]\n";
    return 2;
  }
  if (const char* e = std::getenv("RAGON_KV_PAGES")) kv_pages = std::atoi(e) != 0;  // as the engine reads it
  if (kv_pool_mb >= 0 && !kv_pages) {
    std::cerr << "--kv-pool-mb needs --kv-pages (or RAGON_KV_PAGES=1)\n";
    return 2;
  }
  try {
    // Image prompts are tokenized as ragon-serve tokenizes (production's pre-tokenizer), so their ids are production's.
    ragon::Tokenizer tok(model + "/tokenizer.json", vision_prompts.empty() ? "" : ragon::kProductionPretokenizer);
    std::vector<std::vector<int>> prompts;
    ragon::serve::ImageRegistry image_registry;
    std::vector<ragon::PromptImage> all_images;  // every image of every prompt (begin_prefill finds each by its id)
    if (!prompts_file.empty()) {  // one JSON string per line, each a raw prompt
      std::ifstream f(prompts_file);
      std::string line;
      while (std::getline(f, line))
        if (!line.empty()) prompts.push_back(tok.encode(nlohmann::json::parse(line).get<std::string>()));
    } else if (!ids_arg.empty()) {
      std::vector<int> prompt;
      std::stringstream ss(ids_arg);
      std::string part;
      while (std::getline(ss, part, ',')) prompt.push_back(std::stoi(part));
      prompts.push_back(prompt);
    } else if (!vision_prompts.empty()) {
      // Each line a chat-completions body; file paths in image parts become data: URLs (the server takes only those).
      std::ifstream f(vision_prompts);
      std::string line;
      while (std::getline(f, line)) {
        if (line.empty()) continue;
        // ordered_json: the tools' keys keep the request's order, as ragon-serve parses a body (the template renders them).
        nlohmann::ordered_json body = nlohmann::ordered_json::parse(line);
        for (auto& m : body["messages"])
          if (m.contains("content") && m["content"].is_array())
            for (auto& part : m["content"])
              if (part.is_object() && part.contains("image_url")) {
                auto& u = part["image_url"];
                std::string url = u.is_string() ? u.get<std::string>() : u["url"].get<std::string>();
                if (url.rfind("data:", 0) != 0) {
                  // A path relative to the prompts file's directory.
                  if (!url.empty() && url[0] != '/')
                    url = std::filesystem::absolute(vision_prompts).parent_path().string() + "/" + url;
                  std::ifstream img(url, std::ios::binary);
                  if (!img) throw std::runtime_error("cannot read " + url);
                  std::string bytes((std::istreambuf_iterator<char>(img)), std::istreambuf_iterator<char>());
                  static const char* b64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
                  std::string enc;
                  size_t i = 0;
                  for (; i + 2 < bytes.size(); i += 3) {
                    const uint32_t v = (uint8_t(bytes[i]) << 16) | (uint8_t(bytes[i + 1]) << 8) | uint8_t(bytes[i + 2]);
                    for (int k = 18; k >= 0; k -= 6) enc += b64[(v >> k) & 63];
                  }
                  if (i < bytes.size()) {
                    uint32_t v = uint8_t(bytes[i]) << 16;
                    if (i + 1 < bytes.size()) v |= uint8_t(bytes[i + 1]) << 8;
                    enc += b64[(v >> 18) & 63];
                    enc += b64[(v >> 12) & 63];
                    enc += i + 1 < bytes.size() ? b64[(v >> 6) & 63] : '=';
                    enc += '=';
                  }
                  url = "data:image;base64," + enc;
                }
                if (u.is_string()) u = url;
                else u["url"] = url;
              }
        ragon::serve::Request req = ragon::serve::parse_request(body, true, true);
        std::vector<int> ids = tok.encode(req.prompt_text);
        ragon::serve::attach_images(ids, tok.token_id("<|image_pad|>"), req.image_urls, image_registry,
                                   ragon::serve::ProcessorConfig{}, all_images);
        prompts.push_back(ids);
      }
    } else if (!chat.empty()) {
      nlohmann::ordered_json messages = nlohmann::ordered_json::array();
      messages.push_back({{"role", "user"}, {"content", chat}});
      ragon::ChatOptions o;
      o.enable_thinking = false;
      prompts.push_back(tok.encode(ragon::render_chat(messages, nlohmann::ordered_json(), o)));
    } else {
      prompts.push_back(tok.encode(text));
    }
    auto t0 = std::chrono::steady_clock::now();
    ragon::EngineOptions opts;
    opts.max_context = max_context;
    opts.draft_dir = draft;
    opts.draft_block = block;
    opts.lookup_mode = lookup;
    opts.lookup_min_match = lookup_min;
    if (draft_off_below >= 0) opts.draft_off_below = draft_off_below;
    opts.boundary_token = tok.token_id("<|im_end|>");
    opts.step_graphs = graphs;
    opts.pdl = pdl;
    opts.attention_bulk = attn_bulk;
    opts.attention_prefetch = attn_prefetch;
    opts.prefill_bf16 = prefill_bf16;
    opts.kv_pages = kv_pages;
    if (kv_pool_mb >= 0) {  // P4b: pages of the model's page size (64 positions of every full-attention layer's K and V)
      const size_t page = ragon::serve::kv_page_bytes(ragon::ModelConfig::from_file(model + "/config.json"));
      opts.kv_pool_pages = static_cast<int>(std::max<size_t>(1, (size_t(kv_pool_mb) << 20) / page));
    }
    opts.draft_nvfp4 = !draft_bf16;
    opts.draft_head = draft_head;
    opts.sparse_head_dir = sparse_head_dir;
    opts.sparse_head_dynamic = sparse_head_dynamic;
    opts.sessions = sessions;
    opts.side_prefill = side_prefill;
    opts.vision = !all_images.empty() || !vision_check.empty();
    if (!lora.empty()) opts.lora_slots = 1;
    ragon::Engine engine(model, opts);
    // --lora: the adapter in slot 0, and every session activated on it (activate below, for the checks' sessions).
    uint64_t lora_sc = 0;
    int lora_slot = -1;
    if (!lora.empty()) {
      const std::filesystem::path p = std::filesystem::path(lora).lexically_normal();
      ragon::LoraLibrary lib(p.parent_path().string(), 1, engine.lora_shape(), engine.lora_draft_shape());
      const auto adapter = lib.load(p.filename().string(), p.string());
      engine.lora_upload(0, *adapter);
      lora_sc = ragon::lora_scope(0, adapter.get());
      lora_slot = 0;
      std::fprintf(stderr, "adapter %s (%zu bytes%s)\n", adapter->name.c_str(), adapter->bytes(),
                   adapter->has_drafter ? ", with a drafter part" : "");
    }
    auto activate = [&](int i, const std::string& key, bool fresh) {
      engine.activate_session(i, key, fresh, lora_sc, lora_slot);
    };
    if (lora_slot >= 0) activate(0, "", false);
    const std::vector<ragon::PromptImage>* IMG = all_images.empty() ? nullptr : &all_images;
    if (!vision_check.empty()) return vision_check_run(engine, vision_check, vision_bench, vision_dump);
    auto t1 = std::chrono::steady_clock::now();
    std::fprintf(stderr, "loaded %.1f GB of weights in %.1f s\n", engine.model().device_bytes() / 1e9,
                 std::chrono::duration<double>(t1 - t0).count());
    const int im_end = tok.token_id("<|im_end|>");
    // RAGON_PROBE_SKIP's runs (src/core/launch.cuh) produce garbage tokens; they decode to --max-tokens regardless.
    const bool no_end = std::getenv("RAGON_PROBE_SKIP") != nullptr;
    auto is_end = [&](int t) { return !no_end && (t == engine.config().eos_token || t == im_end); };
    // The agreement study's engine side for these prompts (bench/vision/agree.py), as bench/agreement/ragon_force does
    // for text: plain greedy decoding, and teacher-forced top-20 log-probs over production's continuation.
    const int pad_id = tok.token_id("<|image_pad|>");
    auto as_production = [&](std::vector<int> ids) {  // an image's ids as production's prompt_token_ids shows them
      for (int& t : ids)
        if (t >= ragon::kImageTokenBase) t = pad_id;
      return ids;
    };
    auto topk_json = [&](const std::vector<float>& logits, int K, const std::vector<int>& extra) {
      const int V = static_cast<int>(logits.size());
      double mx = -1e30;
      for (float v : logits) mx = std::max(mx, double(v));
      double sum = 0;
      for (float v : logits) sum += std::exp(double(v) - mx);
      const double lse = mx + std::log(sum);
      std::vector<int> idx(V);
      for (int i = 0; i < V; ++i) idx[i] = i;
      std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                        [&](int a, int b) { return logits[a] > logits[b] || (logits[a] == logits[b] && a < b); });
      nlohmann::json t = nlohmann::json::array();
      for (int i = 0; i < K; ++i) t.push_back({idx[i], double(logits[idx[i]]) - lse});
      nlohmann::json e = nlohmann::json::object();
      for (int id : extra)
        if (id >= 0 && id < V) e[std::to_string(id)] = double(logits[id]) - lse;
      return nlohmann::json{{"top", t}, {"lp", e}};
    };
    if (!greedy_out.empty()) {
      std::ofstream out(greedy_out);
      for (const std::vector<int>& prompt : prompts) {
        engine.reset();
        ragon::EngineStats st;
        const std::vector<float>* logits = &engine.prefill(prompt, &st, false, IMG);
        std::vector<int> ids;
        for (int i = 0; i < max_tokens; ++i) {
          const int t = static_cast<int>(std::max_element(logits->begin(), logits->end()) - logits->begin());
          ids.push_back(t);
          if (is_end(t)) break;
          logits = &engine.step(t);
        }
        out << nlohmann::json{{"ids", ids}, {"text", tok.decode(ids)}, {"prompt_ids", as_production(prompt)},
                              {"prefill_s", st.prefill_seconds}, {"vision_s", st.vision_seconds},
                              {"image_tokens", st.image_tokens}}
                   .dump()
            << "\n";
      }
      return 0;
    }
    if (!force_file.empty()) {
      std::ifstream in(force_file);
      std::string line;
      while (std::getline(in, line)) {
        if (line.empty()) continue;
        const nlohmann::json c = nlohmann::json::parse(line);
        const std::vector<int>& prompt = prompts.at(c["prompt"].get<size_t>());
        const std::vector<int> cont = c["cont"];
        const nlohmann::json want = c.value("want", nlohmann::json::array());
        engine.reset();
        const std::vector<float>* logits = &engine.prefill(prompt, nullptr, false, IMG);
        nlohmann::json pos = nlohmann::json::array();
        for (size_t j = 0; j < cont.size(); ++j) {
          std::vector<int> extra = {cont[j]};
          if (j < want.size())
            for (const auto& w : want[j]) extra.push_back(w.get<int>());
          pos.push_back(topk_json(*logits, 20, extra));
          if (j + 1 < cont.size()) logits = &engine.step(cont[j]);
        }
        std::cout << nlohmann::json{{"pos", pos}}.dump() << "\n" << std::flush;
      }
      return 0;
    }

    // Generates from `prompt` (state reset first): plainly or speculatively. Prints text unless `quiet`.
    auto generate = [&](const std::vector<int>& prompt, const ragon::SamplingParams& sp, bool use_spec, bool quiet,
                        double* decode_s) {
      ragon::EngineStats stats;
      engine.reset();
      std::vector<float> logits = engine.prefill(prompt, &stats, false, IMG);
      if (!quiet)
        std::fprintf(stderr, "prefill: %d tokens in %.3f s (%.0f tok/s); %d image tokens, the tower %.3f s on %d images\n",
                     stats.prefill_tokens, stats.prefill_seconds, stats.prefill_tokens / stats.prefill_seconds,
                     stats.image_tokens, stats.vision_seconds, stats.images_encoded);
      ragon::StreamDecoder dec(tok);
      std::vector<int> out;
      auto t2 = std::chrono::steady_clock::now();
      std::vector<int> pending = {engine.sample(logits, sp, engine.history())};
      bool done = false;
      while (!done) {
        for (int t : pending) {
          out.push_back(t);
          if (is_end(t) || static_cast<int>(out.size()) >= max_tokens) {
            done = true;
            break;
          }
          if (!quiet && !ids_out) std::cout << dec.push(t) << std::flush;
        }
        if (done) break;
        if (use_spec) {
          pending = engine.speculate(pending.back(), sp);
        } else {
          logits = engine.step(pending.back());
          pending = {engine.sample(logits, sp, engine.history())};
        }
      }
      *decode_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t2).count();
      if (!quiet && !ids_out) std::cout << dec.flush() << "\n";
      return out;
    };

    int failures = 0;
    // A sequence decoded speculatively in session `session`, one step at a time (--chunk-check, --interleave-check).
    struct Seq {
      int session = 0;
      ragon::SamplingParams sp;
      std::vector<int> out, pending;
      bool done = false;
    };
    // Emits the pending tokens; unless that ends the sequence, runs one speculative step in its session.
    auto advance = [&](Seq& q, int limit) {
      for (int t : q.pending) {
        q.out.push_back(t);
        if (is_end(t) || static_cast<int>(q.out.size()) >= limit) {
          q.done = true;
          return;
        }
      }
      engine.select_session(q.session);
      q.pending = engine.speculate(q.pending.back(), q.sp);
    };
    struct Result {
      std::vector<int> tokens;
      uint64_t hash = 0, draft = 0;  // the state, and the drafter's context (Engine::draft_hash)
      long steps = 0, accepted = 0;
      bool operator==(const Result& o) const {
        return tokens == o.tokens && hash == o.hash && draft == o.draft && steps == o.steps && accepted == o.accepted;
      }
    };
    auto result_of = [&](const Seq& q) {
      Result r;
      r.tokens = q.out;
      engine.select_session(q.session);
      r.hash = engine.state_hash();
      r.draft = engine.draft_hash();
      r.steps = engine.session_spec_stats(q.session).steps;
      r.accepted = engine.session_spec_stats(q.session).accepted;
      return r;
    };
    if (chunk_check > 0) {
      auto argmax = [](const std::vector<float>& l) { return int(std::max_element(l.begin(), l.end()) - l.begin()); };
      for (size_t i = 0; i < prompts.size(); ++i) {
        Result r[2];
        int next[2];
        uint64_t ctx[2];  // the drafter's context after the prefill
        ragon::EngineStats st[2];
        for (int k = 0; k < 2; ++k) {
          engine.reset_all();
          activate(0, "", true);
          ragon::Engine::PrefillCursor cur = engine.begin_prefill(prompts[i], false, IMG);
          while (!engine.prefill_chunk(cur, k ? chunk_check : 0)) {
          }
          st[k] = cur.stats;
          const std::vector<float> logits = engine.logits();
          next[k] = argmax(logits);
          ctx[k] = engine.draft_hash();
          Seq q;
          q.sp.temperature = 0.f;
          if (engine.has_drafter()) {
            q.pending = {engine.sample(logits, q.sp, engine.history())};
            while (!q.done) advance(q, 64);
          }
          r[k] = result_of(q);
        }
        const bool same = r[0] == r[1] && next[0] == next[1] && ctx[0] == ctx[1];
        failures += !same;
        std::printf("prompt %zu: %zu tokens, %d chunks (%.2f s) against %d chunks of %d (%.2f s): state %s, next token %s, "
                    "drafter context %s, continuation %zu tokens (%s) %ld steps %ld accepted against %ld steps %ld "
                    "accepted: %s\n",
                    i, prompts[i].size(), st[0].chunks, st[0].compute_seconds, st[1].chunks, chunk_check,
                    st[1].compute_seconds, r[0].hash == r[1].hash ? "identical" : "DIFFERS",
                    next[0] == next[1] ? "identical" : "DIFFERS", ctx[0] == ctx[1] ? "identical" : "DIFFERS",
                    r[1].tokens.size(), r[0].tokens == r[1].tokens ? "identical" : "DIFFER", r[0].steps,
                    r[0].accepted, r[1].steps, r[1].accepted, same ? "identical" : "DIFFERS");
      }
      std::printf(failures ? "CHUNK CHECK FAILED\n" : "chunk check passed\n");
      return failures ? 1 : 0;
    }
    if (interleave_check) {
      if (!engine.has_drafter() || engine.sessions() < 2 || prompts.size() < 2)
        throw std::runtime_error("--interleave-check needs --draft, --sessions 2 and at least two prompts");
      for (size_t pi = 0; pi + 1 < prompts.size(); pi += 2) {
        for (int swapped = 0; swapped < 2; ++swapped) {
          const std::vector<int>& A = prompts[pi + swapped];
          const std::vector<int>& B = prompts[pi + 1 - swapped];
          for (int sampled = 0; sampled < 2; ++sampled) {
            ragon::SamplingParams sp;
            sp.temperature = sampled ? (temperature > 0.f ? temperature : 1.f) : 0.f;
            sp.seed = seed ? seed : 1;
            // Each alone, from a cold engine in session 0.
            Result alone[2];
            for (int k = 0; k < 2; ++k) {
              engine.reset_all();
              activate(0, "", true);
              engine.prefill(k ? B : A, nullptr, false, IMG);
              Seq q;
              q.sp = sp;
              q.pending = {engine.sample(engine.logits(), sp, engine.history())};
              while (!q.done) advance(q, max_tokens);
              alone[k] = result_of(q);
            }
            // Interleaved: A in session 0 prefills and takes two steps alone; then B (session 1) prefills in
            // contended chunks, one between each of A's steps, and the two take steps in turn.
            engine.reset_all();
            Seq a, b;
            a.session = 0;
            b.session = 1;
            a.sp = b.sp = sp;
            activate(0, "", true);
            engine.prefill(A, nullptr, false, IMG);
            a.pending = {engine.sample(engine.logits(), sp, engine.history())};
            for (int k = 0; k < 2 && !a.done; ++k) advance(a, max_tokens);
            activate(1, "", true);
            ragon::Engine::PrefillCursor cur = engine.begin_prefill(B, false, IMG);
            int chunks_beside = 0;
            while (!a.done || !b.done) {
              if (!cur.done() && engine.side_enabled() && (engine.side_busy() || !a.done)) {
                // The side path: a full chunk queued beside A's steps; finished once its pass is done (or when A
                // has nothing left to run beside it).
                if (!engine.side_busy()) {
                  engine.prefill_chunk_launch(cur, 0);
                  chunks_beside += !a.done;
                } else if (a.done || engine.side_ready()) {
                  if (engine.prefill_chunk_finish()) b.pending = {engine.sample(engine.logits(), sp, engine.history())};
                }
              } else if (!cur.done()) {
                if (engine.prefill_chunk(cur, a.done ? 0 : contended)) {
                  b.pending = {engine.sample(engine.logits(), sp, engine.history())};
                }
                chunks_beside += !a.done;
              }
              if (!a.done) advance(a, max_tokens);
              if (cur.done() && !b.done) advance(b, max_tokens);
            }
            const Result ra = result_of(a), rb = result_of(b);
            const bool same_a = ra == alone[0], same_b = rb == alone[1];
            failures += !same_a + !same_b;
            std::printf("pair %zu%s, %s: A %zu prompt tokens, %zu out, %ld steps %ld accepted: %s | B %zu prompt tokens "
                        "(%d chunks, %d beside A's steps), %zu out, %ld steps %ld accepted: %s\n",
                        pi / 2, swapped ? " swapped" : "", sampled ? "sampled" : "greedy", A.size(), ra.tokens.size(),
                        ra.steps, ra.accepted, same_a ? "identical" : "DIFFERS", B.size(), cur.stats.chunks, chunks_beside,
                        rb.tokens.size(), rb.steps, rb.accepted, same_b ? "identical" : "DIFFERS");
            if (!same_a)
              std::printf("  A alone: %zu out, %ld steps %ld accepted, state %s, drafter context %s\n",
                          alone[0].tokens.size(), alone[0].steps, alone[0].accepted,
                          alone[0].hash == ra.hash ? "identical" : "differs",
                          alone[0].draft == ra.draft ? "identical" : "differs");
            if (!same_b)
              std::printf("  B alone: %zu out, %ld steps %ld accepted, state %s, drafter context %s\n",
                          alone[1].tokens.size(), alone[1].steps, alone[1].accepted,
                          alone[1].hash == rb.hash ? "identical" : "differs",
                          alone[1].draft == rb.draft ? "identical" : "differs");
          }
        }
      }
      std::printf(failures ? "INTERLEAVE CHECK FAILED\n" : "interleave check passed\n");
      return failures ? 1 : 0;
    }
    if (batch_check) {
      if (!engine.has_drafter() || engine.sessions() < 2 || prompts.size() < 2)
        throw std::runtime_error("--batch-check needs --draft, --sessions 2 and at least two prompts");
      // Emits q's pending tokens; true when q is done.
      auto emit = [&](Seq& q, int limit) {
        for (int t : q.pending) {
          q.out.push_back(t);
          if (is_end(t) || static_cast<int>(q.out.size()) >= limit) return q.done = true;
        }
        return false;
      };
      // Decodes the sequences in `order` together: every step, those still going and able to join take one
      // batched step (rows in this order); one that cannot join takes its own one-row step.
      auto run_batch = [&](std::vector<Seq*> order) {
        while (true) {
          std::vector<Seq*> live;
          for (Seq* q : order)
            if (!q->done && !emit(*q, max_tokens)) live.push_back(q);
          if (live.empty()) return;
          std::vector<ragon::Engine::SeqStep> steps;
          std::vector<Seq*> in;
          for (Seq* q : live) {
            if (engine.can_batch(q->session, q->sp)) {
              ragon::Engine::SeqStep st;
              st.session = q->session;
              st.anchor = q->pending.back();
              st.params = &q->sp;
              steps.push_back(std::move(st));
              in.push_back(q);
            } else {
              engine.select_session(q->session);
              q->pending = engine.speculate(q->pending.back(), q->sp);
            }
          }
          engine.speculate_batch(steps);
          for (size_t i = 0; i < steps.size(); ++i) {
            if (steps[i].error) std::rethrow_exception(steps[i].error);
            in[i]->pending = steps[i].out;
          }
        }
      };
      auto start = [&](Seq& q, int session, const std::vector<int>& prompt) {
        q.session = session;
        activate(session, "", true);
        engine.prefill(prompt, nullptr, false, IMG);
        q.pending = {engine.sample(engine.logits(), q.sp, engine.history())};
      };
      for (size_t pi = 0; pi + 1 < prompts.size(); pi += 2) {
        const std::vector<int>* P[2] = {&prompts[pi], &prompts[pi + 1]};
        if (batch_bench) {  // timing only: the pair greedy, rows [A, B], every step batched while both run
          engine.reset_all();
          Seq q[2];
          q[0].sp.temperature = q[1].sp.temperature = 0.f;
          q[0].sp.seed = q[1].sp.seed = 1;
          start(q[0], 0, *P[0]);
          start(q[1], 1, *P[1]);
          run_batch({&q[0], &q[1]});
          const ragon::SpecStats& s = engine.session_spec_stats(q[0].session);
          const ragon::SpecStats& o = engine.session_spec_stats(q[1].session);
          std::printf("batch bench pair %zu: A %zu out, %ld steps (%ld batched, %ld trees); B %zu out, %ld steps (%ld batched, "
                      "%ld trees); A's step %.1f ms; B's %.1f ms (draft %.1f, verify %.1f, commit %.1f)\n",
                      pi / 2, q[0].out.size(), s.steps, s.batched_steps, s.tree_steps, q[1].out.size(), o.steps,
                      o.batched_steps, o.tree_steps,
                      1e3 * (s.draft_seconds + s.verify_seconds + s.commit_seconds) / s.steps,
                      1e3 * (o.draft_seconds + o.verify_seconds + o.commit_seconds) / o.steps,
                      1e3 * o.draft_seconds / o.steps, 1e3 * o.verify_seconds / o.steps, 1e3 * o.commit_seconds / o.steps);
          continue;
        }
        for (int sampled = 0; sampled < 2; ++sampled) {
          ragon::SamplingParams sp;
          sp.temperature = sampled ? (temperature > 0.f ? temperature : 1.f) : 0.f;
          sp.seed = seed ? seed : 1;
          Result alone[2];
          for (int k = 0; k < 2; ++k) {
            engine.reset_all();
            Seq q;
            q.sp = sp;
            start(q, 0, *P[k]);
            while (!q.done) advance(q, max_tokens);
            alone[k] = result_of(q);
          }
          // Runs: 0 and 1 both prefilled, rows [A, B] and [B, A]; 2 and 3 the first decoding while the second
          // prefills (A first, then B first), then rows [first, second].
          const char* names[4] = {"together, rows [A, B]", "together, rows [B, A]", "B prefilling beside A",
                                  "A prefilling beside B"};
          for (int run = 0; run < 4; ++run) {
            engine.reset_all();
            Seq q[2];
            q[0].sp = q[1].sp = sp;
            int chunks_beside = 0, steps_beside = 0;
            if (run < 2) {
              start(q[0], 0, *P[0]);
              start(q[1], 1, *P[1]);
              run_batch(run == 0 ? std::vector<Seq*>{&q[0], &q[1]} : std::vector<Seq*>{&q[1], &q[0]});
            } else {
              const int f = run == 2 ? 0 : 1, o = 1 - f;  // first (decoding), other (prefilling)
              start(q[f], 0, *P[f]);
              for (int k = 0; k < 2 && !q[f].done; ++k) advance(q[f], max_tokens);
              q[o].session = 1;
              activate(1, "", true);
              ragon::Engine::PrefillCursor cur = engine.begin_prefill(*P[o], false, IMG);
              while (!cur.done()) {
                if (engine.prefill_chunk(cur, q[f].done ? 0 : contended))
                  q[o].pending = {engine.sample(engine.logits(), q[o].sp, engine.history())};
                chunks_beside += !q[f].done;
                if (!q[f].done) {
                  advance(q[f], max_tokens);
                  ++steps_beside;
                }
              }
              run_batch({&q[f], &q[o]});
            }
            Result r[2];
            long batched[2], trees[2];
            for (int k = 0; k < 2; ++k) {
              r[k] = result_of(q[k]);
              batched[k] = engine.session_spec_stats(q[k].session).batched_steps;
              trees[k] = engine.session_spec_stats(q[k].session).tree_steps;
            }
            const bool same[2] = {r[0] == alone[0], r[1] == alone[1]};
            failures += !same[0] + !same[1];
            std::printf("pair %zu, %s, %s%s: A %zu prompt tokens, %zu out, %ld steps (%ld batched, %ld trees) %ld accepted: "
                        "%s | B %zu prompt tokens, %zu out, %ld steps (%ld batched, %ld trees) %ld accepted: %s\n",
                        pi / 2, sampled ? "sampled" : "greedy", names[run],
                        run >= 2 ? (" (" + std::to_string(chunks_beside) + " chunks beside " +
                                    std::to_string(steps_beside) + " steps)").c_str()
                                 : "",
                        P[0]->size(), r[0].tokens.size(), r[0].steps, batched[0], trees[0], r[0].accepted,
                        same[0] ? "identical" : "DIFFERS", P[1]->size(), r[1].tokens.size(), r[1].steps, batched[1],
                        trees[1], r[1].accepted, same[1] ? "identical" : "DIFFERS");
            for (int k = 0; k < 2; ++k)
              if (!same[k]) {
                size_t i = 0;
                while (i < r[k].tokens.size() && i < alone[k].tokens.size() && r[k].tokens[i] == alone[k].tokens[i]) ++i;
                std::printf("  %c alone: %zu out, %ld steps %ld accepted, state %s, drafter context %s; first token "
                            "difference at %zu\n",
                            "AB"[k], alone[k].tokens.size(), alone[k].steps, alone[k].accepted,
                            alone[k].hash == r[k].hash ? "identical" : "differs",
                            alone[k].draft == r[k].draft ? "identical" : "differs", i);
              }
          }
        }
      }
      std::printf(failures ? "BATCH CHECK FAILED\n" : "batch check passed\n");
      return failures ? 1 : 0;
    }
    if (fanout_check) {
      const int n = engine.sessions();
      if (n < 2 || prompts.empty()) throw std::runtime_error("--fanout-check needs --sessions 2 and a prompt");
      auto digest = [](const std::vector<float>& l) {
        uint64_t h = 14695981039346656037ull;
        for (float f : l) {
          uint32_t b;
          std::memcpy(&b, &f, sizeof b);
          h = (h ^ b) * 1099511628211ull;
        }
        return h;
      };
      auto start = [&](Seq& q, int session, const ragon::SamplingParams& sp) {
        q.session = session;
        q.sp = sp;
        activate(session, "", true);
        engine.prefill(prompts[session % prompts.size()], nullptr, false, IMG);
        q.pending = {engine.sample(engine.logits(), q.sp, engine.history())};
      };
      auto emit = [&](Seq& q) {
        for (int t : q.pending) {
          q.out.push_back(t);
          if (is_end(t) || static_cast<int>(q.out.size()) >= max_tokens) return q.done = true;
        }
        return false;
      };
      struct Case {
        const char* name;
        float temperature;
        int top_k;
      };
      bool failed = false;
      for (const Case& cs : {Case{"greedy", 0.f, 20}, Case{"top_k 40", 0.8f, 40}, Case{"no top_k", 0.8f, 0}}) {
        ragon::SamplingParams sp;
        sp.temperature = cs.temperature;
        sp.top_k = cs.top_k;
        sp.top_p = 0.9f;
        sp.seed = 7;
        std::vector<Result> alone(n);
        std::vector<std::vector<uint64_t>> alone_l(n);
        engine.reset_all();
        for (int i = 0; i < n; ++i) {
          Seq q;
          start(q, i, sp);
          while (!emit(q)) {
            const std::vector<float>& l = engine.step(q.pending.back());
            alone_l[i].push_back(digest(l));
            q.pending = {engine.sample(l, q.sp, engine.history())};
          }
          alone[i] = result_of(q);
        }
        for (const bool reversed : {false, true}) {
          engine.reset_all();
          std::vector<Seq> seqs(n);
          std::vector<std::vector<uint64_t>> l(n);
          std::vector<std::vector<float>> rows(n);
          for (int i = 0; i < n; ++i) start(seqs[i], i, sp);
          const auto t0 = std::chrono::steady_clock::now();
          long tokens = 0;
          while (true) {
            std::vector<ragon::Engine::SeqStep> steps;
            std::vector<int> in;
            for (int k = 0; k < n; ++k) {
              const int i = reversed ? n - 1 - k : k;
              if (seqs[i].done || emit(seqs[i])) continue;
              ragon::Engine::SeqStep st;
              st.session = i;
              st.anchor = seqs[i].pending.back();
              st.params = &seqs[i].sp;
              st.row_logits = &rows[i];
              steps.push_back(st);
              in.push_back(i);
            }
            if (steps.empty()) break;
            engine.step_batch(steps);
            for (size_t k = 0; k < steps.size(); ++k) {
              if (steps[k].error) std::rethrow_exception(steps[k].error);
              seqs[in[k]].pending = steps[k].out;
              l[in[k]].push_back(digest(rows[in[k]]));
              ++tokens;
            }
          }
          const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
          std::printf("fanout %s %s: %d sequences, %.1f tok/s\n", cs.name, reversed ? "reversed" : "in order", n,
                      tokens / dt);
          for (int i = 0; i < n; ++i) {
            const Result r = result_of(seqs[i]);
            const bool same = r.tokens == alone[i].tokens && r.hash == alone[i].hash && r.draft == alone[i].draft;
            if (same && l[i] == alone_l[i]) continue;
            failed = true;
            size_t at = 0;
            while (at < l[i].size() && at < alone_l[i].size() && l[i][at] == alone_l[i][at]) ++at;
            std::printf("  sequence %d differs: tokens %s, state %s, drafter %s, logits from step %zu\n", i,
                        r.tokens == alone[i].tokens ? "same" : "DIFFER", r.hash == alone[i].hash ? "same" : "DIFFER",
                        r.draft == alone[i].draft ? "same" : "DIFFER", at);
          }
        }
      }
      std::printf(failed ? "FANOUT CHECK FAILED\n" : "fanout check passed\n");
      return failed ? 1 : 0;
    }
    if (fork_check) {
      // BATCHING P4d, the probabilities test: N forks of one prompt against N separate requests with that prompt, each
      // prefilled from scratch. Every step's logits (hence every probability), every token, the state and the drafter's
      // context must be the same bit for bit, with a prompt ending inside a KV page and one ending on a page boundary.
      const int n = engine.sessions();
      if (n < 2 || prompts.empty()) throw std::runtime_error("--fork-check needs --sessions 2 and a prompt");
      auto digest = [](const std::vector<float>& l) {
        uint64_t h = 14695981039346656037ull;
        for (float f : l) {
          uint32_t b;
          std::memcpy(&b, &f, sizeof b);
          h = (h ^ b) * 1099511628211ull;
        }
        return h;
      };
      std::vector<int> longest = *std::max_element(
          prompts.begin(), prompts.end(), [](const auto& a, const auto& b) { return a.size() < b.size(); });
      const size_t page = 64;  // KvPool::kPageTokens
      if (longest.size() < 2 * page) throw std::runtime_error("--fork-check needs a prompt of two pages");
      const size_t boundary = longest.size() / page * page;
      struct Case {
        const char* name;
        float temperature;
        int top_k;
      };
      struct Arm {
        std::vector<std::vector<uint64_t>> first, steps;
        std::vector<Result> results;
      };
      bool failed = false;
      for (const size_t len : {boundary, boundary - page + 17}) {
        const std::vector<int> prompt(longest.begin(), longest.begin() + len);
        for (const Case& cs : {Case{"greedy", 0.f, 20}, Case{"top_k 40", 0.8f, 40}, Case{"no top_k", 0.8f, 0}}) {
          std::vector<ragon::SamplingParams> params(n);
          for (int i = 0; i < n; ++i) {
            params[i].temperature = cs.temperature;
            params[i].top_k = cs.top_k;
            params[i].top_p = 0.9f;
            params[i].seed = 7 + uint64_t(i) * 0x9E3779B97F4A7C15ull;
          }
          auto decode = [&](std::vector<Seq>& seqs, std::vector<std::vector<uint64_t>>& l) {
            auto emit = [&](Seq& q) {
              for (int t : q.pending) {
                q.out.push_back(t);
                if (is_end(t) || static_cast<int>(q.out.size()) >= max_tokens) return q.done = true;
              }
              return false;
            };
            std::vector<std::vector<float>> rows(n);
            long tokens = 0;
            const auto t0 = std::chrono::steady_clock::now();
            while (true) {
              std::vector<ragon::Engine::SeqStep> steps;
              std::vector<int> in;
              for (int i = 0; i < n; ++i) {
                if (seqs[i].done || emit(seqs[i])) continue;
                ragon::Engine::SeqStep st;
                st.session = i;
                st.anchor = seqs[i].pending.back();
                st.params = &seqs[i].sp;
                st.row_logits = &rows[i];
                steps.push_back(st);
                in.push_back(i);
              }
              if (steps.empty()) break;
              engine.step_batch(steps);
              for (size_t k = 0; k < steps.size(); ++k) {
                if (steps[k].error) std::rethrow_exception(steps[k].error);
                seqs[in[k]].pending = steps[k].out;
                l[in[k]].push_back(digest(rows[in[k]]));
                ++tokens;
              }
            }
            return tokens / std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
          };
          auto arm = [&](bool forked, double* rate, double* setup) {
            Arm a;
            a.steps.resize(n);
            a.first.resize(n);
            std::vector<Seq> seqs(n);
            engine.reset_all();
            const auto t0 = std::chrono::steady_clock::now();
            std::vector<float> prompt_logits;
            for (int i = 0; i < n; ++i) {
              seqs[i].session = i;
              seqs[i].sp = params[i];
              activate(i, "", true);
              if (i == 0 || !forked) prompt_logits = engine.prefill(prompt, nullptr, true, nullptr);
              else {
                engine.fork_session(0);
                prompt_logits = engine.logits();  // the scheduler's fork draws from these: the parent's, unchanged
              }
              a.first[i].push_back(digest(prompt_logits));
              seqs[i].pending = {engine.sample(prompt_logits, seqs[i].sp, engine.history())};
            }
            *setup = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            *rate = decode(seqs, a.steps);
            for (int i = 0; i < n; ++i) a.results.push_back(result_of(seqs[i]));
            return a;
          };
          double rate0, rate1, setup0, setup1;
          const Arm separate = arm(false, &rate0, &setup0), forks = arm(true, &rate1, &setup1);
          std::printf("fork %s, prompt %zu: %d sequences; separate %.2f s then %.1f tok/s, forked %.2f s then %.1f tok/s\n",
                      cs.name, len, n, setup0, rate0, setup1, rate1);
          std::set<std::vector<int>> distinct;
          for (int i = 0; i < n; ++i) {
            distinct.insert(forks.results[i].tokens);
            const Result& a = separate.results[i];
            const Result& b = forks.results[i];
            if (a.tokens == b.tokens && a.hash == b.hash && a.draft == b.draft && separate.steps[i] == forks.steps[i] &&
                separate.first[i] == forks.first[i])
              continue;
            failed = true;
            size_t at = 0;
            while (at < separate.steps[i].size() && at < forks.steps[i].size() &&
                   separate.steps[i][at] == forks.steps[i][at])
              ++at;
            std::printf("  fork %d differs: prompt logits %s, tokens %s, state %s, drafter %s, logits from step %zu\n", i,
                        separate.first[i] == forks.first[i] ? "same" : "DIFFER", a.tokens == b.tokens ? "same" : "DIFFER",
                        a.hash == b.hash ? "same" : "DIFFER", a.draft == b.draft ? "same" : "DIFFER", at);
          }
          // Distinct seeds must give forks of their own: seeds that only shift one stream would pass the rest.
          if (cs.temperature > 0.f && static_cast<int>(distinct.size()) < std::max(2, n / 2)) {
            failed = true;
            std::printf("  only %zu distinct answers among %d sampled forks\n", distinct.size(), n);
          }
        }
      }
      std::printf(failed ? "FORK CHECK FAILED\n" : "fork check passed\n");
      return failed ? 1 : 0;
    }
    if (prefix_check) {
      auto argmax = [](const std::vector<float>& l) { return int(std::max_element(l.begin(), l.end()) - l.begin()); };
      // With a drafter: 64 tokens decoded speculatively (greedy) from the state the prefill left.
      struct After {
        std::vector<int> tokens;
        long steps = 0, accepted = 0;
        uint64_t hash = 0;
      };
      auto continue_spec = [&](const std::vector<float>& logits) {
        After a;
        if (!engine.has_drafter()) return a;
        ragon::SamplingParams gp;
        gp.temperature = 0.f;
        engine.reset_spec_stats();
        std::vector<int> pending = {engine.sample(logits, gp, engine.history())};
        while (true) {
          a.tokens.insert(a.tokens.end(), pending.begin(), pending.end());
          if (a.tokens.size() >= 64 || is_end(a.tokens.back())) break;
          pending = engine.speculate(pending.back(), gp);
        }
        a.steps = engine.spec_stats().steps;
        a.accepted = engine.spec_stats().accepted;
        a.hash = engine.state_hash();
        return a;
      };
      engine.reset();
      for (size_t i = 0; i < prompts.size(); ++i) {
        // Bind the prompt as ragon-serve does: the session it extends, else a new one (a never-used slot,
        // else the least recently used).
        int si = engine.match_session(prompts[i], "");
        const bool fresh_session = si < 0;
        if (fresh_session) si = engine.free_session(0.0, {});
        // P4b: a short pool takes idle sessions' pages, least recently used first (P4c: the prefix KV store's first).
        auto make_room = [&](const std::function<bool()>& short_of_pages) {
          if (!engine.kv_paged()) return;
          std::vector<bool> busy(engine.sessions(), false);
          busy[si] = true;
          while (short_of_pages()) {
            if (engine.kv_store_evict()) {
              std::fprintf(stderr, "prefix-check: a prefix KV store entry evicted for KV pages\n");
              engine.kv_retire(true);
              continue;
            }
            const int victim = engine.kv_lru_idle(busy);
            if (victim < 0) break;
            std::fprintf(stderr, "prefix-check: session %d evicted for KV pages\n", victim);
            engine.kv_evict(victim);
            engine.kv_retire(true);
          }
        };
        make_room([&] { return engine.kv_admit_need(si, prompts[i]) > engine.kv_pages_free(); });
        activate(si, "", fresh_session);
        ragon::EngineStats st;
        const std::vector<float> resumed_logits = engine.prefill(prompts[i], &st, false, IMG);
        const int resumed = argmax(resumed_logits);
        const uint64_t h_resumed = engine.state_hash();
        const After a_resumed = continue_spec(resumed_logits);
        // The reference prefill maps all of the prompt's pages afresh: the pages the resumed one shares with a donor or
        // the store stay theirs (P4c), so a short pool makes room again (only the session's own pages come back).
        make_room([&] {
          return engine.kv_admit_pages(prompts[i].size()) > engine.kv_pages_free() + engine.kv_reclaimable({si}, false);
        });
        ragon::EngineStats fresh_st;
        const std::vector<float> fresh_logits = engine.prefill(prompts[i], &fresh_st, /*from_scratch=*/true, IMG);
        const int fresh = argmax(fresh_logits);
        const uint64_t h_fresh = engine.state_hash();
        const After a_fresh = continue_spec(fresh_logits);
        engine.touch_session();
        const bool same = resumed == fresh && h_resumed == h_fresh;
        const bool spec_same = a_resumed.tokens == a_fresh.tokens && a_resumed.hash == a_fresh.hash &&
                               a_resumed.steps == a_fresh.steps && a_resumed.accepted == a_fresh.accepted;
        failures += !same + !spec_same;
        std::printf("prompt %zu: %zu tokens, session %d%s, resumed at %d%s (%.2f s) against %.2f s from scratch: state %s, "
                    "next token %s",
                    i, prompts[i].size(), si, fresh_session ? " (new)" : "", st.reused_tokens,
                    st.store_tokens > 0 ? " from the prefix KV store" : "", st.prefill_seconds,
                    fresh_st.prefill_seconds, h_resumed == h_fresh ? "identical" : "DIFFERS",
                    resumed == fresh ? "identical" : "DIFFERS");
        if (engine.has_drafter())
          std::printf("; speculative continuation %zu tokens, %ld steps %ld accepted against %ld steps %ld accepted: %s",
                      a_resumed.tokens.size(), a_resumed.steps, a_resumed.accepted, a_fresh.steps, a_fresh.accepted,
                      spec_same ? "identical" : "DIFFERS");
        std::printf("\n");
        // The session is left as a server would leave it after prompt i (and its continuation).
      }
      std::printf(failures ? "PREFIX CHECK FAILED\n" : "prefix check passed\n");
      return failures ? 1 : 0;
    }
    for (const std::vector<int>& prompt : prompts) {
      ragon::SamplingParams sp;
      sp.temperature = temperature;
      sp.seed = seed;
      if (graph_check) {
        for (int sampled = 0; sampled < 2; ++sampled) {
          ragon::SamplingParams gp;
          gp.temperature = sampled ? (temperature > 0.f ? temperature : 1.f) : 0.f;
          gp.seed = seed ? seed : 1;
          double t_eager = 0, t_graph = 0;
          // The steps and accepted drafts must match too: the commit also writes the drafter's context KV, which
          // neither the tokens (the verify corrects every draft) nor the state hash would show if it were wrong.
          engine.set_step_graphs(false);
          engine.reset_spec_stats();
          const std::vector<int> eager = generate(prompt, gp, true, true, &t_eager);
          const uint64_t h_eager = engine.state_hash();
          const long steps_eager = engine.spec_stats().steps, acc_eager = engine.spec_stats().accepted;
          engine.set_step_graphs(true);
          if (!engine.step_graphs()) throw std::runtime_error("--graph-check: step graphs are unavailable");
          engine.reset_spec_stats();
          const std::vector<int> graph = generate(prompt, gp, true, true, &t_graph);
          const uint64_t h_graph = engine.state_hash();
          const long steps_graph = engine.spec_stats().steps, acc_graph = engine.spec_stats().accepted;
          const bool same = eager == graph, state = h_eager == h_graph;
          const bool accept = steps_eager == steps_graph && acc_eager == acc_graph;
          failures += !same + !state + !accept;
          std::printf("prompt %zu tokens, %s: eager %zu tokens %.2f tok/s, %ld steps %ld accepted | graphs %zu tokens "
                      "%.2f tok/s, %ld steps %ld accepted | tokens %s, state %s, acceptance %s\n",
                      prompt.size(), sampled ? "sampled" : "greedy", eager.size(), eager.size() / t_eager, steps_eager,
                      acc_eager, graph.size(), graph.size() / t_graph, steps_graph, acc_graph,
                      same ? "identical" : "DIFFER", state ? "identical" : "DIFFERS", accept ? "identical" : "DIFFERS");
          if (!same) {
            size_t i = 0;
            while (i < eager.size() && i < graph.size() && eager[i] == graph[i]) ++i;
            std::printf("  first difference at output token %zu\n", i);
          }
        }
        continue;
      }
      if (spec_check) {
        sp.temperature = 0.f;
        double t_plain = 0;
        const std::vector<int> plain = generate(prompt, sp, false, true, &t_plain);
        // One speculative run under the drafter-off guard at (off_below, margin), against the plain run: the tokens,
        // and the state and the drafter's context against a plain replay of the same tokens from the same prefill.
        auto spec_run = [&](const char* label, double off_below, double margin) {
          engine.set_draft_guard(off_below, margin);
          engine.reset_spec_stats();
          double t_spec = 0;
          const std::vector<int> fast = generate(prompt, sp, true, true, &t_spec);
          const uint64_t h_spec = engine.state_hash(), d_spec = engine.draft_hash();
          const ragon::SpecStats st = engine.spec_stats();
          const std::vector<int> processed(engine.history().begin() + prompt.size(), engine.history().end());
          // The same tokens through plain decoding, from the same prefill.
          engine.reset();
          engine.prefill(prompt, nullptr, false, IMG);
          for (int t : processed) engine.step(t);
          const uint64_t h_plain = engine.state_hash(), d_plain = engine.draft_hash();
          const bool same = plain == fast, state = h_spec == h_plain, ring = d_spec == d_plain;
          failures += !same + !state + !ring;
          std::printf("prompt %zu tokens%s: plain %zu tokens %.2f tok/s | spec %zu tokens %.2f tok/s, %ld steps, %.2f "
                      "tokens/step | tokens %s, state %s, drafter context %s\n",
                      prompt.size(), label, plain.size(), plain.size() / t_plain, fast.size(), fast.size() / t_spec,
                      st.steps, st.steps ? double(st.accepted + st.steps) / st.steps : 0.0,
                      same ? "identical" : "DIFFER", state ? "identical" : "DIFFERS", ring ? "identical" : "DIFFERS");
          if (st.tree_steps)
            std::printf("  tree: %ld steps, %ld left the chain (%ld off-chain tokens)\n", st.tree_steps,
                        st.tree_branch_steps, st.tree_off_chain);
          if (st.drafter_off_steps || st.drafter_switches)
            std::printf("  drafter-off guard: %ld plain steps, %ld switches\n", st.drafter_off_steps,
                        st.drafter_switches);
          if (!same) {
            size_t i = 0;
            while (i < plain.size() && i < fast.size() && plain[i] == fast[i]) ++i;
            std::printf("  first difference at output token %zu\n", i);
          }
        };
        if (draft_off_below < 0) {
          spec_run("", engine.draft_off_below(), engine.draft_on_margin());
        } else {  // the guard's check (BACKLOG 10)
          const double margin = engine.draft_on_margin();
          spec_run(", guard off", 0, margin);
          char label[96];
          std::snprintf(label, sizeof label, ", guard at %.2f", draft_off_below);
          spec_run(label, draft_off_below, margin);
          std::snprintf(label, sizeof label, ", guard at %.2f, back on at every probe", draft_off_below);
          spec_run(label, draft_off_below, -draft_off_below);
          engine.set_draft_guard(draft_off_below, margin);
        }
        continue;
      }
      double dt = 0;
      engine.reset_spec_stats();
      const std::vector<int> out = generate(prompt, sp, spec, false, &dt);
      if (prompt_ids_out) {
        for (size_t i = 0; i < prompt.size(); ++i) std::cout << (i ? "," : "") << prompt[i];
        std::cout << "\n";
      }
      if (ids_out) {
        for (size_t i = 0; i < out.size(); ++i) std::cout << (i ? "," : "") << out[i];
        std::cout << "\n";
      }
      std::fprintf(stderr, "decode: %zu tokens in %.3f s (%.2f tok/s)\n", out.size(), dt, out.size() / dt);
      if (spec) {
        const ragon::SpecStats& st = engine.spec_stats();
        std::fprintf(stderr, "spec: %ld steps, %.2f tokens/step, %.1f ms/step (draft %.1f, verify %.1f, commit %.1f)\n",
                     st.steps, st.steps ? double(st.accepted + st.steps) / st.steps : 0.0,
                     st.steps ? 1e3 * (st.draft_seconds + st.verify_seconds + st.commit_seconds) / st.steps : 0.0,
                     st.steps ? 1e3 * st.draft_seconds / st.steps : 0.0, st.steps ? 1e3 * st.verify_seconds / st.steps : 0.0,
                     st.steps ? 1e3 * st.commit_seconds / st.steps : 0.0);
        if (st.tree_steps)
          std::fprintf(stderr, "tree: %ld steps, %ld left the chain (%ld off-chain tokens)\n", st.tree_steps,
                       st.tree_branch_steps, st.tree_off_chain);
        if (st.drafter_off_steps || st.drafter_switches)
          std::fprintf(stderr, "drafter-off guard: %ld plain steps, %ld switches\n", st.drafter_off_steps,
                       st.drafter_switches);
        if (st.sparse_steps || st.sparse_fallbacks)
          std::fprintf(stderr, "sparse head: %ld of %ld steps, %ld fallbacks to the full head\n", st.sparse_steps,
                       st.steps, st.sparse_fallbacks);
        if (st.lookup_steps)
          std::fprintf(stderr, "lookup: %ld steps verified its chain, %.2f tokens/step\n", st.lookup_steps,
                       double(st.lookup_accepted + st.lookup_steps) / st.lookup_steps);
        if (st.shadow_total_steps) {
          std::fprintf(stderr, "lookup shadow: drafter alone %.2f accepted/step, best of both %.2f\n",
                       double(st.shadow_dflash_all) / st.shadow_total_steps, double(st.shadow_best) / st.shadow_total_steps);
          const char* names[5] = {"3", "4-7", "8-15", "16-31", "32+"};
          for (int b = 0; b < 5; ++b)
            if (st.shadow_steps[b])
              std::fprintf(stderr, "  match %-5s %5ld steps: lookup %.2f, drafter %.2f accepted/step\n", names[b],
                           st.shadow_steps[b], double(st.shadow_lookup[b]) / st.shadow_steps[b],
                           double(st.shadow_dflash[b]) / st.shadow_steps[b]);
        }
      }
    }
    if (graph_check) {
      std::printf(failures ? "GRAPH CHECK FAILED\n" : "graph check passed\n");
      return failures ? 1 : 0;
    }
    if (spec_check) {
      std::printf(failures ? "SPEC CHECK FAILED\n" : "spec check passed\n");
      return failures ? 1 : 0;
    }
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
