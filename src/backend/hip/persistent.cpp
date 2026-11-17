// The megakernels on HIP (BACKEND §4, kPersistentKernels false): a persistent kernel needs every block resident at
// once, which nothing guarantees here (no cooperative launch under WDDM, AMD §5). Each entry answers false and
// launches nothing, the answer kernels.cuh gives for shapes it was not built for, so the engine runs the rows and
// graph path instead; the engine also leaves the stages off on HIP (engine.cpp), so these are not called.
#include "core/kernels.cuh"

namespace ragon::kernels {

bool attn_persistent(const MkAttnParams&, gpu::Stream) { return false; }
void attn_persistent_prepare() {}
bool ffn_persistent(const gpu::half*, const float*, int, const StreamTarget&, const StreamTarget&, const StreamTarget&,
                    int, int, float*, float*, gpu::half*, float*, unsigned*, gpu::Stream, unsigned long long*,
                    const MkTail*) {
  return false;
}
bool gdn_ffn_persistent(const MkGdnParams&, const StreamTarget&, const StreamTarget&, const StreamTarget&, int, float*,
                        float*, gpu::half*, float*, unsigned*, unsigned long long*, const MkTail*, gpu::Stream) {
  return false;
}
bool gdn_persistent(const MkGdnParams&, gpu::Stream) { return false; }

}  // namespace ragon::kernels
