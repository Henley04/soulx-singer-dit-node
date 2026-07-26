// infer.h — Public API for SoulX-Singer DiT (DiffLlama) GGUF inference
//
// This header exposes the types and functions needed by the N-API binding
// (binding.cc). The implementation lives in infer.cpp.
//
// Design notes:
//   - GGUF weights are mmap'd and kept in their packed (quantized) form.
//     All matmuls go through ggml_mul_mat(), which dispatches to ggml-cpu's
//     quantized kernels. Weights are NEVER expanded to FP32 in memory.
//   - No dequantization occurs anywhere in this code path.

#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef _WIN32
  #include <windows.h>
#endif

// ---------------------------------------------------------------------------
// Model hyperparameters (mirror convert_dit_to_gguf.py DIT_HPARAMS).
//
// MEL_DIM / HIDDEN / NUM_HEADS / HEAD_DIM are architectural constants shared
// by every SoulX-Singer DiT variant — teacher (22L), distilled students (4L
// or 11L), pruned students (4L with FFN/MLP intermediate reduced to 1024 or
// 2048) — so they stay constexpr. RMS_EPS / ROPE_THETA likewise.
//
// NUM_LAYERS and INTERMEDIATE *do* vary across variants, so NUM_LAYERS is
// resolved per-model from the GGUF metadata at load time and stored on the
// Model struct below (see `Model::num_layers`). The constexpr below is the
// DEFAULT used for legacy teacher GGUFs that don't carry the metadata.
// INTERMEDIATE is not referenced by the C++ path at all — the FFN intermediate
// dimension is read implicitly from the `blk.N.ffn_gate.weight` tensor shape
// via ggml_mul_mat, so pruned students (FFN=1024 or 2048) work without any
// code change beyond the layer loop bound.
// ---------------------------------------------------------------------------
static constexpr int   MEL_DIM       = 128;
static constexpr int   HIDDEN        = 1024;
static constexpr int   NUM_LAYERS    = 22;     // default for legacy teacher
                                               // GGUFs; overridden at load
                                               // time when the GGUF carries
                                               // llama.block_count
static constexpr int   NUM_HEADS     = 16;
static constexpr int   HEAD_DIM      = 64;     // 1024 / 16
static constexpr int   INTERMEDIATE  = 4096;   // 4 * hidden (default; pruned
                                               // students use 1024 or 2048.
                                               // Not referenced by C++; FFN
                                               // shapes come from GGUF.)
static constexpr float RMS_EPS       = 1e-6f;
static constexpr float ROPE_THETA    = 10000.0f;

// ---------------------------------------------------------------------------
// Tiny deterministic PRNG — LCG + Box-Muller (no libc rand() so results are
// fully reproducible across machines, no Python dependency).
// ---------------------------------------------------------------------------
struct RNG {
    uint64_t s;
    explicit RNG(uint64_t seed) : s(seed) {}
    uint32_t u32() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return (uint32_t)(s >> 32);
    }
    float uniform() { return (u32() >> 8) * (1.0f / (1u << 24)); }
    float gaussian() {
        float u1 = uniform(); if (u1 < 1e-7f) u1 = 1e-7f;
        float u2 = uniform();
        return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979323846f * u2);
    }
};

// Forward declaration: the per-T compute graph/arena cache lives in infer.cpp.
// It is an optimization cache (does not affect the model's logical state) and
// is therefore mutable so run_forward() can update it through a const Model&.
struct GraphCache;

// ---------------------------------------------------------------------------
// mmap'd GGUF model. Quantized weight blobs stay in their packed form; tensor
// ->data pointers are set directly into the mmap region.
//
// Runtime backend selection:
//   On load, pick_backend() enumerates the ggml backend devices compiled into
//   this .node (CPU always; CUDA/HIP/Metal/Vulkan if enabled at build time)
//   and selects the best one. When a GPU backend is chosen, weights are
//   uploaded into a backend buffer (VRAM) and graph compute runs on the GPU.
//   When CPU is chosen, weights stay mmap'd (zero-copy) and graph compute runs
//   on the host. The same source/build serves every platform; the C++ layer
//   auto-selects the fastest path at runtime.
// ---------------------------------------------------------------------------
struct Model {
    std::string path;
    ggml_context * ctx   = nullptr;
    gguf_context * gguf  = nullptr;
    int   fd             = -1;
    void * mmap_data     = nullptr;
    size_t mmap_size     = 0;
#ifdef _WIN32
    HANDLE file_mapping  = nullptr;  // file mapping handle for mmap cleanup
#endif

    // ---- Runtime backend -------------------------------------------------
    // For GPU backends: backend handle + buffer type used to upload weights
    // and allocate per-call intermediates. For CPU: backend is null and the
    // mmap zero-copy path is used (no backend buffer for weights).
    ggml_backend_t              backend     = nullptr;
    ggml_backend_buffer_type_t  buft        = nullptr;
    ggml_backend_buffer_t       weight_buf  = nullptr;
    bool                        use_gpu     = false;
    int                         n_threads   = 0;   // resolved at load time
    std::string                 backend_name;

    // Per-T compute graph/arena cache (plan item #8). Mutable so it can be
    // built/refreshed by run_forward() through a const Model&. Owned raw
    // pointer freed in unload(); null means "not yet built".
    mutable GraphCache * graph_cache = nullptr;

    // ---- Resolved per-model hyperparameters ----
    // NUM_LAYERS varies across DiT variants (teacher=22, distilled
    // baseline-distill=11, ProbeKD/HiddenMatch/On-Policy/Wass students=4).
    // Resolved at load time from GGUF metadata (llama.block_count); falls
    // back to counting `blk.N.attn_q.weight` tensors, then to the NUM_LAYERS
    // constexpr (22) for legacy teacher GGUFs that carry neither. Used by
    // dit_forward() to bound the decoder layer loop — the only architectural
    // knob that changes between the supported variants.
    int   num_layers   = NUM_LAYERS;

    bool load(const std::string & path_, const std::string & backend_pref = "");
    void unload();

    ggml_tensor * get(const std::string & name) const {
        return ggml_get_tensor(ctx, name.c_str());
    }
    ggml_tensor * get_or_die(const std::string & name) const {
        ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
        if (!t) {
            fprintf(stderr, "FATAL: tensor '%s' missing from GGUF\n", name.c_str());
            exit(1);
        }
        return t;
    }
};

// ---------------------------------------------------------------------------
// Runtime helpers (implemented in infer.cpp).
// ---------------------------------------------------------------------------

// Decide the worker thread count for CPU graph compute. Uses
// std::thread::hardware_concurrency(), caps to the cgroup v2/v1 CPU quota
// (important in containers), and (best-effort on Linux) caps to the physical
// core count to avoid SMT oversubscription. Never returns less than 1.
int runtime_thread_count();

// Pick the best available ggml backend device. Priority: dedicated GPU
// (CUDA/HIP/Metal/SYCL) > Vulkan > CPU. `pref` may be "cpu" to force the CPU
// path, "gpu" to require a GPU (falls back to CPU if none), or "" / "auto"
// for automatic selection. Sets `out_is_gpu` accordingly and returns the
// initialized backend (caller frees via ggml_backend_free).
ggml_backend_t pick_backend(const std::string & pref, bool & out_is_gpu,
                            std::string & out_name);

// ---------------------------------------------------------------------------
// Run a single DiT forward pass.
//   x_data : [MEL_DIM, T]  f32, row-major (x[k*T + t])
//   cond   : [HIDDEN,  T]  f32, row-major
//   t      : scalar diffusion timestep
//   T      : sequence length
//   Returns: [MEL_DIM, T] f32 (the "velocity" output of the flow-matching net)
// ---------------------------------------------------------------------------
std::vector<float> run_forward(
    const Model & m,
    const float * x_data,
    const float * cond_data,
    float t,
    int T);

// ---------------------------------------------------------------------------
// Reverse diffusion (Euler ODE integration, mirrors
// FlowMatchingTransformer.reverse_diffusion).
//   prompt_mel : [MEL_DIM, prompt_len]   row-major
//   cond_full  : [HIDDEN,  T]            row-major, T = prompt_len + target_len
//   z_init     : [MEL_DIM, target_len]   row-major (initial noise)
//   n_steps    : ODE steps (e.g. 8)
//   Returns    : [MEL_DIM, target_len]   row-major (final xt)
// ---------------------------------------------------------------------------
std::vector<float> run_reverse_diffusion(
    const Model & m,
    const std::vector<float> & prompt_mel,
    const std::vector<float> & cond_full,
    const std::vector<float> & z_init,
    int prompt_len, int target_len, int n_steps,
    RNG & rng);
