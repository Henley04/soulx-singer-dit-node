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
#include "gguf.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Model hyperparameters (mirror convert_dit_to_gguf.py DIT_HPARAMS)
// ---------------------------------------------------------------------------
static constexpr int   MEL_DIM       = 128;
static constexpr int   HIDDEN        = 1024;
static constexpr int   NUM_LAYERS    = 22;
static constexpr int   NUM_HEADS     = 16;
static constexpr int   HEAD_DIM      = 64;     // 1024 / 16
static constexpr int   INTERMEDIATE  = 4096;   // 4 * hidden
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

// ---------------------------------------------------------------------------
// mmap'd GGUF model. Quantized weight blobs stay in their packed form; tensor
// ->data pointers are set directly into the mmap region.
// ---------------------------------------------------------------------------
struct Model {
    std::string path;
    ggml_context * ctx   = nullptr;
    gguf_context * gguf  = nullptr;
    int   fd             = -1;
    void * mmap_data     = nullptr;
    size_t mmap_size     = 0;

    bool load(const std::string & path_);
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
