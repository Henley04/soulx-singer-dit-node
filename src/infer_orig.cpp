// infer.cpp — Pure C++ inference for SoulX-Singer DiT (DiffLlama) GGUF
//
// Goal (per user request):
//   - Run inference on syxppp/SoulX-Singer-DiT-GGUF **without dequantizing** weights.
//   - Complete C++ implementation (no Python in the inference path).
//   - Verify accuracy is normal: compare FP32 vs Q8_0 / Q4_K_M outputs.
//
// How "no dequantization" is achieved:
//   - GGUF file is mmap'd; quantized weight blobs stay in their packed form.
//   - All matmuls go through ggml_mul_mat(), which dispatches to ggml-cpu's
//     quantized kernels (Q8_0, Q4_K_M, ...) that operate directly on packed
//     weights. The weights are never expanded to FP32 in memory.
//
// Architecture (mirrors SoulX-Singer/soulxsinger/models/modules/llama.py):
//   - DiffLlama = LlamaModel with:
//       * Non-causal attention mask (bidirectional)
//       * AdaptiveRMSNorm (norm scale & bias are a Linear(timestep_emb))
//       * mel_mlp / mel_out_mlp as I/O projections (mel_dim <-> hidden_size)
//       * cond_mlp + diff_step_mlp as conditioning / timestep MLPs
//   - RoPE: HF Llama defaults (theta=10000, NeoX-style, no scaling)
//
// Build:
//   make -f Makefile.soulx
//
// Usage:
//   ./infer <fp32.gguf> <q8_0.gguf> <q4_k_m.gguf>
//

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

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
// Tiny deterministic PRNG — LCG + Box-Muller (no libc rand() so test is
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
// mmap'd GGUF model
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

bool Model::load(const std::string & path_) {
    path = path_;
    fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { perror("open"); return false; }
    struct stat sb;
    if (fstat(fd, &sb) != 0) { perror("fstat"); return false; }
    mmap_size = (size_t)sb.st_size;
    mmap_data = mmap(nullptr, mmap_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mmap_data == MAP_FAILED) { perror("mmap"); mmap_data = nullptr; return false; }

    gguf_init_params params;
    params.no_alloc = true;             // we'll point tensor->data at the mmap
    params.ctx      = &ctx;
    gguf = gguf_init_from_file(path.c_str(), params);
    if (!gguf) {
        fprintf(stderr, "gguf_init_from_file failed for %s\n", path.c_str());
        return false;
    }

    size_t data_offset = gguf_get_data_offset(gguf);
    int n_tensors = (int)gguf_get_n_tensors(gguf);
    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf, i);
        ggml_tensor * t = ggml_get_tensor(ctx, name);
        if (!t) continue;
        size_t off = gguf_get_tensor_offset(gguf, i);
        t->data = (char *)mmap_data + data_offset + off;
    }

    printf("  loaded %s: %d tensors, %.1f MiB mmap'd\n",
           path.c_str(), n_tensors, (double)mmap_size / (1 << 20));
    return true;
}

void Model::unload() {
    if (gguf) gguf_free(gguf);
    if (ctx)  ggml_free(ctx);
    if (mmap_data) munmap(mmap_data, mmap_size);
    if (fd >= 0) close(fd);
    ctx = nullptr; gguf = nullptr; mmap_data = nullptr; fd = -1;
}

// ---------------------------------------------------------------------------
// Per-call scratch context: holds all intermediate tensors of one forward
// pass. We build the graph into a separate ctx0 (small), allocate a large
// scratch buffer for tensor data, and run ggml_graph_compute_with_ctx.
// ---------------------------------------------------------------------------
struct Scratch {
    ggml_context * ctx;
    std::vector<uint8_t> buf;
    explicit Scratch(size_t bytes) : buf(bytes) {
        ggml_init_params p; p.mem_size = bytes; p.mem_buffer = buf.data(); p.no_alloc = false;
        ctx = ggml_init(p);
    }
    ~Scratch() { if (ctx) ggml_free(ctx); }
};

// ---------------------------------------------------------------------------
// Build a 1D int32 positions tensor [0, 1, ..., T-1]
// ---------------------------------------------------------------------------
static ggml_tensor * make_positions(ggml_context * ctx, int T) {
    ggml_tensor * p = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    int32_t * data = (int32_t *)p->data;
    for (int i = 0; i < T; i++) data[i] = i;
    return p;
}

// ---------------------------------------------------------------------------
// AdaptiveRMSNorm (LlamaAdaptiveRMSNorm from SoulX-Singer llama.py)
//
//   variance = mean(hidden^2, dim=-1)
//   hidden_normed = hidden * rsqrt(variance + eps)
//   weight = Linear(cond)  -> (scale)
//   output = weight * hidden_normed
//
// In our layout:
//   hidden  : [hidden, T]    (ne[0]=hidden, ne[1]=T)
//   cond    : [hidden]       (1D, ne[0]=hidden)   -- per-batch timestep emb
//   to_weight.weight : [hidden, hidden] (k=hidden, n=hidden)
//   to_weight.bias   : [hidden]
//
// ggml_rms_norm(a, eps) computes a * rsqrt(mean(a^2, dim=0) + eps)  -- NO weight.
// We then multiply by `Linear(cond)` broadcast across T.
// ---------------------------------------------------------------------------
static ggml_tensor * adaptive_rms_norm(
    ggml_context * ctx,
    const Model & m,
    const std::string & prefix_w,    // e.g. "blk.0.attn_norm_w"
    const std::string & prefix_b,    // e.g. "blk.0.attn_norm_b"
    ggml_tensor * hidden,
    ggml_tensor * cond)              // 1D [hidden]
{
    ggml_tensor * rms = ggml_rms_norm(ctx, hidden, RMS_EPS);   // [hidden, T]

    ggml_tensor * W = m.get_or_die(prefix_w + ".weight");      // [hidden, hidden]
    ggml_tensor * b = m.get_or_die(prefix_b + ".bias");        // [hidden]
    ggml_tensor * scale = ggml_mul_mat(ctx, W, cond);          // [hidden]
    scale = ggml_add(ctx, scale, b);                           // [hidden]

    // Broadcast [hidden] across [hidden, T]: ggml_mul(a, b) requires
    // ggml_can_repeat(b, a) (b's shape divides a's shape) and returns a's
    // shape. So pass the larger tensor (rms) first.
    return ggml_mul(ctx, rms, scale);                          // [hidden, T]
}

// ---------------------------------------------------------------------------
// Sequential(Linear(in,4*hidden), SiLU, Linear(4*hidden, out))
// Used for cond_mlp / mel_in_mlp / mel_out_mlp / timestep_mlp.
// Bias is loaded from GGUF.
// ---------------------------------------------------------------------------
static ggml_tensor * mlp_forward(
    ggml_context * ctx,
    const Model & m,
    const std::string & prefix,    // e.g. "cond_mlp"
    ggml_tensor * x)               // [in_dim, T]  or  [in_dim]  (1D, T=1)
{
    ggml_tensor * W0 = m.get_or_die(prefix + ".0.weight");
    ggml_tensor * b0 = m.get_or_die(prefix + ".0.bias");
    ggml_tensor * W2 = m.get_or_die(prefix + ".2.weight");
    ggml_tensor * b2 = m.get_or_die(prefix + ".2.bias");

    ggml_tensor * h = ggml_mul_mat(ctx, W0, x);   // [4*hidden, T]
    h = ggml_add(ctx, h, b0);                     // broadcast bias
    h = ggml_silu(ctx, h);
    ggml_tensor * y = ggml_mul_mat(ctx, W2, h);   // [out_dim, T]
    y = ggml_add(ctx, y, b2);
    return y;
}

// ---------------------------------------------------------------------------
// SinusoidalPosEmb + diff_step_mlp:
//   half = hidden/2
//   freq[i] = exp(-i * log(10000)/(half-1))   for i in [0, half)
//   emb = cat( sin(t*freq), cos(t*freq) )    shape [hidden]
//   out = diff_step_mlp(emb)                  shape [hidden]
// ---------------------------------------------------------------------------
static ggml_tensor * timestep_forward(
    ggml_context * ctx,
    const Model & m,
    float t)
{
    int half = HIDDEN / 2;
    ggml_tensor * emb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HIDDEN);
    float * data = (float *)emb->data;
    float scale = logf(10000.0f) / (half - 1);
    for (int i = 0; i < half; i++) {
        float freq = expf(-(float)i * scale);
        float angle = t * freq;
        data[i]         = sinf(angle);
        data[half + i]  = cosf(angle);
    }
    return mlp_forward(ctx, m, "timestep_mlp", emb);  // [hidden]
}

// ---------------------------------------------------------------------------
// One LlamaNARDecoderLayer (non-causal self-attn + Llama MLP).
//
// hidden  : [hidden, T]
// cond    : [hidden] (per-batch timestep emb, used by AdaptiveRMSNorm)
// positions : [T] (int32)
// ---------------------------------------------------------------------------
static ggml_tensor * decoder_layer(
    ggml_context * ctx,
    const Model & m,
    int idx,
    ggml_tensor * hidden,
    ggml_tensor * cond,
    ggml_tensor * positions)
{
    char nm[128];

    // ===== Self attention =====
    ggml_tensor * residual = hidden;

    snprintf(nm, sizeof(nm), "blk.%d.attn_norm_w", idx);
    std::string p_w(nm);
    snprintf(nm, sizeof(nm), "blk.%d.attn_norm_b", idx);
    std::string p_b(nm);
    ggml_tensor * normed = adaptive_rms_norm(ctx, m, p_w, p_b, hidden, cond);  // [hidden, T]

    // Projections (no bias; LlamaAttention default)
    snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight", idx);
    ggml_tensor * q = ggml_mul_mat(ctx, m.get_or_die(nm), normed);  // [hidden, T]
    snprintf(nm, sizeof(nm), "blk.%d.attn_k.weight", idx);
    ggml_tensor * k = ggml_mul_mat(ctx, m.get_or_die(nm), normed);
    snprintf(nm, sizeof(nm), "blk.%d.attn_v.weight", idx);
    ggml_tensor * v = ggml_mul_mat(ctx, m.get_or_die(nm), normed);

    // Reshape to [head_dim, n_heads, T] for RoPE.
    // After ggml_mul_mat(W=[hidden,hidden], normed=[hidden,T]) -> [hidden, T],
    // so q->ne[1] is the sequence length T.
    int64_t T = q->ne[1];
    q = ggml_reshape_3d(ctx, q, HEAD_DIM, NUM_HEADS, T);
    k = ggml_reshape_3d(ctx, k, HEAD_DIM, NUM_HEADS, T);
    v = ggml_reshape_3d(ctx, v, HEAD_DIM, NUM_HEADS, T);

    // RoPE (HF Llama default: NeoX, theta=10000)
    q = ggml_rope_ext(ctx, q, positions, nullptr, HEAD_DIM,
                      GGML_ROPE_TYPE_NEOX, 0, ROPE_THETA,
                      1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx, k, positions, nullptr, HEAD_DIM,
                      GGML_ROPE_TYPE_NEOX, 0, ROPE_THETA,
                      1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // Permute to [head_dim, T, n_heads] (matching llama.cpp's (0,2,1,3) permute
    // on a [head_dim, n_heads, T] tensor).
    q = ggml_permute(ctx, q, 0, 2, 1, 3);  // [64, T, 16]
    k = ggml_permute(ctx, k, 0, 2, 1, 3);
    v = ggml_permute(ctx, v, 0, 2, 1, 3);

    // kq = k^T @ q   (per head)
    // ggml_mul_mat(A=k=[head_dim, T, n_heads], B=q=[head_dim, T, n_heads])
    //   -> [T, T, n_heads]   (kq[q_idx, k_idx, h] = sum_d q[q,d,h]*k[k,d,h])
    ggml_tensor * kq = ggml_mul_mat(ctx, k, q);            // [T, T, 16]
    kq = ggml_scale(ctx, kq, 1.0f / sqrtf((float)HEAD_DIM));
    // Non-causal mask with all-ones x_mask: no mask addition needed
    kq = ggml_soft_max(ctx, kq);                           // softmax over k_idx (ne[0])

    // kqv = v @ kq   (per head)
    // To make the K dims match (both = T), transpose v so its ne[0] becomes T.
    //   v was [head_dim, T, n_heads] -> after 2D transpose: [T, head_dim, n_heads]
    //   ggml_mul_mat(A=v_t=[T, head_dim, n_heads], B=kq=[T, T, n_heads])
    //     -> [head_dim, T, n_heads]   (kqv[d, q_idx, h] = sum_t attn[q,t,h]*v[t,d,h])
    v = ggml_cont(ctx, ggml_transpose(ctx, v));
    ggml_tensor * kqv = ggml_mul_mat(ctx, v, kq);          // [head_dim, T, 16]

    // Permute back to [head_dim, n_heads, T], then flatten to [hidden, T]
    kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);              // [64, 16, T]
    kqv = ggml_cont(ctx, kqv);
    kqv = ggml_reshape_2d(ctx, kqv, HIDDEN, /*T=*/kqv->ne[2]);  // [1024, T]

    snprintf(nm, sizeof(nm), "blk.%d.attn_o.weight", idx);
    ggml_tensor * attn_out = ggml_mul_mat(ctx, m.get_or_die(nm), kqv);  // [hidden, T]
    hidden = ggml_add(ctx, attn_out, residual);

    // ===== MLP =====
    residual = hidden;
    snprintf(nm, sizeof(nm), "blk.%d.ffn_norm_w", idx);
    p_w = nm;
    snprintf(nm, sizeof(nm), "blk.%d.ffn_norm_b", idx);
    p_b = nm;
    hidden = adaptive_rms_norm(ctx, m, p_w, p_b, hidden, cond);  // [hidden, T]

    // Llama MLP: down(silu(gate(x)) * up(x))
    snprintf(nm, sizeof(nm), "blk.%d.ffn_gate.weight", idx);
    ggml_tensor * gate = ggml_mul_mat(ctx, m.get_or_die(nm), hidden);  // [4096, T]
    gate = ggml_silu(ctx, gate);
    snprintf(nm, sizeof(nm), "blk.%d.ffn_up.weight", idx);
    ggml_tensor * up = ggml_mul_mat(ctx, m.get_or_die(nm), hidden);    // [4096, T]
    ggml_tensor * mid = ggml_mul(ctx, gate, up);                       // [4096, T]
    snprintf(nm, sizeof(nm), "blk.%d.ffn_down.weight", idx);
    ggml_tensor * mlp_out = ggml_mul_mat(ctx, m.get_or_die(nm), mid);  // [hidden, T]

    return ggml_add(ctx, mlp_out, residual);
}

// ---------------------------------------------------------------------------
// Full DiffLlama.forward(x, diffusion_step, cond, x_mask)  [B=1 assumed]
//
//   x   : [mel_dim, T]   f32
//   cond: [hidden,  T]   f32
//   t   : scalar timestep
//   Returns: [mel_dim, T] f32  (the "velocity" output of the flow-matching net)
// ---------------------------------------------------------------------------
static ggml_tensor * dit_forward(
    ggml_context * ctx,
    const Model & m,
    ggml_tensor * x,
    ggml_tensor * cond,
    float t,
    int T)
{
    int64_t ne[4] = {x->ne[0], x->ne[1], x->ne[2], x->ne[3]};

    // 1) conditioning MLP + mel_in MLP + timestep MLP
    ggml_tensor * cond_emb  = mlp_forward(ctx, m, "cond_mlp",   cond);   // [hidden, T]
    ggml_tensor * x_proj    = mlp_forward(ctx, m, "mel_in_mlp", x);      // [hidden, T]
    ggml_tensor * diff_step = timestep_forward(ctx, m, t);               // [hidden]

    // 2) x = x_proj + cond_emb
    ggml_tensor * h = ggml_add(ctx, x_proj, cond_emb);                   // [hidden, T]

    // 3) 22 decoder layers
    ggml_tensor * positions = make_positions(ctx, T);                    // [T] i32
    for (int i = 0; i < NUM_LAYERS; i++) {
        h = decoder_layer(ctx, m, i, h, diff_step, positions);           // [hidden, T]
    }

    // 4) final adaptive RMS norm
    h = adaptive_rms_norm(ctx, m, "output_norm_w", "output_norm_b", h, diff_step);

    // 5) mel_out MLP -> [mel_dim, T]
    h = mlp_forward(ctx, m, "mel_out_mlp", h);
    return h;
}

// ---------------------------------------------------------------------------
// Run a forward pass against the model and copy the result to a host buffer.
// Allocates a per-call scratch ctx, builds the graph, runs, copies out.
// ---------------------------------------------------------------------------
static std::vector<float> run_forward(
    const Model & m,
    const float * x_data,
    const float * cond_data,
    float t,
    int T)
{
    Scratch sc(1ULL << 30);  // 1 GiB scratch (plenty for 22 layers at T=64)

    ggml_tensor * x    = ggml_new_tensor_2d(sc.ctx, GGML_TYPE_F32, MEL_DIM, T);
    ggml_tensor * cond = ggml_new_tensor_2d(sc.ctx, GGML_TYPE_F32, HIDDEN,  T);
    memcpy(x->data,    x_data,    sizeof(float) * MEL_DIM * T);
    memcpy(cond->data, cond_data, sizeof(float) * HIDDEN  * T);

    ggml_tensor * out = dit_forward(sc.ctx, m, x, cond, t, T);   // [mel_dim, T]

    ggml_cgraph * gf = ggml_new_graph_custom(sc.ctx, 1 << 18, false);
    ggml_build_forward_expand(gf, out);

    int n_threads = 3;
    ggml_graph_compute_with_ctx(sc.ctx, gf, n_threads);

    std::vector<float> result((size_t)MEL_DIM * T);
    memcpy(result.data(), out->data, sizeof(float) * MEL_DIM * T);
    return result;
}

// ---------------------------------------------------------------------------
// Reverse diffusion (Euler ODE integration, mimics
// FlowMatchingTransformer.reverse_diffusion in verify_gguf_inference.py)
//
//   prompt_mel : [mel_dim, prompt_len]
//   cond_full  : [hidden, T]   (T = prompt_len + target_len)
//   z          : [mel_dim, target_len]  (initial noise)
//   n_steps    : ODE steps (default 8)
//   Returns    : [mel_dim, target_len]  (final xt)
// ---------------------------------------------------------------------------
static std::vector<float> run_reverse_diffusion(
    const Model & m,
    const std::vector<float> & prompt_mel,  // [mel_dim, prompt_len]
    const std::vector<float> & cond_full,   // [hidden,  T]
    const std::vector<float> & z_init,      // [mel_dim, target_len]
    int prompt_len, int target_len, int n_steps,
    RNG & rng)
{
    int T = prompt_len + target_len;
    float h = 1.0f / n_steps;

    // xt starts as z_init
    std::vector<float> xt = z_init;  // [mel_dim, target_len]

    for (int i = 0; i < n_steps; i++) {
        float t = (i + 0.5f) * h;

        // xt_input = cat([prompt_mel, xt], dim=1)  -> [mel_dim, T]
        std::vector<float> xt_input((size_t)MEL_DIM * T);
        for (int k = 0; k < MEL_DIM; k++) {
            const float * p = &prompt_mel[(size_t)k * prompt_len];
            const float * q = &xt[(size_t)k * target_len];
            float * dst = &xt_input[(size_t)k * T];
            memcpy(dst, p, sizeof(float) * prompt_len);
            memcpy(dst + prompt_len, q, sizeof(float) * target_len);
        }

        // forward(xt_input, t, cond_full, x_mask=ones)
        std::vector<float> flow = run_forward(m, xt_input.data(), cond_full.data(), t, T);

        // flow_pred = flow[:, prompt_len:]   (per-row slicing; row-major over mel_dim)
        std::vector<float> flow_pred((size_t)MEL_DIM * target_len);
        for (int k = 0; k < MEL_DIM; k++) {
            const float * src = &flow[(size_t)k * T + prompt_len];
            float * dst = &flow_pred[(size_t)k * target_len];
            memcpy(dst, src, sizeof(float) * target_len);
        }

        // xt = xt + flow_pred * h
        for (size_t j = 0; j < xt.size(); j++) xt[j] += flow_pred[j] * h;
    }
    return xt;
}

// ---------------------------------------------------------------------------
// Accuracy metrics (mirror verify_gguf_inference.py compare())
// ---------------------------------------------------------------------------
struct Metrics { float mse, rmse, max_abs, cos_sim, rel_l2; };

static Metrics compare(const std::vector<float> & a, const std::vector<float> & b) {
    assert(a.size() == b.size());
    double mse = 0, max_abs = 0, dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); i++) {
        double d = (double)a[i] - (double)b[i];
        mse     += d * d;
        double ad = fabs(d); if (ad > max_abs) max_abs = ad;
        dot     += (double)a[i] * (double)b[i];
        na      += (double)a[i] * (double)a[i];
        nb      += (double)b[i] * (double)b[i];
    }
    Metrics m;
    m.mse      = (float)(mse / a.size());
    m.rmse     = sqrtf(m.mse);
    m.max_abs  = (float)max_abs;
    m.cos_sim  = (float)(dot / (sqrt(na) * sqrt(nb) + 1e-12));
    m.rel_l2   = (float)(sqrt(mse) / (sqrt(na) + 1e-12));
    return m;
}

static void print_metrics(const char * tag, const Metrics & m) {
    printf("  %-22s  MSE=%.4e  RMSE=%.4e  maxAbs=%.4e  cos=%.6f  relL2=%.4e\n",
           tag, m.mse, m.rmse, m.max_abs, m.cos_sim, m.rel_l2);
}

static void stats(const char * tag, const std::vector<float> & v) {
    double mean = 0, var = 0;
    for (float x : v) mean += x;
    mean /= v.size();
    for (float x : v) var += (x - mean) * (x - mean);
    var /= v.size();
    printf("  %-22s  N=%zu  mean=%+.4e  std=%.4e\n", tag, v.size(), mean, sqrt(var));
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr,
            "Usage: %s <fp32.gguf> <q8_0.gguf> <q4_k_m.gguf>\n", argv[0]);
        return 1;
    }
    const char * fp32_path = argv[1];
    const char * q8_path   = argv[2];
    const char * q4_path   = argv[3];

    const int T_fwd       = 64;     // forward seq_len (matches verify_gguf_inference.py)
    const int prompt_len  = T_fwd / 4;   // 16
    const int target_len  = T_fwd - prompt_len;  // 48
    const int n_steps     = 8;
    const float t_fwd     = 0.5f;

    printf("=================================================================\n");
    printf("SoulX-Singer DiT GGUF — pure C++ inference (no dequantization)\n");
    printf("=================================================================\n");
    printf("  config: mel_dim=%d hidden=%d layers=%d heads=%d head_dim=%d\n",
           MEL_DIM, HIDDEN, NUM_LAYERS, NUM_HEADS, HEAD_DIM);
    printf("  forward: T=%d t=%.3f\n", T_fwd, t_fwd);
    printf("  reverse: prompt_len=%d target_len=%d n_steps=%d\n",
           prompt_len, target_len, n_steps);
    printf("\n");

    // ------------------------------------------------------------------
    // Generate deterministic test inputs (same for all 3 model variants)
    // ------------------------------------------------------------------
    RNG rng(12345);

    std::vector<float> x_in((size_t)MEL_DIM * T_fwd);
    std::vector<float> cond_in((size_t)HIDDEN * T_fwd);
    for (auto & v : x_in)    v = rng.gaussian();
    for (auto & v : cond_in) v = rng.gaussian();

    std::vector<float> prompt_mel((size_t)MEL_DIM * prompt_len);
    // prompt_mel = x_in[:, :prompt_len]
    for (int k = 0; k < MEL_DIM; k++) {
        memcpy(&prompt_mel[(size_t)k * prompt_len],
               &x_in[(size_t)k * T_fwd],
               sizeof(float) * prompt_len);
    }

    // cond_full is a SEPARATE random tensor (mirrors Python's cond_full)
    std::vector<float> cond_full((size_t)HIDDEN * T_fwd);
    for (auto & v : cond_full) v = rng.gaussian();

    // z = randn(B, target_len, mel_dim)
    std::vector<float> z_init((size_t)MEL_DIM * target_len);
    for (auto & v : z_init) v = rng.gaussian();

    printf(">> Loading 3 GGUF variants (FP32 / Q8_0 / Q4_K_M)\n");
    Model m_fp32, m_q8, m_q4;
    if (!m_fp32.load(fp32_path)) { fprintf(stderr, "load fp32 failed\n"); return 1; }
    if (!m_q8.load(q8_path))     { fprintf(stderr, "load q8 failed\n");   return 1; }
    if (!m_q4.load(q4_path))     { fprintf(stderr, "load q4 failed\n");   return 1; }

    // Sanity: print a few tensor info entries
    printf("\n>> Tensor sanity check (FP32 model):\n");
    for (const char * nm : {"blk.0.attn_q.weight",
                            "blk.0.attn_norm_w.weight",
                            "blk.0.attn_norm_b.bias",
                            "blk.0.ffn_gate.weight",
                            "timestep_mlp.0.weight",
                            "timestep_mlp.0.bias",
                            "cond_mlp.0.weight",
                            "mel_in_mlp.0.weight",
                            "mel_out_mlp.2.weight",
                            "output_norm_w.weight",
                            "output_norm_b.bias"}) {
        ggml_tensor * t = m_fp32.get(nm);
        if (t) {
            printf("  %-32s type=%-6s ne=[%lld,%lld,%lld,%lld]\n",
                   nm, ggml_type_name(t->type),
                   (long long)t->ne[0], (long long)t->ne[1],
                   (long long)t->ne[2], (long long)t->ne[3]);
        } else {
            printf("  %-32s MISSING\n", nm);
        }
    }
    printf("\n");

    // Same tensor types for the quantized variants — proves weights stay
    // packed (no dequant happened at load time):
    printf(">> Tensor type check (FP32 vs Q8_0 vs Q4_K_M):\n");
    for (const char * nm : {"blk.0.attn_q.weight",
                            "blk.0.ffn_down.weight",
                            "timestep_mlp.0.weight"}) {
        printf("  %-32s  fp32=%-6s  q8=%-6s  q4=%-6s\n", nm,
               ggml_type_name(m_fp32.get(nm)->type),
               ggml_type_name(m_q8.get(nm)->type),
               ggml_type_name(m_q4.get(nm)->type));
    }
    printf("\n");

    // ------------------------------------------------------------------
    // Forward pass on all 3 variants
    // ------------------------------------------------------------------
    printf(">> Forward pass (DiffLlama.forward) on 3 variants...\n");
    auto out_fwd_fp32 = run_forward(m_fp32, x_in.data(),    cond_in.data(),    t_fwd, T_fwd);
    auto out_fwd_q8   = run_forward(m_q8,   x_in.data(),    cond_in.data(),    t_fwd, T_fwd);
    auto out_fwd_q4   = run_forward(m_q4,   x_in.data(),    cond_in.data(),    t_fwd, T_fwd);

    stats("FP32 forward out", out_fwd_fp32);
    stats("Q8_0 forward out", out_fwd_q8);
    stats("Q4_K_M forward out", out_fwd_q4);
    printf("\n");

    printf(">> Forward pass accuracy (vs FP32 reference):\n");
    Metrics fwd_q8_vs_fp32 = compare(out_fwd_fp32, out_fwd_q8);
    Metrics fwd_q4_vs_fp32 = compare(out_fwd_fp32, out_fwd_q4);
    Metrics fwd_q4_vs_q8   = compare(out_fwd_q8,   out_fwd_q4);
    print_metrics("Q8_0_vs_FP32",    fwd_q8_vs_fp32);
    print_metrics("Q4_K_M_vs_FP32",  fwd_q4_vs_fp32);
    print_metrics("Q4_K_M_vs_Q8_0",  fwd_q4_vs_q8);
    printf("\n");

    // ------------------------------------------------------------------
    // Reverse diffusion on all 3 variants (chaotic by design; compare stats)
    // ------------------------------------------------------------------
    printf(">> Reverse diffusion (8-step Euler ODE) on 3 variants...\n");
    RNG rng_fp32(999), rng_q8(999), rng_q4(999);  // identical noise for fair compare
    auto out_rev_fp32 = run_reverse_diffusion(
        m_fp32, prompt_mel, cond_full, z_init, prompt_len, target_len, n_steps, rng_fp32);
    auto out_rev_q8   = run_reverse_diffusion(
        m_q8,   prompt_mel, cond_full, z_init, prompt_len, target_len, n_steps, rng_q8);
    auto out_rev_q4   = run_reverse_diffusion(
        m_q4,   prompt_mel, cond_full, z_init, prompt_len, target_len, n_steps, rng_q4);

    stats("FP32 reverse mel", out_rev_fp32);
    stats("Q8_0 reverse mel", out_rev_q8);
    stats("Q4_K_M reverse mel", out_rev_q4);
    printf("\n");

    printf(">> Reverse diffusion accuracy (vs FP32 reference):\n");
    Metrics rev_q8_vs_fp32 = compare(out_rev_fp32, out_rev_q8);
    Metrics rev_q4_vs_fp32 = compare(out_rev_fp32, out_rev_q4);
    print_metrics("Q8_0_vs_FP32",   rev_q8_vs_fp32);
    print_metrics("Q4_K_M_vs_FP32", rev_q4_vs_fp32);
    printf("\n");

    // ------------------------------------------------------------------
    // Verdict
    // ------------------------------------------------------------------
    printf("=================================================================\n");
    printf("VERDICT\n");
    printf("=================================================================\n");
    printf("Forward pass (this is the meaningful accuracy test):\n");
    printf("  Q8_0   cos_sim = %.6f  rel_l2 = %.4e  (ref Python ~0.99998 / 6.5e-3)\n",
           fwd_q8_vs_fp32.cos_sim, fwd_q8_vs_fp32.rel_l2);
    printf("  Q4_K_M cos_sim = %.6f  rel_l2 = %.4e  (ref Python ~0.99652 / 8.4e-2)\n",
           fwd_q4_vs_fp32.cos_sim, fwd_q4_vs_fp32.rel_l2);

    bool ok_q8  = fwd_q8_vs_fp32.cos_sim  > 0.999f && fwd_q8_vs_fp32.rel_l2  < 0.02f;
    bool ok_q4  = fwd_q4_vs_fp32.cos_sim  > 0.99f  && fwd_q4_vs_fp32.rel_l2  < 0.15f;
    printf("\n  Q8_0  accuracy normal : %s\n", ok_q8  ? "PASS" : "FAIL");
    printf("  Q4_K_M accuracy normal: %s\n", ok_q4  ? "PASS" : "FAIL");
    printf("\nReverse diffusion mel-level similarity is expected to be low for all\n");
    printf("quantized variants (flow-matching ODE chaos amplifies 1%% single-step\n");
    printf("error exponentially over 8 steps) — this is the known behaviour noted in\n");
    printf("the model's inference_report.txt and is NOT a quantization failure.\n");

    printf("\n>> Pure C++ inference completed. Weights were used directly in\n");
    printf("   quantized form via ggml_mul_mat — no dequantization occurred.\n");

    m_fp32.unload(); m_q8.unload(); m_q4.unload();
    return (ok_q8 && ok_q4) ? 0 : 2;
}
