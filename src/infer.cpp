// infer.cpp — Pure C++ inference for SoulX-Singer DiT (DiffLlama) GGUF
//
// Library-only build (no main()). The public API is declared in infer.h and
// consumed by binding.cc (N-API). All inference logic is preserved from the
// original infer_orig.cpp; only the CLI test harness (main + metrics helpers)
// has been removed and run_forward / run_reverse_diffusion now have external
// linkage so the binding can call them.
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

#include "infer.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Model::load / Model::unload
// ---------------------------------------------------------------------------
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
// (External linkage — called from binding.cc.)
// ---------------------------------------------------------------------------
std::vector<float> run_forward(
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
// (External linkage — called from binding.cc.)
// ---------------------------------------------------------------------------
std::vector<float> run_reverse_diffusion(
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
