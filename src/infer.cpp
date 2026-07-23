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

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
  #include <windows.h>
#else
  #include <sys/mman.h>
  #include <sys/stat.h>
  #include <fcntl.h>
  #include <unistd.h>
#endif

// ---------------------------------------------------------------------------
// runtime_thread_count — pick the CPU worker thread count at runtime.
//
// Replaces the old hardcoded n_threads=3. We start from
// std::thread::hardware_concurrency() and then:
//   1. Cap to the cgroup CPU quota (cgroup v2 cpu.max or v1 cfs_quota/period).
//      This is essential in containers, where hardware_concurrency() sees the
//      whole host but the container is only allowed e.g. 2 cores.
//   2. Best-effort cap to the physical core count (Linux /proc/cpuinfo) to
//      avoid SMT oversubscription, which hurts matmul throughput.
// Never returns less than 1.
// ---------------------------------------------------------------------------
int runtime_thread_count() {
    unsigned hw = std::thread::hardware_concurrency();
    int n = hw > 0 ? (int)hw : 1;

#ifdef __linux__
    // --- cgroup v2: /sys/fs/cgroup/cpu.max  ->  "MAX PERIOD"  (MAX = "max") ---
    {
        std::ifstream f("/sys/fs/cgroup/cpu.max");
        std::string line;
        if (f && std::getline(f, line)) {
            std::istringstream iss(line);
            std::string a, b;
            if (iss >> a >> b) {
                if (a != "max") {
                    try {
                        long long mx = std::stoll(a);
                        long long pe = std::stoll(b);
                        if (pe > 0 && mx > 0) {
                            long q = (long)((mx + pe - 1) / pe);  // ceil(mx/pe)
                            if (q > 0 && q < n) n = (int)q;
                        }
                    } catch (...) {}
                }
            }
        }
    }
    // --- cgroup v1 fallback ---
    {
        std::ifstream fq("/sys/fs/cgroup/cpu/cpu.cfs_quota_us");
        std::ifstream fp("/sys/fs/cgroup/cpu/cpu.cfs_period_us");
        long long q = -1, pe = -1;
        if (fq) fq >> q;
        if (fp) fp >> pe;
        if (q > 0 && pe > 0) {
            long qc = (q + pe - 1) / pe;
            if (qc > 0 && qc < n) n = (int)qc;
        }
    }

    // --- physical core cap (avoid SMT oversubscription) ---
    {
        std::ifstream f("/proc/cpuinfo");
        std::string line;
        int cpu_cores = -1;                 // "cpu cores" (per package)
        std::set<int> packages;
        while (std::getline(f, line)) {
            if (line.compare(0, 10, "cpu cores") == 0) {
                auto colon = line.find(':');
                if (colon != std::string::npos) {
                    try { cpu_cores = std::stoi(line.substr(colon + 1)); } catch (...) {}
                }
            } else if (line.compare(0, 11, "physical id") == 0) {
                auto colon = line.find(':');
                if (colon != std::string::npos) {
                    try { packages.insert(std::stoi(line.substr(colon + 1))); } catch (...) {}
                }
            }
        }
        if (cpu_cores > 0 && !packages.empty()) {
            int phys = cpu_cores * (int)packages.size();
            if (phys > 0 && phys < n) n = phys;
        }
    }
#endif
    if (n < 1) n = 1;
    return n;
}

// ---------------------------------------------------------------------------
// pick_backend — enumerate ggml backend devices compiled into this .node and
// select the best one. Priority: dedicated GPU (CUDA/HIP/Metal/SYCL) > Vulkan
// > CPU.
//
//   pref values:
//     "" / "auto"          auto-select (prefer GPU)
//     "gpu" / "cuda" ...   require a GPU (fall back to CPU if none)
//     "cpu"                force the CPU path
//
// For the CPU choice we return a null backend handle (out_is_gpu=false): the
// CPU compute path uses ggml_graph_compute_with_ctx() and the mmap'd weights
// directly, so no explicit backend object is needed. For a GPU choice we
// return an initialised ggml_backend_t that the caller frees with
// ggml_backend_free().
// ---------------------------------------------------------------------------
ggml_backend_t pick_backend(const std::string & pref, bool & out_is_gpu,
                            std::string & out_name) {
    out_is_gpu = false;
    out_name = "cpu";

    std::string p = pref;
    for (auto & c : p) c = (char)std::tolower((unsigned char)c);

    bool force_cpu = (p == "cpu");
    bool want_gpu  = (p.empty() || p == "auto" || p == "gpu" || p == "cuda" ||
                      p == "hip" || p == "metal" || p == "vulkan" || p == "sycl");

    int n = (int)ggml_backend_dev_count();
    ggml_backend_dev_t dev_gpu = nullptr;      // dedicated GPU
    ggml_backend_dev_t dev_vulkan = nullptr;   // Vulkan (cross-vendor fallback)
    for (int i = 0; i < n; i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (!d) continue;
        enum ggml_backend_dev_type dt = ggml_backend_dev_type(d);
        const char * nm = ggml_backend_dev_name(d);
        std::string sname(nm ? nm : "");
        for (auto & c : sname) c = (char)std::tolower((unsigned char)c);
        if (dt == GGML_BACKEND_DEVICE_TYPE_GPU) {
            if (sname.find("vulkan") != std::string::npos) {
                if (!dev_vulkan) dev_vulkan = d;
            } else {
                if (!dev_gpu) dev_gpu = d;     // first dedicated GPU wins
            }
        }
    }

    ggml_backend_dev_t chosen = nullptr;
    if (!force_cpu && want_gpu) {
        if (dev_gpu) chosen = dev_gpu;
        else if (dev_vulkan) chosen = dev_vulkan;
    }

    if (!chosen || ggml_backend_dev_type(chosen) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        // CPU path — no explicit backend object required.
        out_is_gpu = false;
        out_name = "cpu";
        return nullptr;
    }

    out_is_gpu = true;
    const char * nm = ggml_backend_dev_name(chosen);
    out_name = nm ? nm : "gpu";

    ggml_backend_t b = ggml_backend_dev_init(chosen, nullptr);
    if (!b) {
        fprintf(stderr, "WARN: ggml_backend_dev_init('%s') failed; falling back to CPU\n",
                out_name.c_str());
        out_is_gpu = false;
        out_name = "cpu";
        return nullptr;
    }
    return b;
}

// ---------------------------------------------------------------------------
// Model::load / Model::unload
// ---------------------------------------------------------------------------
bool Model::load(const std::string & path_, const std::string & backend_pref) {
    path = path_;

    // One-shot NUMA initialisation (no-op on non-NUMA / non-Linux builds).
    // Distributes worker threads across NUMA nodes so multi-socket boxes do
    // not pay cross-socket memory latency on AMX/AVX512 matmuls.
    static bool g_numa_inited = false;
    if (!g_numa_inited) {
        ggml_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);
        g_numa_inited = true;
    }

#ifdef _WIN32
    HANDLE hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "CreateFileA failed for %s (err=%lu)\n", path.c_str(), GetLastError());
        return false;
    }
    LARGE_INTEGER fsz;
    if (!GetFileSizeEx(hFile, &fsz)) {
        fprintf(stderr, "GetFileSizeEx failed for %s (err=%lu)\n", path.c_str(), GetLastError());
        CloseHandle(hFile);
        return false;
    }
    mmap_size = (size_t)fsz.QuadPart;
    file_mapping = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    CloseHandle(hFile);
    if (!file_mapping) {
        fprintf(stderr, "CreateFileMappingA failed for %s (err=%lu)\n", path.c_str(), GetLastError());
        return false;
    }
    mmap_data = MapViewOfFile(file_mapping, FILE_MAP_READ, 0, 0, 0);
    if (!mmap_data) {
        fprintf(stderr, "MapViewOfFile failed for %s (err=%lu)\n", path.c_str(), GetLastError());
        CloseHandle(file_mapping);
        file_mapping = nullptr;
        return false;
    }
#else
    fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { perror("open"); return false; }
    struct stat sb;
    if (fstat(fd, &sb) != 0) { perror("fstat"); return false; }
    mmap_size = (size_t)sb.st_size;
    mmap_data = mmap(nullptr, mmap_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mmap_data == MAP_FAILED) { perror("mmap"); mmap_data = nullptr; return false; }
#endif

    gguf_init_params params;
    params.no_alloc = true;             // we'll fill tensor->data ourselves
    params.ctx      = &ctx;
    gguf = gguf_init_from_file(path.c_str(), params);
    if (!gguf) {
        fprintf(stderr, "gguf_init_from_file failed for %s\n", path.c_str());
        return false;
    }

    // ---- Resolve the runtime backend + thread count ----
    bool is_gpu = false;
    std::string bname;
    backend = pick_backend(backend_pref, is_gpu, bname);
    use_gpu = is_gpu;
    backend_name = bname;
    n_threads = runtime_thread_count();

    size_t data_offset = gguf_get_data_offset(gguf);
    int n_tensors = (int)gguf_get_n_tensors(gguf);

    if (use_gpu && backend) {
        // ---- GPU: upload weights into the backend (VRAM) buffer ----
        buft = ggml_backend_get_default_buffer_type(backend);
        weight_buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!weight_buf) {
            fprintf(stderr, "WARN: backend alloc_ctx_tensors failed; "
                            "falling back to CPU mmap path\n");
            ggml_backend_free(backend);
            backend = nullptr; buft = nullptr; use_gpu = false; backend_name = "cpu";
        } else {
            for (int i = 0; i < n_tensors; i++) {
                const char * name = gguf_get_tensor_name(gguf, i);
                ggml_tensor * t = ggml_get_tensor(ctx, name);
                if (!t) continue;
                size_t off = gguf_get_tensor_offset(gguf, i);
                const void * src = (const char *)mmap_data + data_offset + off;
                ggml_backend_tensor_set(t, src, 0, ggml_nbytes(t));
            }
            // Weights now live in the device buffer; release the host mmap.
#ifdef _WIN32
            if (mmap_data) { UnmapViewOfFile(mmap_data); mmap_data = nullptr; }
            if (file_mapping) { CloseHandle(file_mapping); file_mapping = nullptr; }
#else
            if (mmap_data) { munmap(mmap_data, mmap_size); mmap_data = nullptr; }
            if (fd >= 0) { close(fd); fd = -1; }
#endif
            printf("  loaded %s on GPU '%s': %d tensors uploaded (%.1f MiB), %d cpu threads\n",
                   path.c_str(), bname.c_str(), n_tensors,
                   (double)mmap_size / (1 << 20), n_threads);
            return true;
        }
    }

    // ---- CPU: zero-copy mmap. Point tensor->data straight at the mmap region
    // so ggml-cpu's quantized mul_mat kernels read packed weights directly,
    // with no dequantization and no extra copy. AMX/AVX512/AVX2 kernel
    // selection happens inside ggml-cpu via ggml_cpu_has_*() at runtime. ----
    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf, i);
        ggml_tensor * t = ggml_get_tensor(ctx, name);
        if (!t) continue;
        size_t off = gguf_get_tensor_offset(gguf, i);
        t->data = (char *)mmap_data + data_offset + off;
    }

    printf("  loaded %s on CPU (mmap zero-copy): %d tensors, %.1f MiB, "
           "%d threads, numa=%d, amx_int8=%d\n",
           path.c_str(), n_tensors, (double)mmap_size / (1 << 20),
           n_threads, (int)ggml_is_numa(), (int)ggml_cpu_has_amx_int8());
    return true;
}

// Model::unload() is defined later, after GraphCache (which it must destroy
// with a complete type — see the GraphCache struct below).

// ---------------------------------------------------------------------------
// Per-call scratch context: holds all intermediate tensors of one forward
// pass.
//
//   no_alloc=false (CPU): the host `buf` backs both tensor metadata and tensor
//     data; ggml_graph_compute_with_ctx() computes in place. Leaf inputs are
//     flushed by copying into t->data.
//   no_alloc=true  (GPU): `buf` holds only metadata. After the graph is built,
//     ggml_backend_alloc_ctx_tensors() allocates the intermediates in the
//     device buffer; leaf inputs are flushed with ggml_backend_tensor_set().
//
// Leaf input tensors (x, cond, positions, timestep embedding) are "staged" as
// host-owned byte buffers and flushed once after allocation, so the same graph
// builders work for both the CPU and GPU compute paths.
// ---------------------------------------------------------------------------
struct Scratch {
    ggml_context * ctx;
    std::vector<uint8_t> buf;
    bool no_alloc;
    struct Leaf { ggml_tensor * t; std::vector<uint8_t> data; };
    std::vector<Leaf> leaves;

    Scratch(size_t bytes, bool no_alloc_) : buf(bytes), no_alloc(no_alloc_) {
        ggml_init_params p; p.mem_size = bytes; p.mem_buffer = buf.data(); p.no_alloc = no_alloc;
        ctx = ggml_init(p);
    }
    ~Scratch() { if (ctx) ggml_free(ctx); }

    void stage_leaf(ggml_tensor * t, const void * data, size_t nbytes) {
        Leaf l; l.t = t;
        l.data.assign((const uint8_t *)data, (const uint8_t *)data + nbytes);
        leaves.push_back(std::move(l));
    }
};

// ---------------------------------------------------------------------------
// Build a 1D int32 positions tensor [0, 1, ..., T-1] (staged as a leaf).
// `out_leaf` (if non-null) receives the leaf tensor pointer so callers that
// cache the graph can refresh its contents later.
// ---------------------------------------------------------------------------
static ggml_tensor * make_positions(Scratch & sc, int T,
                                    ggml_tensor ** out_leaf = nullptr) {
    ggml_tensor * p = ggml_new_tensor_1d(sc.ctx, GGML_TYPE_I32, T);
    std::vector<int32_t> vals((size_t)T);
    for (int i = 0; i < T; i++) vals[i] = i;
    sc.stage_leaf(p, vals.data(), sizeof(int32_t) * T);
    if (out_leaf) *out_leaf = p;
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
//
// The sinusoidal embedding is a leaf input produced on the host. We stage it
// via Scratch::stage_leaf() so the GPU compute path can upload it to the
// device buffer; on the CPU path it is memcpy'd into the scratch buffer.
// ---------------------------------------------------------------------------

// Build the sinusoidal timestep embedding [HIDDEN] on the host. Extracted so
// the graph-cache path can recompute it for a new `t` and write it back into
// the existing leaf tensor without rebuilding the graph.
static std::vector<float> build_timestep_emb(float t) {
    int half = HIDDEN / 2;
    std::vector<float> data((size_t)HIDDEN);
    float scale = logf(10000.0f) / (half - 1);
    for (int i = 0; i < half; i++) {
        float freq = expf(-(float)i * scale);
        float angle = t * freq;
        data[i]        = sinf(angle);
        data[half + i] = cosf(angle);
    }
    return data;
}

static ggml_tensor * timestep_forward(
    Scratch & sc,
    const Model & m,
    float t,
    ggml_tensor ** out_emb_leaf = nullptr)
{
    ggml_tensor * emb = ggml_new_tensor_1d(sc.ctx, GGML_TYPE_F32, HIDDEN);

    // Build the sinusoidal embedding in a host buffer, then stage it. This
    // works for both CPU (no_alloc=false) and GPU (no_alloc=true) scratch
    // contexts: on the GPU path emb->data is null until alloc_ctx_tensors.
    std::vector<float> data = build_timestep_emb(t);
    sc.stage_leaf(emb, data.data(), sizeof(float) * HIDDEN);
    if (out_emb_leaf) *out_emb_leaf = emb;

    return mlp_forward(sc.ctx, m, "timestep_mlp", emb);  // [hidden]
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
    ggml_tensor * positions,
    bool use_flash_attn)
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

    ggml_tensor * kqv;
    if (use_flash_attn) {
        // Flash Attention (plan item #9): fused Q*K^T -> softmax -> *V in a
        // single kernel. Reduces memory bandwidth (no materialised [T,T] KQ
        // matrix) and kernel-launch count — especially beneficial under CUDA
        // Graph capture where each launch is a graph node.
        //
        // ggml_flash_attn_ext expects q=[d,T,H,1], k/v=[d,T,H,1] (which we
        // have after the permute above) and returns [d,H,T,1]. K/V should be
        // F16 on GPU for the optimised kernel; we follow llama.cpp's pattern
        // and cast regardless of backend.
        k = ggml_cast(ctx, k, GGML_TYPE_F16);
        v = ggml_cast(ctx, v, GGML_TYPE_F16);

        float attn_scale = 1.0f / sqrtf((float)HEAD_DIM);
        kqv = ggml_flash_attn_ext(ctx, q, k, v,
                                  /*mask=*/nullptr, attn_scale,
                                  /*max_bias=*/0.0f, /*logit_softcap=*/0.0f);
        ggml_flash_attn_ext_set_prec(kqv, GGML_PREC_F32);

        // kqv is [HEAD_DIM, NUM_HEADS, T, 1] — already in the layout we need
        // for the output projection; just flatten to [HIDDEN, T].
        kqv = ggml_cont(ctx, kqv);
        kqv = ggml_reshape_2d(ctx, kqv, HIDDEN, T);
    } else {
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
        kqv = ggml_mul_mat(ctx, v, kq);          // [head_dim, T, 16]

        // Permute back to [head_dim, n_heads, T], then flatten to [hidden, T]
        kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);              // [64, 16, T]
        kqv = ggml_cont(ctx, kqv);
        kqv = ggml_reshape_2d(ctx, kqv, HIDDEN, /*T=*/kqv->ne[2]);  // [1024, T]
    }

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
//
// `use_flash_attn` selects the fused Flash Attention kernel (plan item #9)
// instead of the manual mul_mat+soft_max+mul_mat path. Enabled on GPU where
// the optimised CUDA kernel is available.
//
// `out_emb_leaf` / `out_pos_leaf` (if non-null) receive the leaf tensor
// pointers for the timestep embedding and positions so the caller can refresh
// their contents without rebuilding the graph (plan item #8, graph cache).
// ---------------------------------------------------------------------------
static ggml_tensor * dit_forward(
    Scratch & sc,
    const Model & m,
    ggml_tensor * x,
    ggml_tensor * cond,
    float t,
    int T,
    bool use_flash_attn = false,
    ggml_tensor ** out_emb_leaf = nullptr,
    ggml_tensor ** out_pos_leaf = nullptr)
{
    ggml_context * ctx = sc.ctx;

    // 1) conditioning MLP + mel_in MLP + timestep MLP
    ggml_tensor * cond_emb  = mlp_forward(ctx, m, "cond_mlp",   cond);   // [hidden, T]
    ggml_tensor * x_proj    = mlp_forward(ctx, m, "mel_in_mlp", x);      // [hidden, T]
    ggml_tensor * diff_step = timestep_forward(sc, m, t, out_emb_leaf);  // [hidden]

    // 2) x = x_proj + cond_emb
    ggml_tensor * h = ggml_add(ctx, x_proj, cond_emb);                   // [hidden, T]

    // 3) 22 decoder layers
    ggml_tensor * positions = make_positions(sc, T, out_pos_leaf);       // [T] i32
    for (int i = 0; i < NUM_LAYERS; i++) {
        h = decoder_layer(ctx, m, i, h, diff_step, positions, use_flash_attn);  // [hidden, T]
    }

    // 4) final adaptive RMS norm
    h = adaptive_rms_norm(ctx, m, "output_norm_w", "output_norm_b", h, diff_step);

    // 5) mel_out MLP -> [mel_dim, T]
    h = mlp_forward(ctx, m, "mel_out_mlp", h);
    return h;
}

// ---------------------------------------------------------------------------
// GraphCache (plan item #8 — CUDA Graph capture / graph reuse)
//
// Caches the per-T computation graph, its scratch context, and (on GPU) the
// backend buffer so that repeated forward passes with the same sequence
// length T can:
//   - skip graph rebuilding (the cgraph, tensor metadata, and leaf pointers
//     are all stable),
//   - keep tensor data at stable device addresses, which lets the ggml CUDA
//     backend capture and replay the CUDA graph (eliminating per-kernel
//     launch overhead).
//
// On the first call for a given T the graph is built, intermediates are
// allocated, and leaves are flushed. On subsequent calls only the four leaf
// tensors (x, cond, timestep embedding, positions) are refreshed — the
// timestep embedding changes with `t`; x/cond change every call; positions
// are always [0..T-1] so they are set once and left alone.
// ---------------------------------------------------------------------------
struct GraphCache {
    int   T = 0;                       // sequence length this entry is built for
    Scratch * sc = nullptr;            // persistent scratch (owns ggml_context)
    ggml_cgraph * gf = nullptr;        // pre-built forward graph
    ggml_backend_buffer_t call_buf = nullptr;  // device buffer (GPU only)

    // Leaf tensors — addresses are stable for the lifetime of this cache;
    // only their *contents* are refreshed each call.
    ggml_tensor * x_leaf    = nullptr;  // [MEL_DIM, T]
    ggml_tensor * cond_leaf = nullptr;  // [HIDDEN,  T]
    ggml_tensor * emb_leaf  = nullptr;  // [HIDDEN]  timestep embedding
    ggml_tensor * pos_leaf  = nullptr;  // [T]       positions (set once)

    ggml_tensor * out = nullptr;        // output tensor [MEL_DIM, T]

    ~GraphCache() {
        // call_buf must be freed *before* the scratch ctx (which owns the
        // tensor metadata that call_buf backs on the GPU).
        if (call_buf) ggml_backend_buffer_free(call_buf);
        delete sc;  // frees the ggml_context and backing buffer
    }
};

// ---------------------------------------------------------------------------
// Refresh the leaf tensor contents for a new forward pass without rebuilding
// the graph. x/cond change every call; the timestep embedding changes with
// `t`. Positions are constant [0..T-1] so they are set once during build.
// ---------------------------------------------------------------------------
static void refresh_leaves(GraphCache & gc,
                           const float * x_data, const float * cond_data,
                           float t, bool use_gpu) {
    if (use_gpu) {
        ggml_backend_tensor_set(gc.x_leaf,    x_data,    0, sizeof(float) * MEL_DIM * gc.T);
        ggml_backend_tensor_set(gc.cond_leaf, cond_data, 0, sizeof(float) * HIDDEN  * gc.T);
        std::vector<float> emb = build_timestep_emb(t);
        ggml_backend_tensor_set(gc.emb_leaf,  emb.data(), 0, sizeof(float) * HIDDEN);
    } else {
        memcpy(gc.x_leaf->data,    x_data,    sizeof(float) * MEL_DIM * gc.T);
        memcpy(gc.cond_leaf->data, cond_data, sizeof(float) * HIDDEN  * gc.T);
        std::vector<float> emb = build_timestep_emb(t);
        memcpy(gc.emb_leaf->data,  emb.data(), sizeof(float) * HIDDEN);
    }
}

// ---------------------------------------------------------------------------
// Model::unload — defined here (after GraphCache) so `delete graph_cache`
// sees the complete type and actually invokes GraphCache::~GraphCache().
// ---------------------------------------------------------------------------
void Model::unload() {
    if (graph_cache) { delete graph_cache; graph_cache = nullptr; }
    if (weight_buf) { ggml_backend_buffer_free(weight_buf); weight_buf = nullptr; }
    if (backend)    { ggml_backend_free(backend); backend = nullptr; }
    if (gguf) gguf_free(gguf);
    if (ctx)  ggml_free(ctx);
#ifdef _WIN32
    if (mmap_data) { UnmapViewOfFile(mmap_data); }
    if (file_mapping) { CloseHandle(file_mapping); file_mapping = nullptr; }
#else
    if (mmap_data) munmap(mmap_data, mmap_size);
    if (fd >= 0) close(fd);
#endif
    ctx = nullptr; gguf = nullptr; mmap_data = nullptr; fd = -1;
    buft = nullptr; use_gpu = false; n_threads = 0;
}

// ---------------------------------------------------------------------------
// Run a forward pass against the model and copy the result to a host buffer.
//
// Graph caching (plan item #8): the computation graph, scratch context, and
// device buffer are cached per-T in m.graph_cache. On a cache hit (same T),
// only the leaf tensor values are refreshed and the existing graph is
// re-executed. On GPU this enables automatic CUDA Graph capture/replay by
// the ggml CUDA backend (stable cgraph pointer + stable tensor addresses).
//
// Flash Attention (plan item #9): on GPU the fused ggml_flash_attn_ext
// kernel replaces the manual mul_mat+soft_max+mul_mat attention, reducing
// memory bandwidth and kernel-launch count.
//
// Two compute paths share the same graph builder (dit_forward):
//   - GPU  (m.use_gpu && m.backend): scratch ctx is metadata-only
//     (no_alloc=true); intermediates live in the cached backend buffer.
//   - CPU: scratch ctx backs both metadata and data (no_alloc=false).
//
// (External linkage — called from binding.cc.)
// ---------------------------------------------------------------------------
std::vector<float> run_forward(
    const Model & m,
    const float * x_data,
    const float * cond_data,
    float t,
    int T)
{
    const bool use_gpu = m.use_gpu && m.backend;
    const bool use_flash_attn = use_gpu;  // Flash Attention on GPU only

    std::vector<float> result((size_t)MEL_DIM * T);

    // ---- Cache hit: reuse the existing graph for this T ----
    GraphCache * gc = m.graph_cache;
    if (gc && gc->T == T && gc->gf) {
        refresh_leaves(*gc, x_data, cond_data, t, use_gpu);

        if (use_gpu) {
            if (ggml_backend_graph_compute(m.backend, gc->gf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "FATAL: ggml_backend_graph_compute (cached) failed "
                                "(backend='%s')\n", m.backend_name.c_str());
                return result;
            }
            ggml_backend_tensor_get(gc->out, result.data(), 0,
                                    sizeof(float) * MEL_DIM * T);
        } else {
            ggml_graph_compute_with_ctx(gc->sc->ctx, gc->gf, m.n_threads);
            memcpy(result.data(), gc->out->data, sizeof(float) * MEL_DIM * T);
        }
        return result;
    }

    // ---- Cache miss: build a new graph and cache it ----
    if (gc) { delete gc; gc = nullptr; }
    gc = new GraphCache();
    gc->T = T;
    // CPU scratch also holds tensor *data* (1 GiB is plenty for 22 layers at
    // T=64). GPU scratch is metadata-only; 256 MiB fits the tensor structs +
    // cgraph nodes for the DiT.
    gc->sc = new Scratch(use_gpu ? (1ULL << 28) : (1ULL << 30), use_gpu);

    // Leaf input tensors — their addresses stay fixed for the cache lifetime.
    gc->x_leaf    = ggml_new_tensor_2d(gc->sc->ctx, GGML_TYPE_F32, MEL_DIM, T);
    gc->cond_leaf = ggml_new_tensor_2d(gc->sc->ctx, GGML_TYPE_F32, HIDDEN,  T);
    gc->sc->stage_leaf(gc->x_leaf,    x_data,    sizeof(float) * MEL_DIM * T);
    gc->sc->stage_leaf(gc->cond_leaf, cond_data, sizeof(float) * HIDDEN  * T);

    // Build the DiT graph, exposing the timestep-emb and positions leaves.
    gc->out = dit_forward(*gc->sc, m, gc->x_leaf, gc->cond_leaf, t, T,
                          use_flash_attn, &gc->emb_leaf, &gc->pos_leaf);

    // Build the cgraph.
    gc->gf = ggml_new_graph_custom(gc->sc->ctx, 1 << 18, false);
    ggml_build_forward_expand(gc->gf, gc->out);

    if (use_gpu) {
        // Allocate intermediate tensors in the backend (VRAM) buffer.
        gc->call_buf = ggml_backend_alloc_ctx_tensors(gc->sc->ctx, m.backend);
        if (!gc->call_buf) {
            fprintf(stderr, "FATAL: ggml_backend_alloc_ctx_tensors failed "
                            "(backend='%s')\n", m.backend_name.c_str());
            m.graph_cache = gc;  // keep partial cache so destructor cleans up
            return result;
        }
        // Flush staged leaves into device memory (first time only).
        for (const auto & l : gc->sc->leaves) {
            ggml_backend_tensor_set(l.t, l.data.data(), 0, l.data.size());
        }
        // Run on the GPU backend (first run captures the CUDA graph).
        if (ggml_backend_graph_compute(m.backend, gc->gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "FATAL: ggml_backend_graph_compute failed "
                            "(backend='%s')\n", m.backend_name.c_str());
            m.graph_cache = gc;
            return result;
        }
        ggml_backend_tensor_get(gc->out, result.data(), 0,
                                sizeof(float) * MEL_DIM * T);
    } else {
        // CPU: leaves point into the metadata+data scratch buffer; flush them.
        for (const auto & l : gc->sc->leaves) {
            memcpy(l.t->data, l.data.data(), l.data.size());
        }
        ggml_graph_compute_with_ctx(gc->sc->ctx, gc->gf, m.n_threads);
        memcpy(result.data(), gc->out->data, sizeof(float) * MEL_DIM * T);
    }

    m.graph_cache = gc;
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
