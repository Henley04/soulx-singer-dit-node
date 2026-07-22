# SoulX-Singer-DiT Node.js Bindings

> **English summary:** `soulx-singer-dit` provides native Node.js (N-API) bindings for running
> SoulX-Singer-DiT (DiffLlama) GGUF model inference entirely in C++ — no Python runtime is required.
> Weights stay packed in their quantized form (Q8_0 / Q4_K_M) and all matrix multiplications go
> through ggml's quantized kernels, so **no dequantization** ever happens. Prebuilt Windows x64
> binaries are shipped for three backends: CPU, Vulkan, and CUDA.

本包是 [SoulX-Singer](https://github.com/SoulX-AI/SoulX-Singer) 项目中 DiffLlama (Diffusion Transformer)
推理引擎的 Node.js 原生绑定。它将纯 C++ 推理实现通过 N-API 暴露给 JavaScript，让你能在 Node.js 进程中
直接加载 GGUF 模型、执行前向推理与反向扩散（采样），而无需安装 Python、PyTorch 或任何深度学习运行时。

核心设计原则是 **「不反量化」**：GGUF 文件以 `mmap` 方式映射到内存，权重张量保持其打包格式
（如 `Q8_0`、`Q4_K_M`），所有矩阵乘法都经由 `ggml_mul_mat` 派发到 ggml 内建的量化 kernel 直接在打包
权重上运算。这样既大幅降低了内存占用（权重不会被展开成 FP32），又保留了量化 kernel 的 SIMD 加速。

---

## 特性 / Features

- **纯 C++ 推理，无 Python 依赖** —— 通过 N-API 直接调用，无需 Python 解释器、PyTorch 或 CUDA Python。
- **不反量化** —— 权重保持 `Q8_0` / `Q4_K_M` 打包格式，所有 matmul 走 ggml 量化 kernel。
- **Windows x64 预编译二进制** —— 提供 CPU / Vulkan / CUDA 三种后端的预编译 `.node` 文件，`npm install` 即用。
- **完整 TypeScript 类型支持** —— 随包附带 `index.d.ts`，所有公开 API 均有类型定义与文档注释。
- **完整复刻 DiffLlama 架构** —— 非因果（双向）自注意力、AdaptiveRMSNorm、mel/cond/timestep MLP、RoPE（NeoX, theta=10000）。
- **确定性可复现** —— 自带 LCG + Box-Muller PRNG，反向扩散结果跨机器完全一致，便于与 Python 参考实现做精度对比。
- **完整反向扩散采样** —— 内建 8 步 Euler ODE 积分器（flow matching），无需自己写采样循环。
- **npm OIDC Trusted Publisher** —— 发布使用 GitHub Actions + npm OIDC，附带 provenance 来源签名。

---

## 安装 / Installation

本主包 `soulx-singer-dit` 是一个纯 JavaScript shim，它本身不含 `.node` 二进制，而是在运行时根据当前
平台与请求的后端加载对应的预编译子包。三个预编译子包以 `optionalDependencies` 形式声明，安装主包时
npm 会自动尝试拉取它们。

### CPU 后端（默认）

```bash
npm install soulx-singer-dit
```

主包的 `optionalDependencies` 会自动尝试安装全部三个后端的子包。在 Windows x64 上，CPU 子包
`soulx-singer-dit-win32-x64-cpu` 通常会被成功安装，开箱即用。

### Vulkan 后端

```bash
npm install soulx-singer-dit
```

如上所述，安装主包时会自动尝试安装 Vulkan 子包 `soulx-singer-dit-win32-x64-vulkan`。
若你的网络或 registry 配置导致 `optionalDependencies` 未被安装，可显式单独安装：

```bash
npm install soulx-singer-dit-win32-x64-vulkan
```

> 提示：Vulkan 后端需要你的系统装有兼容的 Vulkan 驱动（绝大多数现代 GPU 厂商驱动均自带）。

### CUDA 后端

```bash
npm install soulx-singer-dit
# 如需显式安装 CUDA 子包：
npm install soulx-singer-dit-win32-x64-cuda
```

> CUDA 后端仅适用于 NVIDIA GPU，且要求系统装有 CUDA 12.x 运行时（cudart / cublas / curand）。

### 从源码构建

当前仅 Windows x64 提供预编译二进制。若你在 macOS / Linux 上使用，或希望自行编译，请参阅
[构建指南](docs/BUILD.md)。

---

## 快速开始 / Quick Start

下面的示例展示如何加载模型、执行一次前向推理，并运行一次完整的反向扩散采样。

```js
const { loadModel, getVersion, listBackends, MEL_DIM, HIDDEN } = require('soulx-singer-dit');

// 1) 诊断信息：无需加载 .node 即可调用
console.log('version :', getVersion());        // -> "0.1.0"
console.log('backends:', listBackends());       // -> ['cpu'] 或 ['cpu','vulkan','cuda']

// 2) 异步加载模型（推荐使用 loadModel 而非 new Model）
const model = await loadModel('/path/to/dit.gguf', { backend: 'cpu' });

// 3) 单次前向推理：预测 flow-matching 的速度场
const T = 64;
const x    = new Float32Array(MEL_DIM * T);   // 输入 mel 状态
const cond = new Float32Array(HIDDEN * T);    // 条件 hidden states
// ... 在此处填充 x / cond 的实际数值 ...
const velocity = model.forward({ x, cond, t: 0.5, T });
// velocity: Float32Array, 长度 = MEL_DIM * T

// 4) 反向扩散：从 prompt mel 续写目标 mel
const promptLen = 16;
const targetLen = 48;
const promptMel = new Float32Array(MEL_DIM * promptLen);
const condFull  = new Float32Array(HIDDEN * (promptLen + targetLen));
const z         = new Float32Array(MEL_DIM * targetLen);   // 初始噪声
const mel = model.reverseDiffusion({
  promptMel, cond: condFull, z,
  promptLen, targetLen,
  nSteps: 8,
  seed: 12345,           // 可选，默认 12345，与 C++ 参考实现保持一致
});
// mel: Float32Array, 长度 = MEL_DIM * targetLen

// 5) 用完显式释放原生资源（可选；GC 时也会自动释放）
model.release();
```

> 更多完整可运行示例（批量推理、后端切换、与 Python 对比、错误处理等）见
> [示例集](docs/EXAMPLES.md)。

---

## API 概览 / API Overview

| 导出 | 类型 | 说明 |
| --- | --- | --- |
| `loadModel(path, options?)` | `function` | 异步工厂，返回 `Promise<Model>`，推荐用法。 |
| `Model` | `class` | 已加载模型；构造函数 `new Model(path, options?)`。 |
| `model.forward(opts)` | `method` | 单次前向推理，返回速度场 `Float32Array(MEL_DIM*T)`。 |
| `model.reverseDiffusion(opts)` | `method` | 反向扩散采样，返回生成 mel `Float32Array(MEL_DIM*targetLen)`。 |
| `model.release()` | `method` | 立即释放原生资源；重复调用是 no-op。 |
| `getVersion()` | `function` | 返回绑定版本字符串，如 `"0.1.0"`。无需加载 `.node` 即可调用。 |
| `listBackends()` | `function` | 返回当前平台已安装并可解析的后端列表，如 `['cpu','vulkan']`。 |
| `MEL_DIM` | `const` | `128`，mel 维度。 |
| `HIDDEN` | `const` | `1024`，hidden 维度。 |

类型定义（`LoadOptions` / `ForwardOptions` / `ReverseDiffusionOptions` / `Backend`）见
`index.d.ts`，完整中文参考见 [API 完整参考](docs/API.md)。

---

## 架构 / Architecture

### DiffLlama 架构

本绑定完整复刻了 SoulX-Singer 中 `FlowMatchingTransformer` 所使用的 DiffLlama 结构
（对应 `soulxsinger/models/modules/llama.py`）。其核心是在标准 Llama 之上做了如下改造：

- **非因果（双向）自注意力** —— 使用全 1 的 `x_mask`，无因果遮罩，使每帧都能看到上下文。
- **AdaptiveRMSNorm** —— 归一化的 scale 由 `Linear(timestep_emb)` 动态产生，而非固定可学习参数。
- **I/O 投影 MLP** —— `mel_in_mlp`（`mel_dim → hidden`）与 `mel_out_mlp`（`hidden → mel_dim`）。
- **条件 / 时间步 MLP** —— `cond_mlp`（条件编码）与 `timestep_mlp`（正弦位置编码 → hidden）。
- **RoPE** —— HF Llama 默认：NeoX 风格，`theta=10000`，不做缩放。

模型超参（与 `convert_dit_to_gguf.py` 的 `DIT_HPARAMS` 保持一致）：

| 常量 | 值 | 说明 |
| --- | --- | --- |
| `MEL_DIM` | 128 | mel 维度 |
| `HIDDEN` | 1024 | hidden 维度 |
| `NUM_LAYERS` | 22 | 解码器层数 |
| `NUM_HEADS` | 16 | 注意力头数 |
| `HEAD_DIM` | 64 | 每头维度（= 1024 / 16） |
| `INTERMEDIATE` | 4096 | FFN 中间维度（= 4 * hidden） |
| `RMS_EPS` | 1e-6 | RMSNorm epsilon |
| `ROPE_THETA` | 10000 | RoPE base theta |

### 量化推理路径（为什么不反量化）

1. **mmap 加载**：GGUF 文件以 `PROT_READ | MAP_PRIVATE` 映射，`tensor->data` 直接指向 mmap 区域内
   对应偏移，不发生任何权重拷贝或展开。
2. **打包权重直接参与运算**：所有 matmul 调用 `ggml_mul_mat`，由 ggml-cpu 根据权重张量的
   `ggml_type`（`Q8_0` / `Q4_K_M` 等）派发到对应的量化 kernel，直接在打包数据上计算。
3. **激活值仍是 FP32**：只有权重保持量化；中间激活、KV、注意力分数等都是 FP32，保证数值精度。

这样做的好处是：内存占用接近权重文件的原始大小（量化后），同时享受量化 kernel 的 SIMD 向量化加速。

### 反向扩散（采样）

`reverseDiffusion` 实现了 flow matching 的 8 步 Euler ODE 积分，复刻
`FlowMatchingTransformer.reverse_diffusion`：

1. 将 `promptMel` 与当前 `xt` 在时间维拼接为 `[MEL_DIM, promptLen + targetLen]`。
2. 对每个时间步 `t = (i + 0.5) / nSteps` 调用一次 `forward` 得到速度场。
3. 取速度场中 `targetLen` 部分，按 `xt = xt + flow_pred * h` 更新（`h = 1/nSteps`）。
4. 迭代 `nSteps` 次后返回最终 `xt`，即生成的目标 mel。

---

## 精度验证 / Accuracy

我们以 Python 参考实现（PyTorch + 原始 FP32 权重）的输出为基准，对本绑定的量化推理结果做余弦相似度
（`cos_sim`）对比。在相同输入与相同 RNG 种子下：

| 量化格式 | 与 Python 参考的 `cos_sim` | 说明 |
| --- | --- | --- |
| `Q8_0` | ≈ 0.9999 | 几乎无损，推荐用于精度敏感场景。 |
| `Q4_K_M` | ≈ 0.9965 | 显著节省显存/内存，精度仍可接受。 |

验证方法见 [示例集 — 与 Python 参考对比](docs/EXAMPLES.md#4-与-python-参考对比)。

---

## 后端选择 / Backend Selection

目前仅 Windows x64 提供预编译后端。三种后端对比：

| 后端 | 适用场景 | 额外依赖 | 速度（相对） | 内存 |
| --- | --- | --- | --- | --- |
| `cpu` | 通用、无 GPU、跨设备兼容 | 无 | 基准（3 线程） | 最低 |
| `vulkan` | 跨厂商 GPU 加速（AMD / Intel / NVIDIA） | Vulkan 驱动 | 较快 | 中 |
| `cuda` | NVIDIA GPU 最高性能 | CUDA 12.x 运行时 | 最快 | 中 |

可在加载时通过 `options.backend` 指定：

```js
const m = await loadModel(path, { backend: 'vulkan' });
```

调用 `listBackends()` 可查看当前平台实际可用的后端。若请求的后端未安装，会抛出带有安装提示的清晰错误。

---

## 文档 / Documentation

- [API 完整参考](docs/API.md) —— 每个导出的签名、参数表、返回值、错误与示例。
- [示例集](docs/EXAMPLES.md) —— 6 个完整可运行的 `.js` 示例及中文讲解。
- [构建指南](docs/BUILD.md) —— 本地从源码构建（CPU / Vulkan / CUDA）与故障排查。
- [发布流程](docs/RELEASE.md) —— npm OIDC Trusted Publisher 配置与版本发布。

---

## 许可证 / License

[MIT](LICENSE)

---

## 相关链接

- **GitHub 仓库**：<https://github.com/Henley04/soulx-singer-dit-node>
- **问题反馈**：<https://github.com/Henley04/soulx-singer-dit-node/issues>
- **llama.cpp**：<https://github.com/ggml-org/llama.cpp> （本绑定的量化 kernel 来源）
- **ggml**：<https://github.com/ggml-org/ggml>
- **SoulX-Singer 架构源码**：参见 SoulX-Singer 仓库中 `soulxsinger/models/modules/llama.py`
