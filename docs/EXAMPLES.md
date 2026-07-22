# 示例集

本文档提供 6 个完整、可运行的 `.js` 示例，覆盖本绑定的典型用法。每个示例都是自包含的脚本，可直接
`node example.js` 运行（需先安装本包及对应后端子包，并准备好 `.gguf` 模型文件）。

> 运行前请将脚本中的 `MODEL_PATH` 替换为你的实际模型路径。所有示例使用 CommonJS（`require`），
> 与包的 `index.js` 入口一致；如需 ESM 可自行改写为 `import`。

---

## 目录

1. [基本前向推理](#1-基本前向推理)
2. [反向扩散生成](#2-反向扩散生成)
3. [后端切换](#3-后端切换)
4. [与 Python 参考对比](#4-与-python-参考对比)
5. [批量推理](#5-批量推理)
6. [错误处理](#6-错误处理)

---

## 1. 基本前向推理

加载模型，构造一段（这里用随机数填充的）输入，调用一次 `forward` 并打印输出统计信息。这是验证
「绑定能正常加载并推理」的最小冒烟测试。

```js
// examples/basic_forward.js
// 基本前向推理：加载模型 -> 单次 forward -> 打印输出统计
'use strict';

const { loadModel, MEL_DIM, HIDDEN } = require('soulx-singer-dit');

const MODEL_PATH = process.env.MODEL_PATH || './dit.gguf';

async function main() {
  // 用默认 CPU 后端加载模型
  const model = await loadModel(MODEL_PATH, { backend: 'cpu' });
  console.log('模型加载完成');

  // 构造输入。这里用随机数填充；实际使用时应填入真实的 mel / cond 数据。
  const T = 64;                                       // 帧数
  const x    = new Float32Array(MEL_DIM * T);         // [128, 64]
  const cond = new Float32Array(HIDDEN * T);          // [1024, 64]
  for (let i = 0; i < x.length;    i++) x[i]    = Math.random() * 2 - 1;
  for (let i = 0; i < cond.length; i++) cond[i] = Math.random() * 2 - 1;

  // 单次前向推理，t 取 0.5 表示扩散过程的中间时间步
  const velocity = model.forward({ x, cond, t: 0.5, T });

  // 打印输出的一些统计量，确认非空且数值正常
  let min = Infinity, max = -Infinity, sum = 0;
  for (const v of velocity) {
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
  }
  console.log('输出长度 :', velocity.length);          // 128 * 64 = 8192
  console.log('min/max  :', min.toFixed(6), max.toFixed(6));
  console.log('mean     :', (sum / velocity.length).toFixed(6));

  model.release();
}

main().catch((err) => {
  console.error('运行失败:', err);
  process.exit(1);
});
```

运行：

```bash
MODEL_PATH=/path/to/dit.gguf node examples/basic_forward.js
```

---

## 2. 反向扩散生成

完整的 8 步 Euler ODE 流程：给定一段 prompt mel 与条件，从噪声出发续写一段目标 mel。
`reverseDiffusion` 内部已封装好采样循环，调用方只需准备输入与超参。

```js
// examples/reverse_diffusion.js
// 反向扩散生成：prompt mel + 条件 + 噪声 -> 目标 mel
'use strict';

const { loadModel, MEL_DIM, HIDDEN } = require('soulx-singer-dit');

const MODEL_PATH = process.env.MODEL_PATH || './dit.gguf';

async function main() {
  const model = await loadModel(MODEL_PATH, { backend: 'cpu' });

  // 采样超参
  const promptLen = 16;      // prompt 帧数
  const targetLen = 48;      // 要生成的目标帧数
  const nSteps    = 8;       // ODE 步数（flow matching 推荐 8）
  const T         = promptLen + targetLen;

  // 准备输入（这里用随机数据占位；实际使用时 promptMel 来自真实音频特征，
  // cond 来自上游模型，z 可用任意噪声）
  const promptMel = new Float32Array(MEL_DIM * promptLen);
  const cond      = new Float32Array(HIDDEN * T);
  const z         = new Float32Array(MEL_DIM * targetLen);   // 初始噪声

  // 用固定种子的伪随机数填充，保证本示例本身可复现
  let s = 12345;
  const rand = () => {
    s = (s * 6364136223846793005 + 1442695040888963407) & 0xffffffff;
    return (s >>> 8) * (1 / (1 << 24)) * 2 - 1;
  };
  for (let i = 0; i < promptMel.length; i++) promptMel[i] = rand();
  for (let i = 0; i < cond.length;      i++) cond[i]      = rand();
  for (let i = 0; i < z.length;         i++) z[i]         = rand();

  console.time('reverseDiffusion');
  const mel = model.reverseDiffusion({
    promptMel,
    cond,
    z,
    promptLen,
    targetLen,
    nSteps,
    seed: 12345,   // 与 C++ 参考一致；省略时默认也是 12345
  });
  console.timeEnd('reverseDiffusion');

  console.log('生成 mel 长度:', mel.length);           // 128 * 48 = 6144
  console.log('前 8 个采样值 :', Array.from(mel.slice(0, 8))
    .map((v) => v.toFixed(6)).join(', '));

  model.release();
}

main().catch((err) => {
  console.error('运行失败:', err);
  process.exit(1);
});
```

> 说明：`reverseDiffusion` 内部会对每个 ODE 步调用一次 `forward`，因此 `nSteps=8` 等价于 8 次前向。
> 增大 `nSteps` 可提升采样质量但会线性增加耗时。

---

## 3. 后端切换

演示如何在 CPU / Vulkan / CUDA 之间切换。先用 `listBackends()` 探测可用后端，再依次加载同一个模型
对比推理耗时。这同样是「如何选择合适后端」的实用参考。

```js
// examples/backend_switch.js
// 后端切换：探测可用后端 -> 逐个加载并计时
'use strict';

const { loadModel, listBackends, MEL_DIM, HIDDEN } = require('soulx-singer-dit');

const MODEL_PATH = process.env.MODEL_PATH || './dit.gguf';

async function benchBackend(backend) {
  console.log(`\n=== 后端: ${backend} ===`);
  const model = await loadModel(MODEL_PATH, { backend });

  // 准备一次前向的输入
  const T = 64;
  const x    = new Float32Array(MEL_DIM * T);
  const cond = new Float32Array(HIDDEN * T);
  for (let i = 0; i < x.length;    i++) x[i]    = Math.sin(i) * 0.1;
  for (let i = 0; i < cond.length; i++) cond[i] = Math.cos(i) * 0.1;

  // 预热一次（首次推理可能包含 kernel JIT / 缓存填充）
  model.forward({ x, cond, t: 0.5, T });

  // 正式计时：跑 3 次取平均
  const N = 3;
  console.time(`forward x${N}`);
  for (let i = 0; i < N; i++) {
    model.forward({ x, cond, t: 0.5, T });
  }
  console.timeEnd(`forward x${N}`);

  model.release();
}

async function main() {
  const available = listBackends();
  console.log('当前平台可用后端:', available);

  if (available.length === 0) {
    console.error('没有可用的预编译后端，请先安装对应子包。');
    process.exit(1);
  }

  // 也可以通过环境变量指定只测某一个后端
  const only = process.env.BACKEND;
  const targets = only ? [only] : available;

  for (const b of targets) {
    try {
      await benchBackend(b);
    } catch (err) {
      console.error(`后端 ${b} 失败:`, err.message);
    }
  }
}

main();
```

运行（只测 vulkan）：

```bash
BACKEND=vulkan MODEL_PATH=/path/to/dit.gguf node examples/backend_switch.js
```

---

## 4. 与 Python 参考对比

如何在 Node.js 中复现精度验证。思路：用**相同**的输入张量与**相同**的 RNG 种子，分别在本绑定与
Python 参考实现（PyTorch + FP32 权重，或量化后的 GGUF 参考脚本）上跑前向，然后计算两者输出的
余弦相似度（`cos_sim`）与最大绝对误差。

```js
// examples/compare_python.js
// 与 Python 参考对比：读取预存的输入/参考输出 -> 本绑定前向 -> 计算 cos_sim
'use strict';

const fs = require('fs');
const { loadModel, MEL_DIM, HIDDEN } = require('soulx-singer-dit');

const MODEL_PATH = process.env.MODEL_PATH || './dit.gguf';
// 这些 .bin 文件由 Python 侧导出（np.save / torch.save 后转成裸 float32）。
// 文件格式：纯 float32 二进制，无头。
const X_PATH     = process.env.X_PATH     || './fixtures/x.bin';
const COND_PATH  = process.env.COND_PATH  || './fixtures/cond.bin';
const REF_PATH   = process.env.REF_PATH   || './fixtures/ref_velocity.bin';
const T = Number(process.env.T || '64');

function readF32(path) {
  const buf = fs.readFileSync(path);
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

function cosSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na  += a[i] * a[i];
    nb  += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function maxAbsDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

async function main() {
  const x    = readF32(X_PATH);
  const cond = readF32(COND_PATH);
  const ref  = readF32(REF_PATH);

  // 校验输入维度
  if (x.length !== MEL_DIM * T)    throw new Error(`x 长度 ${x.length} != ${MEL_DIM * T}`);
  if (cond.length !== HIDDEN * T)  throw new Error(`cond 长度 ${cond.length} != ${HIDDEN * T}`);
  if (ref.length !== MEL_DIM * T)  throw new Error(`ref 长度 ${ref.length} != ${MEL_DIM * T}`);

  const model = await loadModel(MODEL_PATH, { backend: 'cpu' });

  // 用与 Python 参考完全一致的时间步 t
  const t = Number(process.env.T_STEP || '0.5');
  const out = model.forward({ x, cond, t, T });

  const cs = cosSim(ref, out);
  const mad = maxAbsDiff(ref, out);
  console.log('cos_sim      :', cs.toFixed(6));
  console.log('max_abs_diff :', mad.toFixed(6));

  // 经验阈值：Q8_0 应 >= 0.9999，Q4_K_M 应 >= 0.996
  const threshold = Number(process.env.COS_THRESHOLD || '0.996');
  if (cs < threshold) {
    console.error(`FAIL: cos_sim ${cs.toFixed(6)} < 阈值 ${threshold}`);
    process.exit(2);
  } else {
    console.log('PASS');
  }

  model.release();
}

main().catch((err) => {
  console.error('对比失败:', err);
  process.exit(1);
});
```

配套的 Python 导出脚本片段（供参考，不在本仓库内）：

```python
# export_fixtures.py（节选）
import numpy as np, torch
# x, cond: 与模型 forward 相同形状的 numpy float32
# ref = model.forward(x, cond, t=0.5, T=64)  # PyTorch 参考输出
x.astype(np.float32).tofile("fixtures/x.bin")
cond.astype(np.float32).tofile("fixtures/cond.bin")
ref.astype(np.float32).tofile("fixtures/ref_velocity.bin")
```

运行：

```bash
MODEL_PATH=/path/to/dit.q8_0.gguf \
X_PATH=./fixtures/x.bin COND_PATH=./fixtures/cond.bin REF_PATH=./fixtures/ref_velocity.bin \
T=64 T_STEP=0.5 COS_THRESHOLD=0.9999 \
node examples/compare_python.js
```

---

## 5. 批量推理

多次调用 `forward` 的性能模式：复用同一个已加载的 `Model` 实例，循环处理多组输入。重点在于
**避免在循环内反复加载模型**——加载（mmap + GGUF 解析）是一次性开销，应在循环外完成。

```js
// examples/batch_inference.js
// 批量推理：复用模型实例，循环处理多组输入，并统计吞吐
'use strict';

const { loadModel, MEL_DIM, HIDDEN } = require('soulx-singer-dit');

const MODEL_PATH = process.env.MODEL_PATH || './dit.gguf';
const BATCH = Number(process.env.BATCH || '32');   // 处理多少组输入
const T = Number(process.env.T || '64');

async function main() {
  const model = await loadModel(MODEL_PATH, { backend: 'cpu' });
  console.log(`开始批量推理：${BATCH} 组，每组 T=${T}`);

  // 预生成全部输入，避免把数据生成时间算进推理耗时
  const inputs = [];
  for (let b = 0; b < BATCH; b++) {
    const x    = new Float32Array(MEL_DIM * T);
    const cond = new Float32Array(HIDDEN * T);
    for (let i = 0; i < x.length;    i++) x[i]    = Math.sin(i + b) * 0.1;
    for (let i = 0; i < cond.length; i++) cond[i] = Math.cos(i + b) * 0.1;
    inputs.push({ x, cond, t: 0.5, T });
  }

  // 预热
  model.forward(inputs[0]);

  // 正式批量推理
  const start = process.hrtime.bigint();
  let totalElements = 0;
  for (let b = 0; b < BATCH; b++) {
    const out = model.forward(inputs[b]);
    totalElements += out.length;
  }
  const elapsedNs = Number(process.hrtime.bigint() - start);
  const elapsedMs = elapsedNs / 1e6;

  console.log('总耗时         :', elapsedMs.toFixed(2), 'ms');
  console.log('平均每组       :', (elapsedMs / BATCH).toFixed(2), 'ms');
  console.log('吞吐           :', (BATCH / (elapsedMs / 1000)).toFixed(2), 'forward/s');
  console.log('处理元素总数   :', totalElements);

  model.release();
}

main().catch((err) => {
  console.error('运行失败:', err);
  process.exit(1);
});
```

> 提示：若需进一步压榨性能，可在加载时切换到 `vulkan` / `cuda` 后端；GPU 后端在小 `T` 下因
> kernel 启动开销未必占优，但在大 `T` 或大批量下收益显著。

---

## 6. 错误处理

覆盖常见错误场景：模型路径错误、输入维度不匹配、后端不可用、`release()` 之后调用等。每个场景都
用 `try/catch` 捕获并打印可读信息，避免进程崩溃。

```js
// examples/error_handling.js
// 错误处理：演示各类错误及其捕获方式
'use strict';

const { loadModel, listBackends, MEL_DIM, HIDDEN } = require('soulx-singer-dit');

async function expectFail(label, fn) {
  try {
    await fn();
    console.error(`[未如预期抛错] ${label}`);
  } catch (err) {
    console.log(`[OK 已捕获] ${label}`);
    console.log('   ->', err.message.split('\n')[0]);
  }
}

async function main() {
  console.log('可用后端:', listBackends());

  // (a) 模型路径不存在 -> 原生层抛 "Failed to load GGUF model"
  await expectFail('模型路径不存在', async () => {
    await loadModel('./does-not-exist.gguf', { backend: 'cpu' });
  });

  // (b) 非法后端名 -> TypeError
  await expectFail('非法后端名', async () => {
    await loadModel('./dit.gguf', { backend: 'tpu' });
  });

  // (c) 未安装的后端 -> Error（含安装提示）
  await expectFail('未安装的后端', async () => {
    // 仅当 cuda 未安装时才会抛错；若已安装则跳过本场景
    if (listBackends().includes('cuda')) {
      console.log('   (cuda 已安装，跳过本场景)');
      return;
    }
    await loadModel('./dit.gguf', { backend: 'cuda' });
  });

  // (d) forward 维度不匹配 -> TypeError
  await expectFail('forward 维度不匹配', async () => {
    const model = await loadModel(process.env.MODEL_PATH || './dit.gguf', { backend: 'cpu' });
    const T = 64;
    const x = new Float32Array(MEL_DIM * T);
    const cond = new Float32Array(HIDDEN * (T - 1));   // 故意少一帧
    model.forward({ x, cond, t: 0.5, T });
    model.release();
  });

  // (e) forward 缺少必填字段 -> TypeError
  await expectFail('forward 缺少字段', async () => {
    const model = await loadModel(process.env.MODEL_PATH || './dit.gguf', { backend: 'cpu' });
    model.forward({ x: new Float32Array(MEL_DIM * 4), T: 4 });   // 缺 cond / t
    model.release();
  });

  // (f) release() 之后调用 forward -> Error "Model is not loaded"
  await expectFail('release 后调用 forward', async () => {
    const model = await loadModel(process.env.MODEL_PATH || './dit.gguf', { backend: 'cpu' });
    model.release();
    const T = 4;
    model.forward({
      x: new Float32Array(MEL_DIM * T),
      cond: new Float32Array(HIDDEN * T),
      t: 0.5, T,
    });
  });

  // (g) release() 重复调用 -> no-op，不抛错
  {
    const model = await loadModel(process.env.MODEL_PATH || './dit.gguf', { backend: 'cpu' });
    model.release();
    model.release();   // 安全，无副作用
    console.log('[OK] release 重复调用安全');
  }
}

main().catch((err) => {
  console.error('示例本身出错:', err);
  process.exit(1);
});
```

> 注意：场景 (d)~(g) 需要一个真实可加载的模型（由 `MODEL_PATH` 指定），否则会先在 `loadModel`
> 阶段失败。若手头没有模型，可只运行 (a)~(c)。
