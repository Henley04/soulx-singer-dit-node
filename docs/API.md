# API 完整参考

`soulx-singer-dit` 的公开 API 由 `index.js`（JavaScript shim）与 `index.d.ts`（TypeScript 类型）
共同定义。本文档以中文详细描述每个导出：签名、参数、返回值、可能抛出的错误，以及可运行的最小示例。

> 约定：所有张量输入均为 **行主序（row-major）**，前导维度为对应的特征维度。即 `[MEL_DIM, T]` 的
> 数组按 `arr[k * T + t]` 寻址（`k` 为特征维下标，`t` 为时间帧下标）。

---

## 目录

- [类型](#类型)
  - [`Backend`](#backend)
  - [`LoadOptions`](#loadoptions)
  - [`ForwardOptions`](#forwardoptions)
  - [`ReverseDiffusionOptions`](#reversediffusionoptions)
- [函数](#函数)
  - [`loadModel(path, options?)`](#loadmodelpath-options)
  - [`getVersion()`](#getversion)
  - [`listBackends()`](#listbackends)
- [类 `Model`](#类-model)
  - [`constructor(path, options?)`](#constructorpath-options)
  - [`model.forward(opts)`](#modelforwardopts)
  - [`model.reverseDiffusion(opts)`](#modelreversediffusionopts)
  - [`model.release()`](#modelrelease)
  - [`Model.loadModel(path, options?)`](#modelloadmodelpath-options)
- [常量](#常量)
  - [`MEL_DIM`](#mel_dim)
  - [`HIDDEN`](#hidden)

---

## 类型

### `Backend`

```ts
export type Backend = 'cpu' | 'vulkan' | 'cuda';
```

支持的推理后端标识符。

| 取值 | 说明 |
| --- | --- |
| `'cpu'` | 纯 CPU 执行，基于 ggml。在所有支持平台上始终可用（若预编译子包已安装）。 |
| `'vulkan'` | 通过 Vulkan 后端做 GPU 加速。 |
| `'cuda'` | 通过 CUDA 后端做 GPU 加速（仅 NVIDIA）。 |

> 当前仅 `win32-x64` 平台提供预编译二进制。其它平台调用 `loadModel` 会抛出说明性错误。

---

### `LoadOptions`

```ts
export interface LoadOptions {
  backend?: Backend;
}
```

`new Model(...)` 与 `loadModel(...)` 接受的选项。

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `backend` | `Backend` | 否 | 选择要加载的预编译子包。省略时默认为 `'cpu'`。所选后端必须已在当前平台安装（见 `listBackends()`）。 |

---

### `ForwardOptions`

```ts
export interface ForwardOptions {
  x:    Float32Array;
  cond: Float32Array;
  t:    number;
  T:    number;
}
```

`model.forward(...)` 的参数。绑定会执行一次 DiT 前向传播，返回预测的速度场（flow-matching velocity）。
张量按行主序展开，前导维度为特征维度（即数组布局为 `[MEL_DIM, T]` / `[HIDDEN, T]`）。

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `x` | `Float32Array` | 是 | 输入 mel 状态，长度必须为 `MEL_DIM * T`（即 `128 * T`）。 |
| `cond` | `Float32Array` | 是 | 条件 hidden states，长度必须为 `HIDDEN * T`（即 `1024 * T`）。 |
| `t` | `number` | 是 | 连续扩散时间步，取值范围 `[0, 1]`。 |
| `T` | `number` | 是 | 本 chunk 的帧数；必须为正整数。 |

> 维度校验：若 `x.length !== 128 * T`、`cond.length !== 1024 * T` 或 `T <= 0`，将抛出 `TypeError`。

---

### `ReverseDiffusionOptions`

```ts
export interface ReverseDiffusionOptions {
  promptMel:  Float32Array;
  cond:       Float32Array;
  z:          Float32Array;
  promptLen:  number;
  targetLen:  number;
  nSteps:     number;
  seed?:      number;
}
```

`model.reverseDiffusion(...)` 的参数。执行反向扩散（采样），从 prompt mel 续写一段目标 mel，
整个过程由条件 hidden states 引导。

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `promptMel` | `Float32Array` | 是 | prompt mel 频谱，长度必须为 `MEL_DIM * promptLen`。 |
| `cond` | `Float32Array` | 是 | 条件 hidden states，长度必须为 `HIDDEN * (promptLen + targetLen)`。 |
| `z` | `Float32Array` | 是 | 初始噪声，长度必须为 `MEL_DIM * targetLen`。 |
| `promptLen` | `number` | 是 | prompt 帧数；必须为正整数。 |
| `targetLen` | `number` | 是 | 要生成的目标帧数；必须为正整数。 |
| `nSteps` | `number` | 是 | 反向扩散步数；必须为正整数（典型值 `8`）。 |
| `seed` | `number` | 否 | RNG 种子。默认 `12345`，与 C++ 参考实现保持一致，以便跨实现做精度对比。 |

> 维度校验：`promptMel.length === 128 * promptLen`、`z.length === 128 * targetLen`、
> `cond.length === 1024 * (promptLen + targetLen)`，且 `promptLen / targetLen / nSteps` 均为正，
> 否则抛出 `TypeError`。

---

## 函数

### `loadModel(path, options?)`

异步加载一个 `Model`。

```ts
export function loadModel(path: string, options?: LoadOptions): Promise<Model>;
```

#### 参数

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `path` | `string` | 是 | `.gguf` 模型文件的绝对或相对路径。 |
| `options` | `LoadOptions` | 否 | 后端选择；省略时默认 `{ backend: 'cpu' }`。 |

#### 返回值

`Promise<Model>` —— 解析为已加载的 `Model` 实例。底层原生加载本身是同步的；`Promise` 包装用于
符合人体工程学的 `await` 与 `try/catch` 用法，并为将来可能的异步加载保持接口稳定。

#### 抛出错误

- `TypeError`：`options.backend` 不是 `'cpu' | 'vulkan' | 'cuda'` 之一。
- `Error`：当前平台没有预编译二进制（如 macOS / Linux），错误信息会列出受支持平台。
- `Error`：请求的后端子包未安装；错误信息包含对应的 `npm install` 安装命令。
- `Error`：GGUF 文件加载失败（路径不存在、文件损坏、非 GGUF 等），来自原生层。

由于返回 `Promise`，这些错误会以 rejected promise 形式抛出，应使用 `try/await/catch` 捕获。

#### 示例

```js
const { loadModel } = require('soulx-singer-dit');

async function main() {
  try {
    const model = await loadModel('./model.gguf', { backend: 'cpu' });
    // ... 使用 model ...
    model.release();
  } catch (err) {
    console.error('加载失败:', err.message);
  }
}

main();
```

---

### `getVersion()`

返回绑定版本字符串。

```ts
export function getVersion(): string;
```

#### 参数

无。

#### 返回值

`string` —— 例如 `"0.1.0"`。该值由 JS shim 维护，与原生绑定报告的版本保持同步。

#### 抛出错误

无。本函数**不**会触发任何 `.node` 的 `dlopen`，因此即使没有任何后端子包被安装，也可安全用于诊断。

#### 示例

```js
const { getVersion } = require('soulx-singer-dit');
console.log(getVersion());   // "0.1.0"
```

---

### `listBackends()`

返回当前平台已安装且可解析的后端列表。

```ts
export function listBackends(): Backend[];
```

#### 参数

无。

#### 返回值

`Backend[]` —— 例如 `['cpu']` 或 `['cpu', 'vulkan', 'cuda']`。若当前平台没有任何预编译后端（如
macOS / Linux），返回 `[]`。

#### 说明

本调用**不会** `dlopen` 任何 `.node` 文件；它仅通过 `require.resolve` 检查各候选子包是否可解析。
已被加载过的后端会走缓存直接返回。可以安全地用于在尝试加载模型前探测可用后端。

#### 抛出错误

无。

#### 示例

```js
const { listBackends } = require('soulx-singer-dit');

const backends = listBackends();
console.log('可用后端:', backends);

if (backends.length === 0) {
  console.error('当前平台未安装任何预编译后端');
}
```

---

## 类 `Model`

一个已加载的 SoulX-Singer-DiT 模型。

### `constructor(path, options?)`

```ts
class Model {
  constructor(path: string, options?: LoadOptions);
}
```

#### 参数

| 参数 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `path` | `string` | 是 | `.gguf` 模型文件路径。 |
| `options` | `LoadOptions` | 否 | 后端选择；省略时默认 `{ backend: 'cpu' }`。 |

#### 返回值

返回一个原生 `Model` 实例（其原型上带有 `forward` / `reverseDiffusion` / `release`）。
> 注意：shim 中的 `Model` 是一个代理类，构造时解析后端并委托给原生构造函数返回真实原生实例。
> 调用方应依赖文档化的方法表面，而**不要**依赖 `instanceof Model` 判断。

#### 抛出错误

- `TypeError`：`path` 不是字符串，或 `options.backend` 非法。
- `Error`：平台无预编译二进制 / 后端子包未安装（与 `loadModel` 相同）。
- `Error`：原生层加载 GGUF 失败（`"Failed to load GGUF model: <path>"`）。

> 由于构造函数是同步的，这些错误会同步抛出。若希望用 `await`/`try/catch` 处理，请改用 `loadModel`。

#### 示例

```js
const { Model } = require('soulx-singer-dit');

let model;
try {
  model = new Model('./model.gguf', { backend: 'cpu' });
} catch (err) {
  console.error('构造失败:', err.message);
  process.exit(1);
}

// ... 使用 model ...
model.release();
```

---

### `model.forward(opts)`

执行单次前向传播。

```ts
forward(opts: ForwardOptions): Float32Array;
```

#### 参数

见 [`ForwardOptions`](#forwardoptions)：`x`、`cond`、`t`、`T`。

#### 返回值

`Float32Array`，长度为 `MEL_DIM * T`（即 `128 * T`），保存预测的速度场（flow-matching velocity）。

#### 抛出错误

- `Error`：模型未加载（`"Model is not loaded"`，通常在 `release()` 之后调用时出现）。
- `TypeError`：缺少参数对象、缺少必填字段、字段类型错误。
- `TypeError`：`T <= 0`，或 `x.length !== 128 * T`，或 `cond.length !== 1024 * T`。

#### 示例

```js
const { MEL_DIM, HIDDEN } = require('soulx-singer-dit');

const T = 32;
const x    = new Float32Array(MEL_DIM * T);
const cond = new Float32Array(HIDDEN * T);
// 在此处填充实际数据 ...

const velocity = model.forward({ x, cond, t: 0.5, T });
console.log('输出长度:', velocity.length);   // 128 * 32 = 4096
```

---

### `model.reverseDiffusion(opts)`

执行反向扩散采样，生成目标 mel。

```ts
reverseDiffusion(opts: ReverseDiffusionOptions): Float32Array;
```

#### 参数

见 [`ReverseDiffusionOptions`](#reversediffusionoptions)：`promptMel`、`cond`、`z`、`promptLen`、
`targetLen`、`nSteps`、`seed?`。

#### 返回值

`Float32Array`，长度为 `MEL_DIM * targetLen`（即 `128 * targetLen`），保存生成的目标 mel 频谱。

#### 说明

内部实现为 flow matching 的 Euler ODE 积分：

1. `xt` 初始化为 `z`。
2. 对 `i = 0 .. nSteps-1`，令 `t = (i + 0.5) / nSteps`，将 `promptMel` 与 `xt` 拼接为
   `[MEL_DIM, promptLen + targetLen]`，调用前向得到速度场。
3. 取速度场中 `targetLen` 部分，按 `xt = xt + flow_pred * (1/nSteps)` 更新。
4. 返回最终 `xt`。

`seed` 默认为 `12345`，与 C++ 参考实现一致，便于跨实现复现与精度对比。

#### 抛出错误

- `Error`：模型未加载。
- `TypeError`：缺少参数对象、缺少必填字段、字段类型错误。
- `TypeError`：`promptLen / targetLen / nSteps` 非正，或任一数组长度与公式不符。

#### 示例

```js
const { MEL_DIM, HIDDEN } = require('soulx-singer-dit');

const promptLen = 16;
const targetLen = 48;
const T = promptLen + targetLen;

const promptMel = new Float32Array(MEL_DIM * promptLen);
const cond      = new Float32Array(HIDDEN * T);
const z         = new Float32Array(MEL_DIM * targetLen);

const mel = model.reverseDiffusion({
  promptMel, cond, z,
  promptLen, targetLen,
  nSteps: 8,
  // seed: 12345,   // 可省略，默认即 12345
});
console.log('生成 mel 长度:', mel.length);   // 128 * 48 = 6144
```

---

### `model.release()`

立即释放原生资源。

```ts
release(): void;
```

#### 参数

无。

#### 返回值

`undefined`（`void`）。

#### 说明

调用后该实例不再可用，再次调用 `forward` / `reverseDiffusion` 会抛出 `"Model is not loaded"`。
重复调用 `release()` 是 no-op，不会报错。即使不显式调用，原生资源也会在实例被垃圾回收时由析构函数
释放；`release()` 仅用于希望尽早归还内存的场景（如长驻服务中频繁加载/卸载模型）。

#### 抛出错误

无。

#### 示例

```js
const model = await loadModel('./model.gguf');
// ... 使用 ...
model.release();          // 立即释放
model.release();          // 再次调用，no-op，安全
```

---

### `Model.loadModel(path, options?)`

静态便捷工厂，镜像原生绑定上的同名方法。

```ts
static loadModel(path: string, options?: LoadOptions): Model;
```

#### 参数

同 [`loadModel`](#loadmodelpath-options) 顶层函数。

#### 返回值

`Model`（**同步**返回，非 Promise）。这是原生层暴露的同步工厂；底层加载是同步的。

#### 说明

> 注意：shim 暴露的顶层 `loadModel` 返回 `Promise<Model>`，而 `Model.loadModel`（静态方法）
> 同步返回 `Model`。如需 `await` 与 `try/catch` 体验，推荐使用顶层 `loadModel`。

#### 抛出错误

与构造函数相同：`TypeError`（参数非法）、`Error`（平台/后端/GGUF 问题），同步抛出。

#### 示例

```js
const { Model } = require('soulx-singer-dit');

const model = Model.loadModel('./model.gguf', { backend: 'cpu' });
const v = model.forward({ x, cond, t: 0.5, T });
model.release();
```

---

## 常量

### `MEL_DIM`

```ts
export const MEL_DIM: 128;
```

mel 维度。输入/输出 mel 数组以此作为前导维度布局。镜像 C++ 侧的 `MEL_DIM`。

```js
const { MEL_DIM } = require('soulx-singer-dit');
// MEL_DIM === 128
```

---

### `HIDDEN`

```ts
export const HIDDEN: 1024;
```

条件流的 hidden 维度。条件数组以此作为前导维度布局。镜像 C++ 侧的 `HIDDEN`。

```js
const { HIDDEN } = require('soulx-singer-dit');
// HIDDEN === 1024
```

---

## 错误处理速查

| 场景 | 错误类型 | 典型信息 |
| --- | --- | --- |
| 后端名非法 | `TypeError` | `options.backend must be one of 'cpu', 'vulkan', or 'cuda'` |
| 平台无预编译二进制 | `Error` | `soulx-singer-dit has no prebuilt binary for platform "darwin-arm64"...` |
| 后端子包未安装 | `Error` | `failed to load backend "cuda" ... Install it with: npm install ...` |
| GGUF 加载失败 | `Error` | `Failed to load GGUF model: /path/to/model.gguf` |
| `forward` 维度不符 | `TypeError` | `forward: x.length must equal MEL_DIM * T (128 * T)` |
| `forward` 后调用 | `Error` | `Model is not loaded` |
| `reverseDiffusion` 参数缺失 | `TypeError` | `missing field: promptMel` |

更完整的错误处理示例见 [示例集 — 错误处理](EXAMPLES.md#6-错误处理)。
