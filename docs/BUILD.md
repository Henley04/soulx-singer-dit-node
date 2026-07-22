# 构建指南

本指南说明如何在本机从源码构建 `soulx-singer-dit` 的原生 `.node` 二进制。预编译二进制仅覆盖
`win32-x64`；若你在 macOS / Linux 上使用，或希望针对自己的环境做定制编译，请按本文操作。

> 本绑定的原生层是纯 C++（`src/binding.cc` + `src/infer.cpp`），通过 N-API 暴露给 Node.js。
> 量化 kernel 来自 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的 `ggml` 子库，静态链接进
> 最终的 `.node`，因此产物除系统共享库（如 OpenMP、Vulkan loader、CUDA runtime）外是自包含的。

---

## 目录

1. [环境要求](#1-环境要求)
2. [克隆仓库](#2-克隆仓库)
3. [安装依赖](#3-安装依赖)
4. [CPU 构建](#4-cpu-构建)
5. [Vulkan 构建](#5-vulkan-构建)
6. [CUDA 构建](#6-cuda-构建)
7. [构建产物](#7-构建产物)
8. [故障排查 / Troubleshooting](#8-故障排查--troubleshooting)
9. [CI 构建](#9-ci-构建)

---

## 1. 环境要求

| 组件 | 版本要求 | 说明 |
| --- | --- | --- |
| 操作系统 | Windows 10 / 11（x64） | 预编译与 CI 均基于 Windows。macOS/Linux 理论可编译，但未提供预编译二进制。 |
| CMake | 3.18 及以上 | `CMakeLists.txt` 顶部 `cmake_minimum_required(VERSION 3.18)`。 |
| 编译器 | Visual Studio 2022（MSVC，含 C++ 工具集） | 需要 C++17 支持。也需安装 Windows SDK。 |
| Node.js | 16 及以上 | `package.json` 声明 `"engines": { "node": ">=16" }`。需要开发头文件（见下）。 |
| Git | 任意现代版本 | 用于克隆 llama.cpp 源码。 |
| Ninja（可选但推荐） | 任意 | 比 MSBuild 默认生成器快得多；CI 使用 Ninja。 |

**Node.js 开发头文件**：CMake 需要 `node_api.h` 等头文件。最可靠的获取方式是运行
`npx node-gyp install`，它会下载与当前 `node` 版本严格匹配的头文件到 `~/.node-gyp/<version>/include/node`。
随后将 `NODE_INCLUDE_DIR` 环境变量指向该目录（见下文各构建步骤）。

校验环境（PowerShell 示例）：

```powershell
cmake --version          # >= 3.18
node --version           # >= 16
ninja --version          # 可选
git --version
```

---

## 2. 克隆仓库

```bash
git clone https://github.com/Henley04/soulx-singer-dit-node.git
cd soulx-singer-dit-node
```

---

## 3. 安装依赖

### 3.1 安装 `node-addon-api`

本绑定通过 [node-addon-api](https://github.com/nodejs/node-addon-api)（C++ 包装 N-API）与 Node.js 交互。
`CMakeLists.txt` 会在 `node_modules/node-addon-api/` 下查找 `napi.h`。

```bash
npm install node-addon-api
```

> 若只构建原生插件、不需要主包的其余依赖，单独执行上面这一条即可（如 CI 所做）。
> 若你同时在开发主包，可执行 `npm install`（会触发 `scripts/postinstall.js`，仅打印诊断信息，不会阻塞）。

### 3.2 克隆 llama.cpp 源码

`CMakeLists.txt` 默认在 `vendor/llama.cpp` 查找 llama.cpp 源码树（需包含 `CMakeLists.txt`）。

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git vendor/llama.cpp
```

> 也可通过 `-DLLAMA_CPP_DIR=/path/to/llama.cpp` 指向已有副本。
> `--depth 1` 只取最新一次提交，节省时间与磁盘。

---

## 4. CPU 构建

最简单、无额外系统依赖的构建方式。ggml 的 CPU backend 默认开启（`GGML_CPU=ON`）。

### 4.1 准备 Node 头文件

```powershell
npx node-gyp install
$nodeVersion = (node -v).TrimStart('v')
$env:NODE_INCLUDE_DIR = "$env:USERPROFILE\.node-gyp\$nodeVersion\include\node"
# 备选位置（不同 node-gyp 版本可能缓存到 LOCALAPPDATA）：
#   $env:NODE_INCLUDE_DIR = "$env:LOCALAPPDATA\node-gyp\Cache\$nodeVersion\include\node"
echo $env:NODE_INCLUDE_DIR   # 确认指向含 node_api.h 的目录
```

### 4.2 配置 MSVC 环境

在「x64 Native Tools Command Prompt for VS 2022」中打开终端，或用 `vcvarsall.bat`：

```powershell
# 例如（路径按实际安装调整）：
& "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
```

或使用 `ilammy/msvc-dev-cmd` Action 在 CI 中等效完成。

### 4.3 配置并编译

```powershell
cmake -B build -G "Ninja" `
  -DCMAKE_BUILD_TYPE=Release `
  -DGGML_CPU=ON -DGGML_VULKAN=OFF -DGGML_CUDA=OFF `
  -DNODE_INCLUDE_DIR="$env:NODE_INCLUDE_DIR"

cmake --build build --config Release --parallel
```

> 若未安装 Ninja，可省略 `-G "Ninja"`，CMake 会回退到默认生成器（Visual Studio / MSBuild），
> 此时 `--config Release` 决定构建配置。

### 4.4 验证

```powershell
node -e "require('./build/soulx_singer_dit.node'); console.log('LOAD OK')"
```

应输出 `LOAD OK`。若失败，见 [故障排查](#8-故障排查--troubleshooting)。

---

## 5. Vulkan 构建

在 CPU 基础上额外开启 ggml 的 Vulkan backend，用于跨厂商 GPU 加速。

### 5.1 额外依赖：Vulkan SDK

安装 [LunarG Vulkan SDK](https://vulkan.lunarg.com/)（最新版本即可）。SDK 提供 `glslang`、
Vulkan headers 与 loader，CMake 通过 `VULKAN_SDK` 环境变量定位。

校验：

```powershell
echo $env:VULKAN_SDK
vulkaninfo --summary    # 可选，确认驱动可用
```

### 5.2 配置并编译

```powershell
cmake -B build -G "Ninja" `
  -DCMAKE_BUILD_TYPE=Release `
  -DGGML_CPU=ON -DGGML_VULKAN=ON -DGGML_CUDA=OFF `
  -DNODE_INCLUDE_DIR="$env:NODE_INCLUDE_DIR"

cmake --build build --config Release --parallel
```

> 注意：CI 中 Vulkan 构建同时保留 `GGML_CPU=ON`，以便在无 GPU 的环境里也能跑冒烟测试；
> 实际推理时 ggml 会根据设备可用性选择 Vulkan 设备。

---

## 6. CUDA 构建

在 CPU 基础上额外开启 ggml 的 CUDA backend，仅适用于 NVIDIA GPU。

### 6.1 额外依赖：CUDA Toolkit

安装 [CUDA Toolkit 12.x](https://developer.nvidia.com/cuda-toolkit)，需要 `nvcc`、`cudart`、
`cublas`、`curand`。CI 中固定使用 `12.4.0`。

校验：

```powershell
nvcc --version
echo $env:CUDA_PATH     # 通常由安装程序设置，如 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
```

### 6.2 配置并编译

```powershell
cmake -B build -G "Ninja" `
  -DCMAKE_BUILD_TYPE=Release `
  -DGGML_CPU=ON -DGGML_VULKAN=OFF -DGGML_CUDA=ON `
  -DNODE_INCLUDE_DIR="$env:NODE_INCLUDE_DIR"

cmake --build build --config Release --parallel
```

> 编译 CUDA backend 耗时显著长于 CPU/Vulkan（需要 `nvcc` 编译大量 kernel）。`--parallel` 有助加速。

---

## 7. 构建产物

CMake 始终输出一个名为 `soulx_singer_dit.node` 的共享库（`CMakeLists.txt` 中
`OUTPUT_NAME "soulx_singer_dit"`，`SUFFIX ".node"`）。位置取决于生成器：

| 生成器 | 默认产物路径 |
| --- | --- |
| Ninja | `build/soulx_singer_dit.node` |
| Visual Studio / MSBuild | `build/Release/soulx_singer_dit.node` |

CI 在构建后会将其**重命名**为后端专属文件名并放入对应子包目录，以便 JS 层按后端加载：

| 后端 | 目标路径 |
| --- | --- |
| cpu | `packages/win32-x64-cpu/cpu.node` |
| vulkan | `packages/win32-x64-vulkan/vulkan.node` |
| cuda | `packages/win32-x64-cuda/cuda.node` |

本地构建若想被主包的 JS shim 加载，可手动复制并重命名：

```powershell
# 以 CPU 为例
Copy-Item build\soulx_singer_dit.node packages\win32-x64-cpu\cpu.node -Force
# 然后即可在仓库根目录运行：
node -e "const {getVersion}=require('./packages/win32-x64-cpu'); console.log(getVersion())"
```

---

## 8. 故障排查 / Troubleshooting

### (1) 找不到 `node-addon-api`

**现象**：CMake 配置阶段报错
`node-addon-api not found at '.../node_modules/node-addon-api'. Run npm install first...`

**原因**：`node_modules/node-addon-api/napi.h` 不存在。

**解决**：在仓库根目录执行 `npm install node-addon-api`，或通过
`-DNODE_ADDON_API_DIR=/path/to/node-addon-api` 显式指定路径。

---

### (2) 找不到 `Node.h` / `node_api.h`

**现象**：编译时报 `fatal error C1083: 无法打开包括文件: "node_api.h"` 或
`"Cannot find Node.js headers automatically"`。

**原因**：`actions/setup-node` 只装 Node 二进制，不装开发头文件；本地 `node` 安装也可能不带 `include/`。

**解决**：运行 `npx node-gyp install` 下载匹配头文件，并将 `NODE_INCLUDE_DIR` 指向：
`~/.node-gyp/<version>/include/node`（或 `LOCALAPPDATA\node-gyp\Cache\<version>\include\node`），
或传 `-DNODE_INCLUDE_DIR=...` / `-DNODE_INCLUDE_DIRS=...`。校验该目录下存在 `node_api.h`。

---

### (3) `llama.cpp` 子目录缺失

**现象**：CMake 配置阶段报错
`llama.cpp source not found at '.../vendor/llama.cpp'. Populate it first...`

**原因**：`vendor/llama.cpp/CMakeLists.txt` 不存在（未克隆子模块）。

**解决**：

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git vendor/llama.cpp
```

或用 `-DLLAMA_CPP_DIR=/path/to/llama.cpp` 指向已有副本（需含 `CMakeLists.txt` 与 `ggml/`）。

---

### (4) Vulkan SDK 未安装 / 找不到

**现象**：Vulkan 构建时 CMake 报 `Could NOT find Vulkan`，或编译期缺少 `vulkan/vulkan.h`。

**原因**：未安装 Vulkan SDK，或 `VULKAN_SDK` 环境变量未设置。

**解决**：从 <https://vulkan.lunarg.com/> 安装 Vulkan SDK，安装后确认 `echo $env:VULKAN_SDK` 非空，
并重启终端让环境变量生效。CI 通过 `humbletim/install-vulkan-sdk` Action 完成。

---

### (5) CUDA `nvcc` 找不到

**现象**：CUDA 构建时 CMake 报 `Could NOT find CUDA` / `nvcc` not found，或 `nvcc --version` 失败。

**原因**：未安装 CUDA Toolkit，或 `CUDA_PATH` / `PATH` 未包含 `nvcc`。

**解决**：安装 CUDA Toolkit 12.x，确认 `nvcc --version` 可用。若已安装但仍找不到，手动设置：
`$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"`，并将
`%CUDA_PATH%\bin` 加入 `PATH`。CI 通过 `Jimver/cuda-toolkit` Action 完成（network 安装方式）。

---

### (6) MSVC 编译器未配置

**现象**：`cmake` 配置时报 `No CMAKE_CXX_COMPILER could be found`，或 `cl` 不是内部命令。

**原因**：未在 MSVC 环境的终端中运行；`cl.exe` 不在 `PATH`。

**解决**：使用「x64 Native Tools Command Prompt for VS 2022」打开终端，或运行 `vcvars64.bat`。
确认 `where cl` 能找到编译器。CI 通过 `ilammy/msvc-dev-cmd@v1` 完成等效配置。

---

### (7) 链接错误：找不到 ggml 符号

**现象**：链接阶段报 `unresolved external symbol ggml_mul_mat` / `ggml_init` 等，或
`.node` 加载时 `Module did not self-register` / 找不到符号。

**原因**：`target_link_libraries(... ggml)` 未生效，或 llama.cpp 的 `BUILD_SHARED_LIBS` 未关，
或构建顺序导致 ggml 未先编译。

**解决**：
- 确认 `CMakeLists.txt` 中 `add_subdirectory(${LLAMA_CPP_DIR} ...)` 在
  `add_library(soulx_singer_dit ...)` **之前**（本项目已如此排列）。
- 确认 `BUILD_SHARED_LIBS=OFF`（本项目 `FORCE` 设置），ggml 应静态链接进 `.node`。
- 若手动改过选项，清理 `build/` 后重新配置：`Remove-Item -Recurse -Force build` 再 `cmake -B build ...`。

---

### (8) `.node` 加载失败：模块不匹配 ABI / 架构

**现象**：`node -e "require('./build/soulx_singer_dit.node')"` 报
`The module was compiled against a different version of Node.js` 或 `not a valid Win32 application`。

**原因**：用错误 Node 版本的头文件编译，或在 32-bit 终端里构建了 32-bit 产物。

**解决**：
- 用 `npx node-gyp install` 安装与**当前运行 `node`** 严格匹配版本的头文件。
- 确保使用 **x64** 的 MSVC 工具链（`vcvars64.bat`，而非 `vcvars32.bat`）。
- CI 中通过 `actions/setup-node` 固定 Node 20 并匹配头文件。

---

## 9. CI 构建

本仓库的 CI 由三个 GitHub Actions workflow 组成（位于 `.github/workflows/`）：

| Workflow | 触发 | 作用 |
| --- | --- | --- |
| `ci.yml` | push/PR 到 main | 校验 `package.json` JSON 合法性、所有 `.js` 语法、workflow YAML 语法。不编译原生插件。 |
| `build-prebuilt.yml` | 推送 `v*.*.*` tag / 手动 | 在 `windows-latest` 上用矩阵构建 cpu/vulkan/cuda 三种 `.node`，校验可加载，收集为 `prebuilt-packages` artifact。 |
| `publish.yml` | 由 `build-prebuilt.yml` 完成后触发 / 手动 | 下载 artifact，把 `.node` 放入子包目录，用 `npm publish --provenance` 发布 4 个包并创建 GitHub Release。 |

CI 构建的关键步骤与本地一致：安装 `node-addon-api` → 克隆 llama.cpp → 配置 MSVC / Vulkan SDK / CUDA →
`npx node-gyp install` 取头文件 → `cmake` 配置（Ninja）→ `cmake --build` → 重命名 `.node` →
`node -e "require(...)"` 验证可加载。

发布流程（含 npm OIDC Trusted Publisher 配置）详见 [RELEASE.md](RELEASE.md)。
