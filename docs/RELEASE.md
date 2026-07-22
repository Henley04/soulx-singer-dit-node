# 发布流程

本文档说明 `soulx-singer-dit` 及其三个平台子包的发布流程，重点是如何配置 **npm OIDC Trusted
Publisher**（无需长期 NPM_TOKEN，并附带 provenance 来源签名）。

发布涉及 4 个 npm 包：

| 包名 | 说明 |
| --- | --- |
| `soulx-singer-dit` | 主包（JS shim + 类型 + 文档）。 |
| `soulx-singer-dit-win32-x64-cpu` | Windows x64 CPU 预编译后端。 |
| `soulx-singer-dit-win32-x64-vulkan` | Windows x64 Vulkan 预编译后端。 |
| `soulx-singer-dit-win32-x64-cuda` | Windows x64 CUDA 预编译后端。 |

---

## 目录

1. [发布流程概览](#1-发布流程概览)
2. [首次包创建](#2-首次包创建)
3. [配置 npm OIDC Trusted Publisher](#3-配置-npm-oidc-trusted-publisher)
4. [发布新版本](#4-发布新版本)
5. [版本管理策略](#5-版本管理策略)
6. [验证发布](#6-验证发布)
7. [回滚](#7-回滚)

---

## 1. 发布流程概览

发布由推送版本 tag 触发，全自动完成构建与发布，无需人工干预（首次配置除外）。流程如下：

```
推送 v0.1.0 tag
       │
       ▼
[build-prebuilt.yml]   在 windows-latest 上构建 cpu/vulkan/cuda 三个 .node
       │                （矩阵：每后端一条 job，含 Vulkan SDK / CUDA 安装）
       ▼
[collect job]          合并三个 .node 到 packages/ 目录，打包为 prebuilt-packages artifact
       │
       ▼ (workflow_run: completed == success)
[publish.yml]          下载 artifact -> 放入子包目录 ->
       │                npm publish --provenance（4 个包）-> 创建 GitHub Release v0.1.0
       ▼
完成：npm 上 4 个包更新 + GitHub Release 发布
```

关键点：

- **无长期 token**：`publish.yml` 中**没有**任何 `NPM_TOKEN` / `NODE_AUTH_TOKEN` secret。认证完全
  依赖 npm 的 OIDC Trusted Publisher，由 workflow 的 `id-token: write` 权限换取 OIDC claim。
- **provenance**：每个包都用 `--provenance` 发布，npm 网站会显示来源签名（构建出处可验证）。
- **顺序保证**：`publish.yml` 通过 `workflow_run` 事件监听 `build-prebuilt.yml` 的完成，仅当
  `conclusion == 'success'` 时才执行发布。

---

## 2. 首次包创建

npm OIDC Trusted Publisher 要求目标包**已存在**于 npm（即便是空骨架）。但首次发布时包尚不存在，
形成「鸡生蛋」问题。解决方法：用一个**一次性的 npm granular access token** 手动创建包骨架，
之后即可切换到 OIDC 模式。

### 2.1 生成 granular token

1. 登录 <https://www.npmjs.com>（用包所有者账号）。
2. 头像 → **Access Tokens** → **Generate New Token** → 选择 **Granular Access Token**。
3. 配置：
   - **Token name**：`soulx-first-publish`（临时用）。
   - **Expiration**：选短一些，如 7 天。
   - **Packages and scopes**：选 **Only select packages and scopes**，填入要创建的包名
     （如 `soulx-singer-dit`、`soulx-singer-dit-win32-x64-cpu` 等）。
     首次创建时包不存在，可改为 **Read and write** 对应 scope。
   - **Permissions**：Read and write。
4. 生成后立即复制 token（只显示一次）。

### 2.2 首次发布骨架

在本地（或一个临时 CI job）用该 token 发布一次：

```bash
# 临时写入 ~/.npmrc（发布后删除，不要提交到仓库）
echo "//registry.npmjs.org/:_authToken=YOUR_GRANULAR_TOKEN" >> ~/.npmrc

cd packages/win32-x64-cpu
# 确保对应 .node 已就位（可先放占位构建产物或从 CI artifact 下载）
npm publish --access public

cd ../win32-x64-vulkan
npm publish --access public

cd ../win32-x64-cuda
npm publish --access public

cd ../..
npm publish --access public

# 清理 token
npm token revoke soulx-first-publish   # 或在网页上 revoke
# 编辑 ~/.npmrc 删除 _authToken 行
```

> 这一步的目的是让 4 个包在 npm 上「存在」，从而能在网页上进入它们的设置页配置 Trusted Publisher。
> 首次发布可以不带 `--provenance`（granular token 不支持 provenance）；后续版本都用 OIDC + provenance。

---

## 3. 配置 npm OIDC Trusted Publisher

对**每个**包（共 4 个）都执行一次下面的配置。配置完成后，`publish.yml` 即可无需 token 自动发布。

### 3.1 进入包设置页

1. 登录 <https://www.npmjs.com>。
2. 打开对应包页面，例如 <https://www.npmjs.com/package/soulx-singer-dit>。
3. 点击 **Settings**（齿轮图标，仅包 owner 可见）。

### 3.2 添加 Trusted Publisher

1. 在 Settings 页找到 **Publishing access** 区域。
2. 选择 **Require provenance**（推荐开启，强制 provenance）。
3. 在 **Trusted Publishers** 下点击 **Add GitHub Action**。
4. 填入以下值（4 个包都填**相同**的值，因为它们由同一个 workflow 发布）：

   | 字段 | 值 |
   | --- | --- |
   | **Repository owner** | `Henley04` |
   | **Repository name** | `soulx-singer-dit-node` |
   | **Workflow filename** | `publish.yml` |
   | **Environment** | （可留空，或填如 `release`；若填了则 workflow 需在 publish job 上声明对应 `environment:`） |

5. 保存。

### 3.3 对 4 个包重复

依次对以下 4 个包重复 3.1 ~ 3.2：

- `soulx-singer-dit`
- `soulx-singer-dit-win32-x64-cpu`
- `soulx-singer-dit-win32-x64-vulkan`
- `soulx-singer-dit-win32-x64-cuda`

### 3.4 确认 workflow 权限

`publish.yml` 的 publish job 已声明（见仓库内文件）：

```yaml
permissions:
  id-token: write   # OIDC trusted publisher / provenance 必需
  contents: read
  actions: read     # 下载触发 workflow_run 的 artifact 必需
```

并且 `npm publish` 步骤**没有**设置 `NODE_AUTH_TOKEN` 环境变量——这是关键：一旦设置了该 env，
npm 会改用 token 认证而**绕过** OIDC 流程。

> 若你在 3.2 中填写了 Environment，则需在 job 上加 `environment: release`，否则 OIDC claim 中
> 不会包含 environment 字段，与 npm 侧配置不匹配会导致发布失败。

---

## 4. 发布新版本

日常发布完全自动化，只需三步：

### 4.1 修改版本号

在仓库根目录与三个子包目录中，把 `package.json` 的 `version` 改为新版本（4 处保持一致，见
[版本管理策略](#5-版本管理策略)）：

```bash
# 例如发布 0.2.0
# 编辑以下文件，把 "version": "0.1.0" 改为 "0.2.0"：
#   package.json
#   packages/win32-x64-cpu/package.json
#   packages/win32-x64-vulkan/package.json
#   packages/win32-x64-cuda/package.json
```

> 也建议同步更新 `index.js` 与 `src/binding.cc` 中硬编码的 `"0.1.0"` 版本字符串，使
> `getVersion()` 返回值一致。

### 4.2 提交并打 tag

```bash
git add package.json packages/*/package.json index.js src/binding.cc
git commit -m "release: v0.2.0"
git tag v0.2.0
git push origin main --tags
```

tag 命名必须匹配 `v*.*.*`（见 `build-prebuilt.yml` 的 `on.push.tags`）。

### 4.3 等待 CI 完成

推送 tag 后：

1. `build-prebuilt.yml` 自动触发，在 Windows runner 上构建三个后端的 `.node`，收集为 artifact。
2. 构建成功后，`publish.yml` 由 `workflow_run` 事件触发，下载 artifact 并 `npm publish --provenance`
   发布 4 个包，随后用 `softprops/action-gh-release` 创建 GitHub Release `v0.2.0`（自动生成
   release notes）。

可在 GitHub 仓库的 **Actions** 标签页观察两个 workflow 的运行状态。若失败，定位到对应 job 查看日志
（常见原因见 [BUILD.md — 故障排查](BUILD.md#8-故障排查--troubleshooting)）。

### 4.4 手动触发（可选）

若需用已存在的 artifact 重新发布（例如 `publish.yml` 本身失败需重试），可在 Actions 页面手动
dispatch `publish.yml`（workflow 已声明 `workflow_dispatch`）。它 会取最近的 `prebuilt-packages`
artifact 重新发布。也可手动 dispatch `build-prebuilt.yml` 重新构建。

---

## 5. 版本管理策略

- **语义化版本（SemVer）**：`MAJOR.MINOR.PATCH`。
  - `PATCH`：bug 修复、构建脚本调整、文档更新（向后兼容）。
  - `MINOR`：新增 API、性能改进、支持新模型格式（向后兼容）。
  - `MAJOR`：破坏性 API 变更（如方法签名改变、常量值改变）。
- **子包与主包同步**：三个平台子包的版本号始终与主包一致，即便某次发布只改了主包的 JS shim。
  这样 `optionalDependencies` 中的 `"^0.1.0"` 范围能正确解析到配套的子包，避免主包与子包 ABI
  不匹配。
- **预发布**：如需发预发布版，可用 `0.2.0-rc.1` 这样的 tag，对应 git tag `v0.2.0-rc.1`
  （注意 `v*.*.*` 的 glob 不含 `-`，需调整 workflow 的 tag 匹配模式或用 `workflow_dispatch` 手动触发）。
- **不要跳过版本号**：保持单调递增，避免 npm 上的版本空洞给下游造成困惑。

---

## 6. 验证发布

### 6.1 检查 provenance / 来源签名

发布成功后，每个包页面（如 <https://www.npmjs.com/package/soulx-singer-dit>）会显示
**Provenance** 徽标，点开可查看 Sigstore 签名的出处声明（仓库、commit、workflow、runner 等）。

命令行验证（需要 `npm@9.5+`）：

```bash
npm view soulx-singer-dit dist.attestation   # 查看 attestation 元数据
npm audit signatures                          # 校验已安装包的签名
```

也可用 `sigstore` 工具链做完整验证：

```bash
npm install -g @sigstore/verify
sigstore verify github:soulx-singer-dit-node soulx-singer-dit@0.2.0
```

### 6.2 在 npm 网站查看

- 包页面：`https://www.npmjs.com/package/<包名>`
- 版本列表：包页面右侧 **Versions** 标签。
- 确认 4 个包的新版本号一致，且都带 Provenance 徽标。

### 6.3 实际安装验证

在一个干净的临时目录里安装并冒烟测试：

```bash
mkdir /tmp/verify && cd /tmp/verify
npm install soulx-singer-dit
node -e "const s=require('soulx-singer-dit'); console.log(s.getVersion()); console.log(s.listBackends());"
# 期望输出版本号与可用后端列表
```

---

## 7. 回滚

npm 对已发布版本的处理有严格限制，回滚策略如下：

### 7.1 unpublish 限制

- npm 仅允许在发布后 **72 小时内** `unpublish`，且要求该版本**没有任何下游依赖者**。
- 超过 72 小时，或已有包依赖该版本，**无法** unpublish（npm 政策，不可绕过）。
- 即使能 unpublish，也**强烈不建议**对已被人安装的版本使用，会破坏下游。

```bash
# 仅在 72 小时内、且确认无下游依赖时：
npm unpublish soulx-singer-dit@0.2.0       # 撤销特定版本
# 或撤销整个包（更危险，仅在包刚创建且无人使用时）：
# npm unpublish soulx-singer-dit --force
```

### 7.2 推荐方案：deprecate

对有问题的版本打上弃用标记，指引用户升级或回退到稳定版本，而**不**删除文件：

```bash
npm deprecate soulx-singer-dit@0.2.0 "已知问题：xxx，请使用 0.2.1 或回退到 0.1.0"
npm deprecate soulx-singer-dit-win32-x64-cpu@0.2.0 "同上"
npm deprecate soulx-singer-dit-win32-x64-vulkan@0.2.0 "同上"
npm deprecate soulx-singer-dit-win32-x64-cuda@0.2.0 "同上"
```

deprecate 后，用户安装该版本时 npm 会显示弃用警告，但已安装的仍可用。

### 7.3 发修复版本

最稳妥的回滚方式是**发一个修复版本**而非撤销：

1. 在 git 上回退到上一个稳定 commit（或基于其开新分支修 bug）。
2. 递增 `PATCH` 版本号（如 `0.2.0` → `0.2.1`）。
3. 走正常的 [发布新版本](#4-发布新版本) 流程。

### 7.4 撤回 GitHub Release

GitHub Release（由 `publish.yml` 创建）可随时在仓库的 **Releases** 页面删除或转为 draft，
这与 npm 包是独立的。删除 GitHub Release **不会**影响 npm 上已发布的包。

```bash
gh release delete v0.2.0 --yes   # 可选：同时删 tag
gh release delete v0.2.0 --cleanup-tag --yes
```
